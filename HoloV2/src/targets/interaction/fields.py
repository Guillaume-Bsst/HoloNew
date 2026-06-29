"""eval_fields — evaluate a posed cloud against ALL channels -> ``MultiChannelField`` (channel-first
``(C, P)``). The clouds x channels matrix runs the SAME op for the human cloud AND each object cloud.

For each ``Channel``: a channel with ``object_idx is None`` is the static GROUND (world frame); a
channel with ``object_idx=i`` maps the points into object i's local frame via that object's per-frame
``(R, t)``. Then the channel's ``sdf`` is sampled by trilinear interpolation — ONE path, no
flat-ground special case (the flat ground is an exact plane SDF) — and the contact direction is
reconstructed from the trilinearly interpolated witness (stable at box edges/corners). Pure,
array-oriented (axis = points), torch-free. Ported from HoloNew ``contact/contact_field`` +
``contact/combined`` + ``backends/floor``.

The matrix diagonal (an OBJECT cloud vs its OWN channel) is the one exception: there the cloud sits on
its own surface, a degenerate self-contact, so ``self_idx`` short-circuits it to the closed form
(distance 0, witness = the point itself) WITHOUT sampling the SDF — keeping the ``(C, P)`` layout
homogeneous while letting the downstream solve ignore that diagonal cheaply.
"""
from __future__ import annotations

import numpy as np

from ..contracts import MultiChannelField
from ...prepare.contracts import Channel, SDF


def eval_fields(points: np.ndarray, channels: tuple[Channel, ...], object_rot: np.ndarray,
                object_pos: np.ndarray, margin: float, self_idx: int | None = None) -> MultiChannelField:
    """``(C, P)`` field of ``points`` (``(P, 3)`` world) vs every ``Channel``.

    ``object_rot (N, 3, 3)`` / ``object_pos (N, 3)`` are the per-frame object world transforms (from
    ``FramePose``, reused — no recompute): a channel with ``object_idx=i`` reads
    ``(object_rot[i], object_pos[i])`` as its frame, a channel with ``object_idx is None`` uses the
    world frame (ground). A probe is ``active`` within ``margin`` of the surface; inactive probes
    carry ``distance = +margin`` and zeroed direction/witness, so the output is homogeneous ``(C, P)``.

    ``self_idx`` is the object index of the cloud being evaluated (its OWN object), or ``None`` for a
    cloud that is no object (the human). When a channel's ``object_idx == self_idx`` the cloud sits on
    its OWN surface, so the field is closed-form — distance 0, witness = the point itself (object-local),
    zero normal, active everywhere — and is filled DIRECTLY without sampling that SDF (skipping the
    degenerate self round-trip). The diagonal of the object×channel matrix; the human side has none.
    """
    pts = np.asarray(points, np.float64)                            # (P, 3) world probe positions
    margin = float(margin)
    p = len(pts)

    dist_ch, dir_ch, wit_ch, act_ch = [], [], [], []
    for ch in channels:
        if ch.object_idx is None:
            probe = pts                                            # ground: local frame == world
        else:
            rot = np.asarray(object_rot[ch.object_idx], np.float64)  # (3, 3) object world rotation
            pos = np.asarray(object_pos[ch.object_idx], np.float64)  # (3,)   object world position
            probe = (pts - pos) @ rot                              # (P, 3) = R.T @ (p - t), object-local

        if self_idx is not None and ch.object_idx == self_idx:
            # Self channel: this cloud IS object ``self_idx``, so every probe lies on its own surface.
            # Closed-form fill (no SDF sample): distance 0, witness = the probe itself (object-local),
            # no contact normal, active everywhere (on-surface => within margin).
            dist_ch.append(np.zeros(p))                            # (P,)
            dir_ch.append(np.zeros((p, 3)))                       # (P, 3) no normal on self
            wit_ch.append(probe)                                  # (P, 3) own point, object-local
            act_ch.append(np.ones(p, bool))                       # (P,)
            continue

        dist, witness, in_grid = _sample(ch.sdf, probe)            # (P,), (P, 3), (P,) — local frame
        active = in_grid & (dist < margin)                         # (P,) within the band AND in grid

        delta = probe - witness                                    # (P, 3) surface -> point, channel frame
        norm = np.linalg.norm(delta, axis=1, keepdims=True)        # (P, 1)
        # Direction from the (interpolated) stored witness, NOT a distance gradient: a true unit
        # vector even at box edges/corners. The guarded denominator (1.0 where on-surface) avoids a
        # 0/0 warning; ``where`` still zeroes the direction where probe == witness (on-surface) and
        # absorbs the float32 witness round-trip residual there.
        den = np.where(norm > 1e-6, norm, 1.0)                     # (P, 1) safe divisor
        direction = np.where(norm > 1e-6, delta / den, 0.0)        # (P, 3) unit contact normal

        # Homogenise the (C, P) layout: inactive probes carry distance=+margin, zeroed dir/witness.
        dist_ch.append(np.where(active, dist, margin))             # (P,)
        dir_ch.append(np.where(active[:, None], direction, 0.0))   # (P, 3)
        wit_ch.append(np.where(active[:, None], witness, 0.0))     # (P, 3)
        act_ch.append(active)                                      # (P,)

    return MultiChannelField(
        distance=np.stack(dist_ch),                                # (C, P)
        direction=np.stack(dir_ch),                                # (C, P, 3)
        witness=np.stack(wit_ch),                                  # (C, P, 3)
        active=np.stack(act_ch),                                   # (C, P)
        channels=tuple(ch.name for ch in channels),               # (C,)
    )


def _sample(sdf: SDF, probe: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trilinear sample of an ``SDF`` at local ``probe`` (``(P, 3)``) -> ``(distance (P,), witness
    (P, 3), in_grid (P,))``, all in the SDF's local frame. Node ``(ix, iy, iz)`` sits at ``origin +
    spacing*(ix, iy, iz)``; the distance AND the stored witness grids are interpolated with the SAME
    8-corner weights. ``in_grid`` flags probes whose enclosing cell is fully inside the grid (both
    corners on every axis); out-of-grid probes are clamped only so the gather stays in bounds — the
    caller treats them as inactive. Float64 compute; vectorised over P (the 8-corner loop is fixed)."""
    shape = np.array(sdf.grid.shape)                               # (3,) nodes per axis
    g = (probe - sdf.origin) / sdf.spacing                         # (P, 3) continuous grid index
    i0 = np.floor(g).astype(np.int64)                             # (P, 3) lower corner
    in_grid = np.all((i0 >= 0) & (i0 < shape - 1), axis=1)         # (P,) both corners valid per axis
    t = g - i0                                                    # (P, 3) fractional offset in [0, 1)
    i0 = np.clip(i0, 0, shape - 2)                                # clamp so the 8-corner gather is in bounds
    ix, iy, iz = i0[:, 0], i0[:, 1], i0[:, 2]

    dist = np.zeros(len(probe), np.float64)                        # (P,)
    witness = np.zeros((len(probe), 3), np.float64)               # (P, 3)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (np.where(dx, t[:, 0], 1 - t[:, 0]) *
                     np.where(dy, t[:, 1], 1 - t[:, 1]) *
                     np.where(dz, t[:, 2], 1 - t[:, 2]))           # (P,) trilinear corner weight
                dist += w * sdf.grid[ix + dx, iy + dy, iz + dz]    # (P,)
                witness += w[:, None] * sdf.witness[ix + dx, iy + dy, iz + dz]  # (P, 3)
    return dist, witness, in_grid
