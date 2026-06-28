"""eval_fields — evaluate a posed cloud against ALL channels -> ``MultiChannelField`` (channel-first
``(C, P)``). The clouds x channels matrix runs the SAME op for the human cloud AND each object cloud.

For each ``Channel``: a channel with ``object_idx is None`` is the static GROUND (world frame); a
channel with ``object_idx=i`` maps the points into object i's local frame via that object's per-frame
``(R, t)``. Then the channel's ``sdf`` is sampled by trilinear interpolation — ONE path, no
flat-ground special case (the flat ground is an exact plane SDF) — and the contact direction is
reconstructed from the trilinearly interpolated witness (stable at box edges/corners). Pure,
array-oriented (axis = points), torch-free. Ported from HoloNew ``contact/contact_field`` +
``contact/combined`` + ``backends/floor``.
"""
from __future__ import annotations

import numpy as np

from ..contracts import MultiChannelField
from ...prepare.contracts import Channel


def eval_fields(points: np.ndarray, channels: tuple[Channel, ...], object_rot: np.ndarray,
                object_pos: np.ndarray, margin: float) -> MultiChannelField:
    """``(C, P)`` field of ``points`` (``(P, 3)`` world) vs every ``Channel``.

    ``object_rot (N, 3, 3)`` / ``object_pos (N, 3)`` are the per-frame object world transforms (from
    ``FramePose``, reused — no recompute): a channel with ``object_idx=i`` reads
    ``(object_rot[i], object_pos[i])`` as its frame, a channel with ``object_idx is None`` uses the
    world frame (ground). A probe is ``active`` within ``margin`` of the surface; inactive probes
    carry ``distance = +margin`` and zeroed direction/witness, so the output is homogeneous ``(C, P)``.
    """
    raise NotImplementedError
