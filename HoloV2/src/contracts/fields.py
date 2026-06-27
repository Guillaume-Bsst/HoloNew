"""Field-evaluation results + the per-channel signed-distance source.

``ContactField`` (one channel) / ``MultiChannelField`` (all channels, channel-first ``(C, P)``)
are the per-frame eval outputs; ``Channel`` binds an ``SDF`` to its pose; ``SDF`` is the grid.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContactField:
    """One cloud vs ONE channel, ONE frame. Inactive probes: distance=+margin, rest 0."""

    distance: np.ndarray   # (P,)    signed distance
    direction: np.ndarray  # (P, 3)  contact normal (surface -> point)
    witness: np.ndarray    # (P, 3)  nearest surface point
    active: np.ndarray     # (P,)    bool, within margin


@dataclass(frozen=True)
class MultiChannelField:
    """One cloud vs ALL channels, ONE frame. Channel-first, homogeneous (C = ground + N obj)."""

    distance: np.ndarray         # (C, P)
    direction: np.ndarray        # (C, P, 3)
    witness: np.ndarray          # (C, P, 3)
    active: np.ndarray           # (C, P) bool
    channels: tuple[str, ...]    # (C,) channel names

    def __post_init__(self) -> None:
        c = len(self.channels)
        for name in ("distance", "direction", "witness", "active"):
            got = getattr(self, name).shape[0]
            if got != c:
                raise ValueError(f"{name} has {got} channels, expected {c}")

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def n_points(self) -> int:
        return self.distance.shape[1]


@dataclass(frozen=True)
class Channel:
    """One evaluation channel = a signed-distance source + its per-frame pose binding. Makes the
    ground/object alignment EXPLICIT (no implicit N vs N+1 offset). EVERY channel carries an ``sdf``
    so the eval has a SINGLE trilinear path (homogeneous, no flat-ground special case); ``object_idx``
    only sets the pose binding:

    - ``object_idx is None`` => the static GROUND in the world frame. Its ``sdf`` is a plane grid by
      default (a plane is affine, so a tiny grid reproduces ``z`` EXACTLY) or a TERRAIN grid
      (stairs/slope/climbing).
    - ``object_idx`` set      => object ``object_idx``, its ``sdf`` posed by ``object_poses[object_idx][f]``."""

    name: str
    object_idx: int | None        # None = static ground (world) ; else index into object_poses/clouds
    sdf: "SDF"                     # the signed-distance grid (ground plane / terrain / object)


@dataclass(frozen=True)
class SDF:
    """Signed-distance grid of a rigid surface, in its local frame — for objects, terrain ground
    AND the flat ground (a plane is an affine field, so trilinear sampling reproduces it EXACTLY on a
    tiny grid; it is an ordinary SDF too, keeping every channel homogeneous — see ``build_plane_sdf``).

    Carries a WITNESS grid (nearest surface point per node) alongside the distance: the eval
    reconstructs the contact direction as ``normalize(probe - witness)`` from the trilinearly
    interpolated witness, which stays a true unit vector near sharp box edges/corners — where a
    finite-difference gradient of the distance grid is unstable. Sampled by trilinear interpolation
    in the eval (``targets/interaction/eval.py``); pure data here (no method) so ``contracts`` stays
    logic-free."""

    grid: np.ndarray     # (Nx, Ny, Nz) signed distance (negative = inside)
    witness: np.ndarray  # (Nx, Ny, Nz, 3) nearest surface point per node, local frame
    origin: np.ndarray   # (3,) local coords of node (0, 0, 0)
    spacing: float       # isotropic voxel size (m)
    name: str            # channel name, e.g. "obj0" / "ground"

    def __post_init__(self) -> None:
        if self.witness.shape != self.grid.shape + (3,):
            raise ValueError(
                f"witness shape {self.witness.shape} != grid shape {self.grid.shape} + (3,)")
