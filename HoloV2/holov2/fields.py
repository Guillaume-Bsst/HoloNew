"""Concrete distance fields implementing ``contracts.Field`` — sampled in the field's own
local frame. Kept out of ``contracts.py`` because they carry sampling LOGIC (not pure data).

- ``GridSDF``   : trilinear on a cached signed-distance grid — rigid OBJECTS *and* arbitrary
                  TERRAIN ground (stairs / slope / climbing boxes).
- ``PlaneField``: analytic flat ground (the DEFAULT) — infinite plane, no grid. Avoids a
                  wasteful finite 3D grid and the "walked past the grid" failure. The ground is
                  NOT hard-locked to this: a terrain scene uses a GridSDF ground instead.

Both return a ``ContactField`` (signed distance + contact direction + witness + active mask).
The eval loops uniformly: ``channel.field.sample_local(local_pts, margin)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import ContactField


@dataclass(frozen=True)
class GridSDF:
    """Signed-distance grid of a rigid surface in its local frame (the cached SDF asset)."""

    grid: np.ndarray     # (Nx, Ny, Nz) signed distance (negative = inside)
    origin: np.ndarray   # (3,) local coords of node (0, 0, 0)
    spacing: float       # isotropic voxel size (m)
    name: str            # channel name, e.g. "obj0" or "ground" (terrain)

    def sample_local(self, points_local: np.ndarray, margin: float) -> ContactField:
        """Trilinear distance + analytic gradient (contact direction); inactive beyond margin."""
        raise NotImplementedError


@dataclass(frozen=True)
class PlaneField:
    """Analytic plane = the DEFAULT flat ground: signed distance = normal . p - offset.
    Infinite, no grid. (A terrain ground uses a GridSDF instead.)"""

    normal: np.ndarray   # (3,) unit normal (e.g. +z)
    offset: float        # plane offset along the normal (0 for z=0 ground)
    name: str            # e.g. "ground"

    def sample_local(self, points_local: np.ndarray, margin: float) -> ContactField:
        """distance = normal·p - offset ; direction = normal ; witness = p - distance·normal."""
        raise NotImplementedError
