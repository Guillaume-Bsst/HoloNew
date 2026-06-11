# src/test_pipe_retargeting/test_pipe_retargeting/fields/probes.py
from __future__ import annotations

import numpy as np

from .constants import (
    FLOOR_GRID_DENSITY,
    FLOOR_GRID_SIZE,
    OBJECT_GRID_DENSITY,
)


def make_floor_grid(
    center_xy: tuple[float, float] = (0.0, 0.0),
    size: float = FLOOR_GRID_SIZE,
    density: float = FLOOR_GRID_DENSITY,
) -> np.ndarray:
    """Return (N, 3) float32 grid at z=0 centred on center_xy, based on density."""
    num_pts = int(size * size * density)
    resolution = int(np.sqrt(num_pts))
    xs = np.linspace(center_xy[0] - size / 2, center_xy[0] + size / 2, resolution)
    ys = np.linspace(center_xy[1] - size / 2, center_xy[1] + size / 2, resolution)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(resolution * resolution)])
    return pts.astype(np.float32)


def make_object_grid(
    mesh: object,
    density: float = OBJECT_GRID_DENSITY,
) -> np.ndarray:
    """Return (N, 3) float32 grid sampled on the object surface, based on density."""
    import trimesh
    area = mesh.area
    num_pts = int(area * density)
    # sample_surface_even produces a more uniform point cloud than random sampling.
    pts, _ = trimesh.sample.sample_surface_even(mesh, num_pts)
    return pts.astype(np.float32)
