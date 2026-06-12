# src/test_pipe_retargeting/test_pipe_retargeting/fields/backends/floor.py
"""Analytic flat-floor (z=0) contact field backend.

Both FloorField and ObjectSDF expose query(points, margin) -> ContactField —
that duck-typed contract is the shared "Field" interface.
"""
from __future__ import annotations

import numpy as np

from ..contact_field import ContactField


def floor_field(probe_pts: np.ndarray, margin: float) -> ContactField:
    """Analytic flat-floor (z=0) fast path: signed distance = z.

    direction follows the same surface->probe convention as surface_field, i.e.
    (probe - witness) normalized = (0, 0, sign(z)): +z for a probe above the floor,
    -z for a penetrating (below-floor) probe.
    """
    z = probe_pts[:, 2].astype(np.float32)
    active = z < margin
    n = len(probe_pts)
    direction = np.zeros((n, 3), dtype=np.float32)
    direction[active, 2] = np.sign(z[active])
    witness = probe_pts.copy().astype(np.float32)
    witness[:, 2] = 0.0
    return ContactField(
        distance=np.where(active, z, margin).astype(np.float32),
        direction=direction, witness=witness, active=active,
    )


class FloorField:
    """Analytic z=0 floor as a Field: query() delegates to floor_field. Exact everywhere."""

    def query(self, pts_local: np.ndarray, margin: float) -> ContactField:
        return floor_field(np.asarray(pts_local, dtype=np.float64), margin)
