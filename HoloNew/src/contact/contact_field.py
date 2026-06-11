# src/test_pipe_retargeting/test_pipe_retargeting/fields/contact_field.py
"""Signed surface contact/approach field via Coal.

For each probe point we compute the signed distance to a target surface
(negative = penetrating), the closest (witness) point on that surface, and the
contact normal (the approach direction). The unsigned distance + witness + normal
come from Coal's BVH distance query; the sign comes from a trimesh inside/outside
test, because Coal's BVH distance is unsigned.

Probes farther than `margin` from the surface are inactive: distance is clamped to
+margin and direction is zero, so only the active set carries a contact signal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Batch size for trimesh.contains ray casting (peaks at O(batch x faces)).
_CONTAINS_BATCH: int = 8192


@dataclass(frozen=True)
class ContactField:
    """Per-probe field for one frame / one channel.

    Frozen prevents field reassignment; the numpy arrays themselves stay mutable
    (same convention as PointCloudCache in human_body.py) — treat them as read-only.
    """
    distance: np.ndarray   # (N,)    signed distance; +margin where inactive
    direction: np.ndarray  # (N, 3)  contact normal (surface -> probe); 0 where inactive
    witness: np.ndarray    # (N, 3)  closest surface point for active probes; 0 for inactive (batched path)
    active: np.ndarray     # (N,)    bool, within margin of the surface


def _contains(mesh, pts, batch=_CONTAINS_BATCH):
    n = len(pts)
    inside = np.empty(n, dtype=bool)
    for i in range(0, n, batch):
        inside[i:i + batch] = mesh.contains(pts[i:i + batch])
    return inside


def _probe_distance(pt, target_bvh, sphere, tf_pt, tf_mesh, req):
    """Exact Coal point->surface query for one probe (reusing preallocated objects).

    Returns (unsigned_dist, witness(3,), direction(3,)). direction is the unit
    vector surface->probe, or zero when probe and witness coincide. Shared by
    surface_field and surface_field_batched so the Coal idiom lives in one place.
    """
    import coal
    tf_pt.setTranslation(pt)
    res = coal.DistanceResult()
    coal.distance(sphere, tf_pt, target_bvh, tf_mesh, req, res)
    unsigned = abs(res.min_distance)
    w = np.asarray(res.getNearestPoint2(), dtype=np.float64)
    d = pt - w
    norm = np.linalg.norm(d)
    direction = d / norm if norm > 1e-9 else np.zeros(3)
    return unsigned, w, direction


def _stack(fields: list[ContactField]) -> ContactField:
    return ContactField(
        distance=np.stack([f.distance for f in fields]),
        direction=np.stack([f.direction for f in fields]),
        witness=np.stack([f.witness for f in fields]),
        active=np.stack([f.active for f in fields]),
    )


def _to_object_frame(pts_world: np.ndarray, pose: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R
    q_xyzw = np.array([pose[1], pose[2], pose[3], pose[0]], dtype=np.float64)
    t = pose[4:7].astype(np.float64)
    return R.from_quat(q_xyzw).inv().apply(pts_world.astype(np.float64) - t)


def _from_object_frame(pts_local: np.ndarray, pose: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R
    q_xyzw = np.array([pose[1], pose[2], pose[3], pose[0]], dtype=np.float64)
    t = pose[4:7].astype(np.float64)
    return R.from_quat(q_xyzw).apply(pts_local.astype(np.float64)) + t
