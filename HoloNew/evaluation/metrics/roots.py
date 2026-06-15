"""Root-pose sanity metrics: position + orientation error of a rigid pose vs a reference.

Reusable for the robot floating base and for a movable object — both are SE(3) poses
scored against a reference trajectory. Pure: takes rotation matrices to avoid any
quaternion-convention ambiguity; the caller converts its quats once.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def compute_pose_error(pos: np.ndarray, rot: np.ndarray,
                      pos_ref: np.ndarray, rot_ref: np.ndarray) -> dict[str, float]:
    """Mean position (m) and geodesic orientation (rad) error of a pose trajectory.

    pos, pos_ref: (T, 3). rot, rot_ref: (T, 3, 3) world rotation matrices.
    """
    pos_err = float(np.mean(np.linalg.norm(pos - pos_ref, axis=-1)))
    delta = np.einsum("tji,tjl->til", rot, rot_ref)   # rot^T @ rot_ref
    ang = np.linalg.norm(R.from_matrix(delta).as_rotvec(), axis=-1)
    return {"pos_err": pos_err, "rot_err": float(np.mean(ang))}
