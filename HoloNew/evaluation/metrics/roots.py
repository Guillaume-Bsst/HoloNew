"""Root-pose sanity metrics: position + orientation error of a rigid pose vs a reference.

Reusable for the robot floating base and for a movable object — both are SE(3) poses
scored against a reference trajectory. Pure: takes rotation matrices to avoid any
quaternion-convention ambiguity; the caller converts its quats once.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def pose_error_series(pos: np.ndarray, rot: np.ndarray,
                     pos_ref: np.ndarray, rot_ref: np.ndarray) -> dict[str, np.ndarray]:
    """Per-frame position (m) and geodesic orientation (rad) error of a pose trajectory.

    pos, pos_ref: (T, 3). rot, rot_ref: (T, 3, 3) world rotation matrices. Returns
    ``pos_err`` (T,) and ``rot_err`` (T,); ``compute_pose_error`` is the mean of these.
    """
    pos_err = np.linalg.norm(pos - pos_ref, axis=-1)
    delta = np.einsum("tji,tjl->til", rot, rot_ref)   # rot^T @ rot_ref
    rot_err = np.linalg.norm(R.from_matrix(delta).as_rotvec(), axis=-1)
    return {"pos_err": pos_err, "rot_err": rot_err}


def compute_pose_error(pos: np.ndarray, rot: np.ndarray,
                      pos_ref: np.ndarray, rot_ref: np.ndarray) -> dict[str, float]:
    """Mean position (m) and geodesic orientation (rad) error of a pose trajectory (reduces the series)."""
    s = pose_error_series(pos, rot, pos_ref, rot_ref)
    return {"pos_err": float(np.mean(s["pos_err"])), "rot_err": float(np.mean(s["rot_err"]))}
