"""Tracking fidelity: robot mapped keypoints vs SMPL reference joints."""
from __future__ import annotations

import numpy as np


def compute_tracking(robot_kpts: np.ndarray, ref_kpts: np.ndarray, root_idx: int,
                    base_xyz: np.ndarray | None = None,
                    ref_root_xyz: np.ndarray | None = None) -> dict[str, float]:
    """Per-keypoint position error between robot links and SMPL reference joints.

    robot_kpts, ref_kpts: (T, J, 3), aligned by the joints_mapping order. No
    Procrustes alignment — global placement is part of the retargeting task.
    ``mpjpe_root_rel`` subtracts the root (``root_idx``) from both before erroring.
    base_xyz / ref_root_xyz: optional (T, 3) for the base tracking error.
    """
    err = np.linalg.norm(robot_kpts - ref_kpts, axis=-1)  # (T, J)
    rb = robot_kpts - robot_kpts[:, root_idx:root_idx + 1, :]
    rf = ref_kpts - ref_kpts[:, root_idx:root_idx + 1, :]
    err_rr = np.linalg.norm(rb - rf, axis=-1)
    out = {
        "mpjpe_global": float(np.mean(err)),
        "mpjpe_root_rel": float(np.mean(err_rr)),
    }
    if base_xyz is not None and ref_root_xyz is not None:
        out["base_track_err"] = float(
            np.mean(np.linalg.norm(base_xyz - ref_root_xyz, axis=-1)))
    return out
