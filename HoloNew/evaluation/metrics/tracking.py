"""Tracking fidelity: robot mapped keypoints vs SMPL reference joints."""
from __future__ import annotations

import numpy as np


def tracking_series(robot_kpts: np.ndarray, ref_kpts: np.ndarray, root_idx: int,
                   base_xyz: np.ndarray | None = None,
                   ref_root_xyz: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Per-frame, per-keypoint tracking error arrays underlying the tracking scalars.

    robot_kpts, ref_kpts: (T, J, 3), aligned by the joints_mapping order. No Procrustes
    alignment — global placement is part of the retargeting task. Returns ``mpjpe``
    (T, J), ``mpjpe_root_rel`` (T, J; after subtracting ``root_idx`` from both) and,
    when a base is given, ``base_track`` (T,). ``compute_tracking`` is exactly the mean
    of these, so series and scalar can't drift.
    """
    err = np.linalg.norm(robot_kpts - ref_kpts, axis=-1)  # (T, J)
    rb = robot_kpts - robot_kpts[:, root_idx:root_idx + 1, :]
    rf = ref_kpts - ref_kpts[:, root_idx:root_idx + 1, :]
    err_rr = np.linalg.norm(rb - rf, axis=-1)             # (T, J)
    out = {"mpjpe": err, "mpjpe_root_rel": err_rr}
    if base_xyz is not None and ref_root_xyz is not None:
        out["base_track"] = np.linalg.norm(base_xyz - ref_root_xyz, axis=-1)  # (T,)
    return out


def compute_tracking(robot_kpts: np.ndarray, ref_kpts: np.ndarray, root_idx: int,
                    base_xyz: np.ndarray | None = None,
                    ref_root_xyz: np.ndarray | None = None) -> dict[str, float]:
    """Per-keypoint position error between robot links and reference joints (reduces the series)."""
    s = tracking_series(robot_kpts, ref_kpts, root_idx, base_xyz, ref_root_xyz)
    out = {
        "mpjpe_global": float(np.mean(s["mpjpe"])),
        "mpjpe_root_rel": float(np.mean(s["mpjpe_root_rel"])),
    }
    if "base_track" in s:
        out["base_track_err"] = float(np.mean(s["base_track"]))
    return out
