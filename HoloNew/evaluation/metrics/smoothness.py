"""Smoothness metrics: acceleration / jerk RMS of base and joints (pure qpos)."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def _base_angular_velocity(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    """Angular velocity (T-1, 3) from a (T, 4) wxyz quaternion trajectory."""
    q = quat_wxyz[:, [1, 2, 3, 0]]  # scipy wants [x, y, z, w]
    rot = R.from_quat(q)
    rel = rot[:-1].inv() * rot[1:]
    return rel.as_rotvec() / dt


def smoothness_series(qpos: np.ndarray, dof: int, dt: float) -> dict[str, np.ndarray]:
    """Per-frame finite-difference arrays underlying the smoothness scalars.

    qpos: (T, 7+dof[+7]) with [0:3] base xyz, [3:7] base quat wxyz, [7:7+dof] joints.
    Returns the raw (un-reduced) arrays, with the natural finite-difference lengths
    (accel = T-2, jerk = T-3): ``base_acc`` / ``base_ang_acc`` (m/s^2, rad/s^2),
    ``joint_accel`` (m or rad /s^2), ``joint_jerk`` (.../s^3), and ``joint_jerk_nodt``
    (the no-dt 3rd difference kept for the W^r continuity scalar). ``compute_smoothness``
    is exactly the reduction of these arrays, so the exported series and the scoreboard
    scalar can never drift.
    """
    base_pos = qpos[:, 0:3]
    quat = qpos[:, 3:7]
    joints = qpos[:, 7:7 + dof]
    return {
        "base_acc": np.diff(base_pos, n=2, axis=0) / dt ** 2,
        "base_ang_acc": np.diff(_base_angular_velocity(quat, dt), n=1, axis=0) / dt,
        "joint_accel": np.diff(joints, n=2, axis=0) / dt ** 2,
        "joint_jerk": np.diff(joints, n=3, axis=0) / dt ** 3,
        "joint_jerk_nodt": np.diff(joints, n=3, axis=0),
    }


def compute_smoothness(qpos: np.ndarray, dof: int, dt: float) -> dict[str, float]:
    """Acceleration / jerk RMS of the base and actuated joints (reduces the series)."""
    s = smoothness_series(qpos, dof, dt)
    return {
        "base_pos_accel_rms": _rms(s["base_acc"]),
        "base_ang_accel_rms": _rms(s["base_ang_acc"]),
        "joint_accel_rms": _rms(s["joint_accel"]),
        "joint_jerk_rms": _rms(s["joint_jerk"]),
        "joint_jerk_meanabs": float(np.mean(np.abs(s["joint_jerk_nodt"]))),
    }
