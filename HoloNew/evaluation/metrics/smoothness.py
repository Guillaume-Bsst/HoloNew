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


def compute_smoothness(qpos: np.ndarray, dof: int, dt: float) -> dict[str, float]:
    """Acceleration / jerk RMS of the base and actuated joints.

    qpos: (T, 7+dof[+7]) with [0:3] base xyz, [3:7] base quat wxyz, [7:7+dof] joints.
    Accelerations are 2nd finite differences / dt**2; jerk is the 3rd / dt**3.
    ``joint_jerk_meanabs`` is the per-frame (no-dt) definition kept for continuity
    with the W^r A/B test.
    """
    base_pos = qpos[:, 0:3]
    quat = qpos[:, 3:7]
    joints = qpos[:, 7:7 + dof]

    base_acc = np.diff(base_pos, n=2, axis=0) / dt ** 2
    omega = _base_angular_velocity(quat, dt)
    base_ang_acc = np.diff(omega, n=1, axis=0) / dt
    j_acc = np.diff(joints, n=2, axis=0) / dt ** 2
    j_jerk = np.diff(joints, n=3, axis=0) / dt ** 3

    return {
        "base_pos_accel_rms": _rms(base_acc),
        "base_ang_accel_rms": _rms(base_ang_acc),
        "joint_accel_rms": _rms(j_acc),
        "joint_jerk_rms": _rms(j_jerk),
        "joint_jerk_meanabs": float(np.mean(np.abs(np.diff(joints, n=3, axis=0)))),
    }
