"""Effort metrics: joint-limit margin / saturation and joint velocity."""
from __future__ import annotations

import numpy as np

_SAT_EPS = 0.02


def compute_effort(joints: np.ndarray, q_lower: np.ndarray, q_upper: np.ndarray,
                   dt: float) -> dict[str, float]:
    """Joint-limit margin / saturation and joint-velocity statistics.

    joints: (T, dof) actuated joint trajectory. q_lower/q_upper: (dof,) limits.
    Margin is normalized by joint range; negative = limit violation. Saturation is
    the fraction of (frame, joint) with normalized margin below ``_SAT_EPS``.
    """
    rng = np.maximum(q_upper - q_lower, 1e-9)
    margin = np.minimum(joints - q_lower, q_upper - joints) / rng  # (T, dof)
    vel = np.diff(joints, n=1, axis=0) / dt
    return {
        "joint_limit_margin_min": float(np.min(margin)),
        "joint_limit_saturation_frac": float(np.mean(margin < _SAT_EPS)),
        "joint_vel_rms": float(np.sqrt(np.mean(np.square(vel)))) if vel.size else 0.0,
        "joint_vel_max": float(np.max(np.abs(vel))) if vel.size else 0.0,
    }
