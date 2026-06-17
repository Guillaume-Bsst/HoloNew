"""Effort metrics: joint-limit margin / saturation and joint velocity."""
from __future__ import annotations

import numpy as np

_SAT_EPS = 0.02


def effort_series(joints: np.ndarray, q_lower: np.ndarray, q_upper: np.ndarray,
                  dt: float) -> dict[str, np.ndarray]:
    """Per-frame margin / saturation / velocity arrays underlying the effort scalars.

    joints: (T, dof) actuated joint trajectory. q_lower/q_upper: (dof,) limits.
    Returns ``margin`` (T, dof; normalized to range, negative = violation), ``saturated``
    (T, dof bool; margin below ``_SAT_EPS``) and ``vel`` (T-1, dof; first difference / dt).
    ``compute_effort`` is exactly the reduction of these, so series and scalar can't drift.
    """
    rng = np.maximum(q_upper - q_lower, 1e-9)
    margin = np.minimum(joints - q_lower, q_upper - joints) / rng  # (T, dof)
    return {
        "margin": margin,
        "saturated": margin < _SAT_EPS,
        "vel": np.diff(joints, n=1, axis=0) / dt,
    }


def compute_effort(joints: np.ndarray, q_lower: np.ndarray, q_upper: np.ndarray,
                   dt: float) -> dict[str, float]:
    """Joint-limit margin / saturation and joint-velocity statistics (reduces the series)."""
    s = effort_series(joints, q_lower, q_upper, dt)
    vel = s["vel"]
    return {
        "joint_limit_margin_min": float(np.min(s["margin"])),
        "joint_limit_saturation_frac": float(np.mean(s["saturated"])),
        "joint_vel_rms": float(np.sqrt(np.mean(np.square(vel)))) if vel.size else 0.0,
        "joint_vel_peak": float(np.max(np.abs(vel))) if vel.size else 0.0,
    }
