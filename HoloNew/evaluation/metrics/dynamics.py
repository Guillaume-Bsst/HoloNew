"""Dynamic-plausibility metrics: CoM acceleration error and angular-momentum RMS."""
from __future__ import annotations

import numpy as np


def _ddot(x: np.ndarray, dt: float) -> np.ndarray:
    return np.diff(x, n=2, axis=0) / dt ** 2


def dynamics_series(com: np.ndarray, ref_com: np.ndarray, dt: float, *,
                   L: np.ndarray | None = None,
                   L_ref: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Per-frame CoM-acceleration error and angular-momentum magnitude arrays.

    com, ref_com: (T, 3) robot / reference CoM. ``com_accel_err`` (T-2,) is the per-frame
    norm of the difference of their 2nd finite differences (m/s^2). When L is given,
    ``ang_momentum_mag`` (T,) is |L| (or |L - L_ref|). ``compute_dynamics`` reduces these,
    so series and scalar can't drift.
    """
    out = {"com_accel_err": np.linalg.norm(_ddot(com, dt) - _ddot(ref_com, dt), axis=-1)}
    if L is not None:
        Lc = L if L_ref is None else (L - L_ref)
        out["ang_momentum_mag"] = np.sqrt(np.sum(np.square(Lc), axis=-1))
    return out


def compute_dynamics(com: np.ndarray, ref_com: np.ndarray, dt: float, *,
                    L: np.ndarray | None = None,
                    L_ref: np.ndarray | None = None) -> dict[str, float]:
    """CoM-acceleration tracking error and centroidal angular-momentum RMS (reduces the series)."""
    s = dynamics_series(com, ref_com, dt, L=L, L_ref=L_ref)
    out = {"com_accel_err": float(np.mean(s["com_accel_err"]))}
    if "ang_momentum_mag" in s:
        out["ang_momentum_rms"] = float(np.sqrt(np.mean(np.square(s["ang_momentum_mag"]))))
    return out
