"""Dynamic-plausibility metrics: CoM acceleration error and angular-momentum RMS."""
from __future__ import annotations

import numpy as np


def _ddot(x: np.ndarray, dt: float) -> np.ndarray:
    return np.diff(x, n=2, axis=0) / dt ** 2


def compute_dynamics(com: np.ndarray, ref_com: np.ndarray, dt: float, *,
                    L: np.ndarray | None = None,
                    L_ref: np.ndarray | None = None) -> dict[str, float]:
    """CoM-acceleration tracking error and centroidal angular-momentum RMS.

    com, ref_com: (T, 3) robot and reference CoM trajectories. ``com_accel_err`` is
    the mean norm of the difference of their 2nd finite differences (m/s^2) — the
    same definition as the centroidal A/B test.
    L, L_ref: optional (T, 3) centroidal angular momentum (and reference). When L is
    given, ``ang_momentum_rms`` is the RMS magnitude of L (or L - L_ref).
    """
    cdd = _ddot(com, dt)
    rdd = _ddot(ref_com, dt)
    out = {"com_accel_err": float(np.mean(np.linalg.norm(cdd - rdd, axis=-1)))}
    if L is not None:
        Lc = L if L_ref is None else (L - L_ref)
        out["ang_momentum_rms"] = float(np.sqrt(np.mean(np.sum(np.square(Lc), axis=-1))))
    return out
