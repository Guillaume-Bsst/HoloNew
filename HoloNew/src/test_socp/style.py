"""Pelvis-relative Style objective pieces for TEST-SOCP.
See docs/specs/2026-06-13-brick3-pelvis-relative-style-design.md."""
from __future__ import annotations

import numpy as np

_ZHAT = np.array([0.0, 0.0, 1.0])
_ZSKEW = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # [zhat]_x


def pelvis_tilt_residual(rt, q_mj, R_B_ref):
    """Roll/pitch tilt residual r0 - A@dqa for ||(R_ref)ᵀẑ − R_Bᵀẑ||²."""
    R_B = rt.body_rotation(q_mj, "pelvis")
    _, Jr_B = rt._body_jac(q_mj, "pelvis")
    u = R_B.T @ _ZHAT
    r0 = R_B_ref.T @ _ZHAT - u
    A = R_B.T @ _ZSKEW @ Jr_B
    return r0, A
