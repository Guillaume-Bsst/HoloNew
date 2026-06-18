"""Temporal regularization (W^r) for TEST-SOCP — acceleration penalty in the
pinocchio tangent space.
See docs/specs/2026-06-13-brick2-temporal-regularization-design.md."""
from __future__ import annotations

import numpy as np


def build_temporal_block(rt, q_t0, q_tm1, q_tm2, lambda_r, sigma_qddot, sigma_Vdot, dt):
    """ResidualBlock form of build_temporal_term (same math, weights already folded in A/b).

    Returns a list of one ResidualBlock. The block satisfies ‖b.c‖² ==
    the cvxpy term value at dqa=0.
    """
    from HoloNew.src.test_socp.solve.spec import ResidualBlock

    nv = rt.pin.model.nv
    v0, J = rt.pin.difference_and_jac(q_tm1, q_t0)
    v_tm1 = rt.pin.difference_and_jac(q_tm2, q_tm1)[0]
    Jc = J[:, rt.v_a_indices]                          # (nv, nv_a)
    # per-DOF sqrt weights / dt^2
    w = np.full(nv, 1.0 / sigma_qddot)
    w[:6] = 1.0 / sigma_Vdot
    s = np.sqrt(lambda_r) * w / dt ** 2               # (nv,)
    A = s[:, None] * Jc                                # (nv, nv_a)
    b = s * (v0 - v_tm1)                               # (nv,)
    return [ResidualBlock(A=A, c=b, name="temporal")]  # c == the +b constant of the original sum_squares(A@dqa + b)
