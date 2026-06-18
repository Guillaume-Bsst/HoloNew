"""Temporal regularization (W^r) for TEST-SOCP — acceleration penalty in the
pinocchio tangent space.
See docs/specs/2026-06-13-brick2-temporal-regularization-design.md."""
from __future__ import annotations

import cvxpy as cp
import numpy as np


def build_temporal_term(rt, q_t0, q_tm1, q_tm2, dqa, lambda_r, sigma_qddot, sigma_Vdot, dt):
    """One cvxpy expression penalizing the tangent-space acceleration.

    v_t   = difference(q_tm1, integrate(q_t0, v_full))  ~= v0 + J @ dqa  (linearized)
    v_tm1 = difference(q_tm2, q_tm1)                                     (constant)
    cost  = lambda_r * sum_k w_k * ((v_t - v_tm1)/dt^2)_k^2
    with base tangent rows (0:6) weighted by 1/sigma_Vdot^2, joints by 1/sigma_qddot^2.
    """
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
    return cp.sum_squares(A @ dqa + b)


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
