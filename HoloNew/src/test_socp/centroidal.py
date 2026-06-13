"""Centroidal W^c (CoM acceleration) + W^L (angular momentum) for TEST-SOCP.

See docs/specs/2026-06-13-brick4-centroidal-design.md.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np


def build_centroidal_terms(rt, q_t0, q_tm1, c_tm1, c_tm2, cddot_ref, dqa,
                            lambda_c, lambda_L, dt):
    """Assemble W^c (CoM acceleration tracking) + W^L (angular momentum -> 0).

    W^c = lambda_c * ||c_ddot - cddot_ref||^2
        c_ddot = (c0 + Jc @ dqa - 2*c_tm1 + c_tm2) / dt^2   (linearised in dqa)
        Jc = com_jacobian(q_t0)[:, v_a_indices]

    W^L = lambda_L * ||L||^2
        L = (A_G @ v)[3:6],  v = difference(q_tm1, q_t0) + Jd[:, v_a_indices] @ dqa
        A_G = centroidal_map(q_t0)

    Each term is represented as cp.sum_squares(A @ dqa + b) so that the sqrt(lambda)
    folding makes the squared norm equal to the weighted residual.

    Args:
        rt: TestSocpRetargeter instance (provides rt.pin, rt.v_a_indices).
        q_t0: pinocchio config at current time step (q_t).
        q_tm1: pinocchio config at previous time step (q_{t-1}).
        c_tm1: CoM (3,) at t-1, previously solved.
        c_tm2: CoM (3,) at t-2, previously solved.
        cddot_ref: reference CoM acceleration (3,).
        dqa: cvxpy Variable (nv_a,), the active-joint tangent increment.
        lambda_c: weight for W^c.
        lambda_L: weight for W^L.
        dt: time step in seconds.

    Returns:
        List of two cvxpy expressions [W_c_expr, W_L_expr].
    """
    # --- W^c: CoM acceleration ---
    c0 = rt.pin.com(q_t0)                                      # (3,)
    Jc = rt.pin.com_jacobian(q_t0)[:, rt.v_a_indices]          # (3, nv_a)
    # Fold sqrt(lambda_c) and 1/dt^2 into A and b so that:
    #   ||A_c @ dqa + b_c||^2 = lambda_c * ||cddot - cddot_ref||^2
    s_c = np.sqrt(lambda_c) / dt**2
    A_c = s_c * Jc                                              # (3, nv_a)
    b_c = (s_c * (c0 - 2.0*np.asarray(c_tm1) + np.asarray(c_tm2))
           - np.sqrt(lambda_c) * np.asarray(cddot_ref))        # (3,)

    # --- W^L: angular centroidal momentum -> 0 ---
    Ag = rt.pin.centroidal_map(q_t0)                           # (6, nv)
    v0, Jd = rt.pin.difference_and_jac(q_tm1, q_t0)           # v0: (nv,), Jd: (nv, nv)
    AgL = Ag[3:6, :]                                           # (3, nv)
    # v_active contribution via Jd columns for active joints
    A_L = np.sqrt(lambda_L) * (AgL @ Jd[:, rt.v_a_indices])   # (3, nv_a)
    b_L = np.sqrt(lambda_L) * (AgL @ v0)                       # (3,)

    return [cp.sum_squares(A_c @ dqa + b_c), cp.sum_squares(A_L @ dqa + b_L)]
