"""Movable-entity W^o (object motion regularization) for TEST-SOCP.
See docs/specs/2026-06-13-brick5-movable-entities-design.md."""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pinocchio as pin


def build_wo_term(
    T_obj0,
    T_obj_tm1,
    T_obj_tm2,
    vdot_ref,
    omega_ref,
    dxi,
    lambda_o,
    lambda_omega,
    dt,
):
    """W^o: lambda_o*||vdot - vdot_ref||^2 + lambda_omega*||omega - omega_ref||^2,
    linearized in the object tangent step dxi (object pose T = exp6(dxi) * T_obj0).

    The object velocity at t is V_t = (1/dt) log6(T_obj_tm1^{-1} exp6(dxi) T_obj0).
    At dxi=0, let M0 = T_obj_tm1^{-1} T_obj0. The Jacobian of V_t wrt dxi is:

        dV_t/d(dxi) = (1/dt) * Jlog6(M0) @ Ad(T_obj0^{-1})

    which follows from the identity:
        T_obj_tm1^{-1} exp6(dxi) T_obj0
        = exp6(Ad(T_obj_tm1^{-1}) dxi) * M0
        = M0 * exp6(Ad(M0^{-1}) Ad(T_obj_tm1^{-1}) dxi)
    and that Jlog6(M) is the right Jacobian of log6 at M.
    Composing Ad(M0^{-1}) Ad(T_obj_tm1^{-1}) = Ad(T_obj0^{-1}) via the Ad homomorphism.

    Args:
        T_obj0: pin.SE3, current object pose (linearization point, T_obj at t).
        T_obj_tm1: pin.SE3, object pose at t-1.
        T_obj_tm2: pin.SE3, object pose at t-2.
        vdot_ref: (3,) linear acceleration reference.
        omega_ref: (3,) angular velocity reference.
        dxi: cp.Variable of shape (6,), world-frame SE(3) tangent step.
        lambda_o: weight on the linear acceleration term.
        lambda_omega: weight on the angular velocity term.
        dt: timestep in seconds.

    Returns:
        A scalar cvxpy expression (the W^o cost).
    """
    # Velocity at t linearized in dxi.
    M0 = T_obj_tm1.inverse() * T_obj0
    v0 = pin.log6(M0).vector / dt                               # (6,) V_t at dxi=0
    J = (pin.Jlog6(M0) @ T_obj0.inverse().action) / dt         # (6,6)

    # Velocity at t-1: constant (no dxi dependence).
    v_tm1 = pin.log6(T_obj_tm2.inverse() * T_obj_tm1).vector / dt   # (6,)

    # Linear acceleration (vdot) and angular velocity (omega) as affine in dxi.
    # vdot = (V_t[:3] - V_tm1[:3]) / dt, omega = V_t[3:6]
    A_vdot = J[:3, :] / dt                                      # (3, 6)
    b_vdot = (v0[:3] - v_tm1[:3]) / dt - np.asarray(vdot_ref)  # (3,)
    A_omega = J[3:6, :]                                          # (3, 6)
    b_omega = v0[3:6] - np.asarray(omega_ref)                   # (3,)

    r1 = np.sqrt(lambda_o) * (A_vdot @ dxi + b_vdot)
    r2 = np.sqrt(lambda_omega) * (A_omega @ dxi + b_omega)
    return cp.sum_squares(r1) + cp.sum_squares(r2)
