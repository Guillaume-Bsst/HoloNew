"""Movable-entity W^o (object motion regularization) for TEST-SOCP.
See docs/specs/2026-06-13-brick5-movable-entities-design.md."""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pinocchio as pin


def pose_to_se3(pose7: np.ndarray) -> pin.SE3:
    """Convert a 7-vector [qw, qx, qy, qz, x, y, z] to a pinocchio SE3.

    Pinocchio's Quaternion constructor takes (w, x, y, z) in scalar-first order,
    which matches the OMOMO dataset convention [qw, qx, qy, qz, x, y, z].

    Args:
        pose7: array of shape (7,) with [qw, qx, qy, qz, x, y, z].

    Returns:
        pin.SE3 with the corresponding rotation and translation.
    """
    pose7 = np.asarray(pose7, dtype=float)
    qw, qx, qy, qz = pose7[0], pose7[1], pose7[2], pose7[3]
    t = pose7[4:7].copy()
    R = pin.Quaternion(qw, qx, qy, qz).matrix()
    return pin.SE3(R, t)


def se3_to_pose(M: pin.SE3) -> np.ndarray:
    """Convert a pinocchio SE3 to a 7-vector [qw, qx, qy, qz, x, y, z].

    Args:
        M: pin.SE3 instance.

    Returns:
        Array of shape (7,) with [qw, qx, qy, qz, x, y, z].
    """
    q = pin.Quaternion(M.rotation)
    # pin.Quaternion stores (x, y, z, w) internally but exposes .w, .x, .y, .z
    return np.array([q.w, q.x, q.y, q.z, M.translation[0], M.translation[1], M.translation[2]])


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


def build_wo_position_anchor(T_obj0, p_ref, dxi, lambda_o_pos):
    """W^o position anchor: lambda_o_pos * ||p_obj(dxi) - p_ref||^2.

    W^o (build_wo_term) regularizes only the object's linear acceleration and
    angular velocity, which are invariant to a constant position offset.  With
    nothing anchoring the absolute object position, the solved object pose can
    drift along the reference path while still matching the reference
    acceleration profile (the same position-blindness as the centroidal W^c
    term).  This term pins the absolute object position to p_ref.

    The object position is p(dxi) = (exp6(dxi) * T_obj0).translation, whose
    first-order expansion about dxi=0 (with p0 = T_obj0.translation and the
    pinocchio motion ordering dxi = [v; omega]) is:

        p(dxi) ~= p0 + [I3 | -skew(p0)] @ dxi

    so the residual is A_pos @ dxi + (p0 - p_ref).

    Args:
        T_obj0: pin.SE3, current object pose (linearization point, T_obj at t).
        p_ref: (3,) reference object position to anchor to.
        dxi: cp.Variable of shape (6,), world-frame SE(3) tangent step.
        lambda_o_pos: weight on the position anchor.

    Returns:
        A scalar cvxpy expression (the position-anchor cost).
    """
    p0 = np.asarray(T_obj0.translation, dtype=float)
    A_pos = np.hstack([np.eye(3), -pin.skew(p0)])     # (3, 6)
    b_pos = p0 - np.asarray(p_ref, dtype=float)        # (3,)
    r = np.sqrt(lambda_o_pos) * (A_pos @ dxi + b_pos)
    return cp.sum_squares(r)
