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


def feedforward_object_warmstart(ref_t, ref_tm1, solved_tm1) -> np.ndarray:
    """Feed-forward object linearization point for frame t.

        T_warm = (T_ref[t] · T_ref[t-1]^{-1}) · T_solved[t-1]

    Advance the previous SOLVED object pose by the reference's per-frame world
    increment ΔT_ref = T_ref[t]·T_ref[t-1]^{-1}. This rides on the solved pose (so
    the accumulated grasp correction persists) while pre-applying the reference
    motion, so the SQP step only has to fix the contact residual — no lag on fast
    object motion. Reduces to T_ref[t] when the previous solve matched the reference.

    Args:
        ref_t, ref_tm1, solved_tm1: pose7 [qw,qx,qy,qz,x,y,z] (reference at t / t-1,
            solved object at t-1).

    Returns:
        pose7 warm-start linearization point.
    """
    T_ref_t = pose_to_se3(np.asarray(ref_t, dtype=float))
    T_ref_tm1 = pose_to_se3(np.asarray(ref_tm1, dtype=float))
    T_solved_tm1 = pose_to_se3(np.asarray(solved_tm1, dtype=float))
    dT_ref = T_ref_t * T_ref_tm1.inverse()        # reference world increment
    return se3_to_pose(dT_ref * T_solved_tm1)


def sample_object_surface(mesh_file: str, density: float = 200.0,
                          seed: int = 0) -> np.ndarray:
    """Sample control points on the object mesh surface (object-local frame).

    Mirrors the robot correspondence: the object is a movable carrier whose
    surface points interact with entities (here the floor). Even surface sampling
    at ``density`` points/m^2 (minimum 64). Returns (M, 3) object-local offsets,
    motion-independent (sampled once).
    """
    import trimesh
    mesh = trimesh.load(mesh_file, force="mesh")
    n = max(64, int(float(mesh.area) * density))
    pts, _ = trimesh.sample.sample_surface_even(mesh, n, seed=seed)
    return np.asarray(pts, dtype=np.float64)


def build_object_floor_terms(rt, dxi, obj_pose, lambda_d_obj, lambda_x_obj, margin,
                             obj_pose_ref=None):
    """Object<->floor D + X (the object-environment pair, object = carrier). Object
    surface points are carried by the object pose T = exp6(dxi)*T0 and query the
    analytic floor field (z=0 plane). Each near-floor point gets a D (normal/height)
    and X (tangential/placement) residual, with INDEPENDENT weights lambda_d_obj /
    lambda_x_obj (mirroring the robot lambda_d / lambda_x). Activation comes from the
    floor distance, so the term acts only when the object is near the floor and
    VANISHES when it is lifted (-> object free -> W^o ballistic).

    Linearization point is the object pose (obj_pose). Mirroring the robot floor D/X
    (interaction.build_dx_terms), each residual carries a REFERENCE TARGET drawn from
    obj_pose_ref: D drives the near-floor point's floor distance from its current value
    d0 toward the reference object's floor distance d_ref, and X drives its tangential
    (x,y) footprint toward the reference object's footprint. WITHOUT this target the
    residual is just A@dxi (minimised at dxi=0), so the box only freezes at the
    warm-start and never aligns regardless of the weight. obj_pose_ref=None keeps that
    legacy no-target behaviour. For a world point p, p(dxi) ~= p0 + [I, -skew(p0)] @ dxi.

    Args:
        rt: retargeter (provides rt.object_surface_local).
        dxi: cp.Variable(6), object SE(3) tangent step.
        obj_pose: (7,) [qw,qx,qy,qz,x,y,z] object pose at this frame (linearization point).
        lambda_d_obj: object-floor normal-proximity (D) weight (0 disables).
        lambda_x_obj: object-floor tangential-placement (X) weight (0 disables).
        margin: floor activation band L (metres).
        obj_pose_ref: (7,) reference object pose providing the D/X contact target.
            None -> zero target (legacy motion-penalty behaviour).

    Returns:
        List of up to two cvxpy expressions [D_term, X_term] (empty if no active
        near-floor object points or both weights are 0).
    """
    import cvxpy as cp
    from HoloNew.src.test_socp.interaction import _activation, _skew

    p_local = rt.object_surface_local                       # (M, 3)
    M = p_local.shape[0]
    T0 = pose_to_se3(obj_pose)
    p_w0 = p_local @ T0.rotation.T + T0.translation         # (M, 3) world at linearization
    d0 = p_w0[:, 2]                                          # floor signed distance (z)

    alpha = np.array([_activation(float(d0[i]), margin) for i in range(M)])
    active = np.where(alpha > 0)[0]
    if active.size == 0:
        return []

    # Reference contact target: the near-floor points are driven toward the REFERENCE
    # object's floor distance / footprint (so the solved box reproduces the reference
    # contact, not just the warm-start). None -> target == current pose -> zero target.
    if obj_pose_ref is not None:
        Tref = pose_to_se3(obj_pose_ref)
        p_ref = p_local @ Tref.rotation.T + Tref.translation   # (M, 3) reference world
    else:
        p_ref = p_w0

    z = np.array([0.0, 0.0, 1.0])
    Pi0 = np.eye(3) - np.outer(z, z)
    # Normalise by the CONTACT PATCH (active points), not the full surface-sample count M:
    # the floor term must not be diluted by points sampled away from the floor (top/sides),
    # so lambda_d_obj/lambda_x_obj stay comparable to the robot's per-link-normalised lambdas.
    w = alpha[active] / (margin * margin * active.size)     # (na,) per-point weight

    A_D = np.empty((active.size, 6))          # D: z^T [I,-skew(p0)] per point
    c_D = np.empty(active.size)               # D target: ref height - current height
    AX_blocks = []                            # X: Pi0 [I,-skew(p0)] per point
    rX_blocks = []                            # X target: Pi0 (ref footprint - current)
    for k, i in enumerate(active):
        Ji = np.hstack([np.eye(3), -_skew(p_w0[i])])        # (3, 6)
        sd = np.sqrt(lambda_d_obj * w[k])
        A_D[k] = sd * (z @ Ji)
        c_D[k] = sd * float(p_ref[i, 2] - p_w0[i, 2])
        sx = np.sqrt(lambda_x_obj * w[k])
        AX_blocks.append(sx * (Pi0 @ Ji))
        rX_blocks.append(sx * (Pi0 @ (p_ref[i] - p_w0[i])))
    A_X = np.vstack(AX_blocks)                                # (3*na, 6)
    r_X = np.concatenate(rX_blocks)                           # (3*na,)

    terms = []
    if lambda_d_obj > 0:
        terms.append(cp.sum_squares(A_D @ dxi - c_D))
    if lambda_x_obj > 0:
        terms.append(cp.sum_squares(A_X @ dxi - r_X))
    return terms


def build_object_floor_persistence(rt, dxi, obj_pose, obj_pose_prev, ref_t, ref_tm1,
                                   lambda_p_obj, sigma_v, margin, dt):
    """Object<->floor persistence P (the object-environment no-slip pair), symmetric
    to the robot P (interaction.build_p_terms). Near-floor object surface points
    resist tangential slip on the floor between frames, tracking the REFERENCE
    object's tangential displacement (static source -> no-slip; sliding source ->
    reproduce the slide). Coupled through the object pose variable dxi.

        residual_i = Π0 · ( Δp_obj_i(dxi) − Δp_ref_i ),
        Δp_obj_i  = p_i(dxi) − p_prev_i      (current solved-history displacement),
        Δp_ref_i  = p_ref_t_i − p_ref_{t-1}_i (reference object displacement),
        p_i(dxi) ≈ p_i0 + [I, −skew(p_i0)] · dxi,   Π0 = I − z zᵀ.

    Normalized by (σ_v·Δt)² (the faithful per-frame slide, like the robot P), and
    activated by floor proximity (acts only on near-floor points).

    Args:
        rt: retargeter (provides rt.object_surface_local).
        dxi: cp.Variable(6), object SE(3) tangent step.
        obj_pose: (7,) object pose linearization point at t.
        obj_pose_prev: (7,) SOLVED object pose at t-1 (the no-slip anchor).
        ref_t, ref_tm1: (7,) reference object poses at t and t-1.
        lambda_p_obj, sigma_v, margin, dt: weight / slide scale / floor band / timestep.

    Returns:
        List with one cvxpy expression (empty if no active near-floor points or weight 0).
    """
    import cvxpy as cp
    from HoloNew.src.test_socp.interaction import _activation, _skew, _p_scale_sq

    if lambda_p_obj <= 0:
        return []
    p_local = rt.object_surface_local                       # (M, 3)
    M = p_local.shape[0]
    T0 = pose_to_se3(obj_pose)
    p_w0 = p_local @ T0.rotation.T + T0.translation
    alpha = np.array([_activation(float(p_w0[i, 2]), margin) for i in range(M)])
    active = np.where(alpha > 0)[0]
    if active.size == 0:
        return []

    Tprev = pose_to_se3(obj_pose_prev)
    p_prev = p_local @ Tprev.rotation.T + Tprev.translation
    Trt = pose_to_se3(ref_t)
    Trtm1 = pose_to_se3(ref_tm1)
    p_ref_t = p_local @ Trt.rotation.T + Trt.translation
    p_ref_tm1 = p_local @ Trtm1.rotation.T + Trtm1.translation

    z = np.array([0.0, 0.0, 1.0])
    Pi0 = np.eye(3) - np.outer(z, z)
    scale_sq = _p_scale_sq(lambda_p_obj, sigma_v, dt)

    B_rows, r_rows = [], []
    for i in active:
        Bi = np.hstack([np.eye(3), -_skew(p_w0[i])])        # (3, 6)
        dp_obj_const = p_w0[i] - p_prev[i]
        dp_ref = p_ref_t[i] - p_ref_tm1[i]
        # Normalise by the contact patch (active points), not the full sample count M,
        # so the no-slip term isn't diluted by off-floor surface points (mirrors D/X).
        sw = float(np.sqrt(scale_sq * alpha[i] / active.size))
        B_rows.append(sw * (Pi0 @ Bi))                      # (3, 6)
        # residual = Π0·(dp_obj_const + Bi·dxi − dp_ref); rhs = Π0·(dp_ref − dp_obj_const)
        r_rows.append(sw * (Pi0 @ (dp_ref - dp_obj_const)))
    B = np.vstack(B_rows)
    r = np.concatenate(r_rows)
    return [cp.sum_squares(B @ dxi - r)]


def build_wo_term(
    T_obj0,
    T_obj_tm1,
    T_obj_tm2,
    vdot_ref,
    omega_ref,
    dxi,
    lambda_o,
    dt,
    sigma_ao=1.0,
    sigma_omega=1.0,
):
    """W^o: single lambda_o weight with sigma_ao/sigma_omega carrying the
    linear/angular asymmetry.

    cost = lambda_o * ( ||(vdot - vdot_ref) / sigma_ao||^2
                      + ||(omega - omega_ref) / sigma_omega||^2 )

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
        lambda_o: single weight on both the linear acceleration and angular
            velocity terms. The linear/angular asymmetry is carried by sigma_ao
            and sigma_omega.
        dt: timestep in seconds.
        sigma_ao: characteristic scale for the linear acceleration residual
            (divides the linear residual). Default 1.0 (no scaling).
        sigma_omega: characteristic scale for the angular velocity residual
            (divides the angular residual). Default 1.0 (no scaling).

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

    r1 = (np.sqrt(lambda_o) / sigma_ao) * (A_vdot @ dxi + b_vdot)
    r2 = (np.sqrt(lambda_o) / sigma_omega) * (A_omega @ dxi + b_omega)
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
