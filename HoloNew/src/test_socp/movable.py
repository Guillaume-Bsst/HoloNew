"""Movable-entity W^o (object motion regularization) for TEST-SOCP.
See docs/specs/2026-06-13-brick5-movable-entities-design.md."""
from __future__ import annotations

import numpy as np
import pinocchio as pin

from HoloNew.src.test_socp.solve.spec import ResidualBlock


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


def ground_object_pose(obj_poses_scaled, object_surface_local, dataset):
    """Ground the object onto z=0 with one constant per-clip z-shift (HODome only).

    Mirrors the human floor correction on the object side: lift the (already XY/Z
    scaled) object so its lowest surface point over the whole clip rests on z=0 — the
    nominal floor the robot stands on. HODome reconstructs the human (SMPL-X/RGB) and the
    object (optitrack) in frames whose floors disagree by a few cm, leaving the object
    below z=0; OMOMO objects are already floor-consistent (and golden-locked), so the
    shift is gated to HODome.

    Args:
        obj_poses_scaled: (T, 7) [qw, qx, qy, qz, x, y, z], XY/Z scaling already applied.
        object_surface_local: (M, 3) object-local surface samples, or None.
        dataset: dataset key; the shift is applied only when it equals "hodome".

    Returns:
        (grounded (T, 7), shift float). shift == 0.0 and poses are returned unchanged
        when dataset != "hodome" or object_surface_local is None.
    """
    poses = np.asarray(obj_poses_scaled).copy()
    if dataset != "hodome" or object_surface_local is None:
        return poses, 0.0
    from scipy.spatial.transform import Rotation as _Rot
    osl = np.asarray(object_surface_local, dtype=float)
    fsamp = np.unique(np.linspace(0, len(poses) - 1, min(len(poses), 60)).astype(int))
    obj_low = min(
        float((osl @ _Rot.from_quat(poses[f, [1, 2, 3, 0]]).as_matrix().T
               + poses[f, 4:7])[:, 2].min())
        for f in fsamp
    )
    shift = float(-obj_low)
    poses[:, 6] += shift
    return poses, shift


# ---------------------------------------------------------------------------
# ResidualBlock versions (numpy, solver-agnostic)
# ---------------------------------------------------------------------------

def build_wo_block(
    T_obj0,
    T_obj_tm1,
    T_obj_tm2,
    vdot_ref,
    omega_ref,
    nv_a,
    lambda_o,
    dt,
    sigma_ao=1.0,
    sigma_omega=1.0,
):
    """ResidualBlock version of build_wo_term.

    Same linearization as build_wo_term; no cvxpy Variable arguments.
    Returns two ResidualBlocks (one for the linear-acceleration residual,
    one for the angular-velocity residual).

    The object-tangent step dxi has 6 columns (n_obj=6).  The robot-step
    columns A are zeros because W^o has no dqa dependence.

    Block 1 (W_o_lin):
        A_obj = (sqrt(lambda_o) / sigma_ao)  * A_vdot    (3, 6)
        c     = (sqrt(lambda_o) / sigma_ao)  * b_vdot    (3,)
        A     = zeros((3, nv_a))

    Block 2 (W_o_ang):
        A_obj = (sqrt(lambda_o) / sigma_omega) * A_omega  (3, 6)
        c     = (sqrt(lambda_o) / sigma_omega) * b_omega  (3,)
        A     = zeros((3, nv_a))

    Args:
        T_obj0: pin.SE3, current object pose (linearization point).
        T_obj_tm1: pin.SE3, object pose at t-1.
        T_obj_tm2: pin.SE3, object pose at t-2.
        vdot_ref: (3,) linear acceleration reference.
        omega_ref: (3,) angular velocity reference.
        nv_a: number of actuated robot tangent DOF (sets A column count).
        lambda_o: weight on both residuals.
        dt: timestep in seconds.
        sigma_ao: scale for the linear acceleration residual.
        sigma_omega: scale for the angular velocity residual.

    Returns:
        list[ResidualBlock] with two elements.
    """
    M0 = T_obj_tm1.inverse() * T_obj0
    v0 = pin.log6(M0).vector / dt
    J = (pin.Jlog6(M0) @ T_obj0.inverse().action) / dt

    v_tm1 = pin.log6(T_obj_tm2.inverse() * T_obj_tm1).vector / dt

    A_vdot = J[:3, :] / dt                                  # (3, 6)
    b_vdot = (v0[:3] - v_tm1[:3]) / dt - np.asarray(vdot_ref)  # (3,)
    A_omega = J[3:6, :]                                      # (3, 6)
    b_omega = v0[3:6] - np.asarray(omega_ref)               # (3,)

    s_lin = np.sqrt(lambda_o) / sigma_ao
    s_ang = np.sqrt(lambda_o) / sigma_omega

    zeros3 = np.zeros((3, nv_a))
    return [
        ResidualBlock(
            A=zeros3,
            c=s_lin * b_vdot,
            A_obj=s_lin * A_vdot,
            name="W_o_lin",
        ),
        ResidualBlock(
            A=zeros3,
            c=s_ang * b_omega,
            A_obj=s_ang * A_omega,
            name="W_o_ang",
        ),
    ]


def build_wo_position_anchor_block(T_obj0, p_ref, nv_a, lambda_o_pos):
    """ResidualBlock version of build_wo_position_anchor.

    Residual = sqrt(lambda_o_pos) * (A_pos @ dxi + b_pos)  where
        A_pos = [I3 | -skew(p0)]  (3, 6),
        b_pos = p0 - p_ref        (3,).

    Block (W_o_pos):
        A     = zeros((3, nv_a))
        A_obj = sqrt(lambda_o_pos) * A_pos    (3, 6)
        c     = sqrt(lambda_o_pos) * b_pos    (3,)

    Args:
        T_obj0: pin.SE3, current object pose (linearization point).
        p_ref: (3,) reference object position to anchor to.
        nv_a: number of actuated robot tangent DOF (sets A column count).
        lambda_o_pos: weight on the position anchor.

    Returns:
        list[ResidualBlock] with one element.
    """
    p0 = np.asarray(T_obj0.translation, dtype=float)
    A_pos = np.hstack([np.eye(3), -pin.skew(p0)])          # (3, 6)
    b_pos = p0 - np.asarray(p_ref, dtype=float)             # (3,)
    s = np.sqrt(lambda_o_pos)
    return [
        ResidualBlock(
            A=np.zeros((3, nv_a)),
            c=s * b_pos,
            A_obj=s * A_pos,
            name="W_o_pos",
        )
    ]


def build_object_floor_blocks(rt, obj_pose, lambda_d_obj, lambda_x_obj, margin,
                               obj_pose_ref=None):
    """ResidualBlock version of build_object_floor_terms.

    Same coefficient computation as the original; no cvxpy Variable arguments.
    nv_a is taken from rt.nv_a.

    D block (obj_floor_D):
        A     = zeros((na, nv_a))
        A_obj = A_D                           (na, 6)
        c     = -c_D                          (na,)
      (original: cp.sum_squares(A_D @ dxi - c_D)  =>  ‖A_obj·dxi + c‖² )

    X block (obj_floor_X):
        A     = zeros((3*na, nv_a))
        A_obj = A_X                           (3*na, 6)
        c     = -r_X                          (3*na,)
      (original: cp.sum_squares(A_X @ dxi - r_X)  =>  ‖A_obj·dxi + c‖² )

    Args:
        rt: retargeter (provides rt.object_surface_local and rt.nv_a).
        obj_pose: (7,) [qw,qx,qy,qz,x,y,z] object pose (linearization point).
        lambda_d_obj: object-floor normal-proximity (D) weight (0 disables).
        lambda_x_obj: object-floor tangential-placement (X) weight (0 disables).
        margin: floor activation band L (metres).
        obj_pose_ref: (7,) reference object pose. None -> zero target.

    Returns:
        list[ResidualBlock] with 0–2 elements.
    """
    from HoloNew.src.test_socp.interaction import _activation, _skew

    p_local = rt.object_surface_local                        # (M, 3)
    M = p_local.shape[0]
    T0 = pose_to_se3(obj_pose)
    p_w0 = p_local @ T0.rotation.T + T0.translation          # (M, 3)
    d0 = p_w0[:, 2]

    alpha = np.array([_activation(float(d0[i]), margin) for i in range(M)])
    active = np.where(alpha > 0)[0]
    if active.size == 0:
        return []

    if obj_pose_ref is not None:
        Tref = pose_to_se3(obj_pose_ref)
        p_ref = p_local @ Tref.rotation.T + Tref.translation
    else:
        p_ref = p_w0

    z = np.array([0.0, 0.0, 1.0])
    Pi0 = np.eye(3) - np.outer(z, z)
    w = alpha[active] / (margin * margin * active.size)

    nv_a = rt.nv_a
    A_D = np.empty((active.size, 6))
    c_D = np.empty(active.size)
    AX_blocks = []
    rX_blocks = []
    for k, i in enumerate(active):
        Ji = np.hstack([np.eye(3), -_skew(p_w0[i])])         # (3, 6)
        sd = np.sqrt(lambda_d_obj * w[k])
        A_D[k] = sd * (z @ Ji)
        c_D[k] = sd * float(p_ref[i, 2] - p_w0[i, 2])
        sx = np.sqrt(lambda_x_obj * w[k])
        AX_blocks.append(sx * (Pi0 @ Ji))
        rX_blocks.append(sx * (Pi0 @ (p_ref[i] - p_w0[i])))
    A_X = np.vstack(AX_blocks)                                # (3*na, 6)
    r_X = np.concatenate(rX_blocks)                           # (3*na,)

    blocks = []
    if lambda_d_obj > 0:
        blocks.append(ResidualBlock(
            A=np.zeros((active.size, nv_a)),
            c=-c_D,
            A_obj=A_D,
            name="obj_floor_D",
        ))
    if lambda_x_obj > 0:
        blocks.append(ResidualBlock(
            A=np.zeros((A_X.shape[0], nv_a)),
            c=-r_X,
            A_obj=A_X,
            name="obj_floor_X",
        ))
    return blocks


def build_object_floor_persistence_blocks(rt, obj_pose, obj_pose_prev, ref_t, ref_tm1,
                                          lambda_p_obj, sigma_v, margin, dt):
    """ResidualBlock version of build_object_floor_persistence.

    Same coefficient computation as the original; no cvxpy Variable arguments.
    nv_a is taken from rt.nv_a.

    P block (obj_floor_P):
        A     = zeros((3*na, nv_a))
        A_obj = B                             (3*na, 6)
        c     = -r                            (3*na,)
      (original: cp.sum_squares(B @ dxi - r)  =>  ‖A_obj·dxi + c‖² )

    Args:
        rt: retargeter (provides rt.object_surface_local and rt.nv_a).
        obj_pose: (7,) object pose linearization point at t.
        obj_pose_prev: (7,) SOLVED object pose at t-1.
        ref_t, ref_tm1: (7,) reference object poses at t and t-1.
        lambda_p_obj, sigma_v, margin, dt: weight / slide scale / floor band / timestep.

    Returns:
        list[ResidualBlock] with 0–1 elements.
    """
    from HoloNew.src.test_socp.interaction import _activation, _skew, _p_scale_sq

    if lambda_p_obj <= 0:
        return []
    p_local = rt.object_surface_local                        # (M, 3)
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

    nv_a = rt.nv_a
    B_rows, r_rows = [], []
    for i in active:
        Bi = np.hstack([np.eye(3), -_skew(p_w0[i])])         # (3, 6)
        dp_obj_const = p_w0[i] - p_prev[i]
        dp_ref = p_ref_t[i] - p_ref_tm1[i]
        sw = float(np.sqrt(scale_sq * alpha[i] / active.size))
        B_rows.append(sw * (Pi0 @ Bi))
        r_rows.append(sw * (Pi0 @ (dp_ref - dp_obj_const)))
    B = np.vstack(B_rows)
    r = np.concatenate(r_rows)
    return [
        ResidualBlock(
            A=np.zeros((B.shape[0], nv_a)),
            c=-r,
            A_obj=B,
            name="obj_floor_P",
        )
    ]
