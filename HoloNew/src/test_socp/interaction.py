"""Per-frame interaction data and D/X/P residual assembly for TEST-SOCP.

Queries the object SDF and floor fields at the robot control points (the G1 side
of the correspondence), extracts the source references, and builds the cvxpy
normal-proximity (D), tangential-placement (X) and persistence (P) terms. See
docs/specs/2026-06-13-brick1-interaction-dxp-design.md.
"""
from __future__ import annotations

import numpy as np

from HoloNew.src.holosoma.interaction_mesh import transform_points_world_to_local
from HoloNew.src.test_socp.contact.backends.floor import floor_field


def _world_to_object_local(pts_world: np.ndarray, obj_pose: np.ndarray) -> np.ndarray:
    """Transform world-frame points to the object-local frame.

    Mirrors the exact convention used in smplx_field.SmplxGroundProbe.__call__:
    ``transform_points_world_to_local(obj_quat[t], obj_trans[t], world)``, which
    builds a 4x4 homogeneous matrix from [qw, qx, qy, qz] and then inverts it.

    Args:
        pts_world: (N, 3) points in the world frame.
        obj_pose: (7,) array [qw, qx, qy, qz, x, y, z].

    Returns:
        (N, 3) points in the object-local frame.
    """
    quat = obj_pose[:4]    # [qw, qx, qy, qz]
    trans = obj_pose[4:7]  # [x, y, z]
    return transform_points_world_to_local(quat, trans, pts_world)


def robot_control_points(rt, q_pin: np.ndarray) -> np.ndarray:
    """(M, 3) world positions of the G1 correspondence control points at q_pin.

    For point i, the world position is:
        body_position(link) + body_rotation(link) @ offset_local[i]
    where link = correspondence.link_names[link_idx[i]].

    Args:
        rt: TestSocpRetargeter instance (must have rt.correspondence and rt.pin).
        q_pin: Pinocchio configuration vector from rt.pin.qpos_mj_to_q_pin(q[:36]).

    Returns:
        Array of shape (M, 3) in the world frame.
    """
    corr = rt.correspondence
    M = corr.link_idx.shape[0]
    out = np.zeros((M, 3))
    # One FK pass for all links; assemble points per link group (vectorized).
    placements = rt.pin.link_placements(q_pin, corr.link_names)
    for li, name in enumerate(corr.link_names):
        mask = corr.link_idx == li
        if not mask.any():
            continue
        Rw, pw = placements[name]
        out[mask] = pw + corr.offset_local[mask] @ Rw.T
    return out


def frame_references(rt, t: int):
    """Per-control-point source references at frame t, indexed by correspondence.human_idx.

    Calls the source probe at the same arguments retarget() uses so that the returned
    fields are consistent with the per-frame solve context.

    Args:
        rt: TestSocpRetargeter instance (must have smplx_ground_probe, human_quat,
            gmr_grounded, and correspondence set).
        t: Frame index.

    Returns:
        Tuple (d_obj_ref, x_obj_ref, d_flr_ref, x_flr_ref, p_ref) where:
            d_obj_ref (M,): object signed distance at each human correspondence point
                (object-local frame), indexed by correspondence.human_idx.
            x_obj_ref (M, 3): object closest-surface witness at each human correspondence
                point (object-local frame).
            d_flr_ref (M,): floor signed distance at each probe world point, indexed by
                correspondence.human_idx.
            x_flr_ref (M, 3): floor closest-surface witness at each probe world point.
            p_ref (M, 3): world positions of the source probe points at each human
                correspondence index (pf.points[human_idx]). Used by the P term to
                compute the source tangential displacement Δp_ref = p_ref_t - p_ref_{t-1}.

    The source references are constant within a frame (they depend only on the
    source motion, not the robot config), but build_dx_terms / build_p_terms call
    this once per SQP iteration. A small per-frame memo (keeping the two most
    recent frames so persistence can read t and t-1) avoids re-running the
    expensive SMPL-X probe every inner iteration.
    """
    cache = rt.__dict__.setdefault("_frame_ref_cache", {})
    if t in cache:
        return cache[t]
    pf = rt.smplx_ground_probe(t, rt.human_quat[t], rt.gmr_grounded[:, 0][t])
    hi = rt.correspondence.human_idx
    fflr = floor_field(pf.points, rt.smplx_ground_probe.margin)
    refs = (pf.field.distance[hi], pf.field.witness[hi],
            fflr.distance[hi], fflr.witness[hi],
            np.asarray(pf.points, dtype=np.float64)[hi])   # (M, 3) world source points
    cache[t] = refs
    for old in [k for k in cache if k < t - 1]:   # keep only t and t-1
        del cache[old]
    return refs


def query_entities(rt, pts_world: np.ndarray, obj_pose: np.ndarray,
                   margin: float | None = None):
    """Query the object SDF and floor at robot control points.

    Returns:
        (fobj, fflr): object ContactField (object-local frame) and floor
        ContactField (world frame). Both have shape (M,) / (M, 3) fields.

    Args:
        rt: TestSocpRetargeter instance.
        pts_world: (M, 3) world positions of the robot control points.
        obj_pose: (7,) [qw, qx, qy, qz, x, y, z] object pose, as stored in
            smplx_ground_probe.obj_quat[t] / obj_trans[t].
        margin: SDF query band half-width. Defaults to
            rt.smplx_ground_probe.margin when the probe is available.
    """
    if margin is None:
        margin = rt.smplx_ground_probe.margin

    # Object SDF: query in the object-local frame (mirrors smplx_field exactly).
    pts_local = _world_to_object_local(pts_world, obj_pose)
    fobj = rt.object_sdf.query(pts_local, margin)

    # Floor: analytic z=0 field in the world frame.
    fflr = floor_field(pts_world.astype(np.float32), margin)

    return fobj, fflr


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _activation(d_ref: float, L: float) -> float:
    """Scalar activation weight alpha(d_ref) = clamp((1 - d_ref/L)^2, 0).

    Returns 0 when d_ref >= L (point outside the margin band).
    """
    if d_ref >= L:
        return 0.0
    a = 1.0 - d_ref / L
    return max(a, 0.0) ** 2


def _robj_from_pose(obj_pose: np.ndarray) -> np.ndarray:
    """Extract the 3x3 rotation matrix R_obj (object-to-world) from obj_pose.

    Uses the same wxyz convention as transform_points_world_to_local / trimesh:
    obj_pose[:4] = [qw, qx, qy, qz].  trimesh.transformations.quaternion_matrix
    interprets this as scalar-first and returns a 4x4 homogeneous matrix whose
    top-left 3x3 is R_obj (columns are the object axes in world coordinates).

    To map world→object-local: x_loc = R_obj.T @ (x_world - t).
    """
    import trimesh
    return np.asarray(
        trimesh.transformations.quaternion_matrix(obj_pose[:4])[:3, :3]
    )


# ---------------------------------------------------------------------------
# D + X residual assembly
# ---------------------------------------------------------------------------

def _skew(v: np.ndarray) -> np.ndarray:
    """3x3 skew-symmetric matrix for cross-product: skew(v) @ w == v x w."""
    v = np.asarray(v, dtype=float).ravel()
    return np.array([
        [     0.0, -v[2],  v[1]],
        [  v[2],    0.0, -v[0]],
        [ -v[1],  v[0],   0.0],
    ])


def build_dx_terms(rt, q_pin: np.ndarray, dqa, t: int,
                   obj_pose: np.ndarray,
                   lambda_D: float, lambda_X: float,
                   dxi=None) -> list:
    """Build cvxpy D (normal proximity) and X (tangential placement) residual terms.

    Uses an active-set strategy: only points where alpha > 0 AND the robot-side
    field is active are included, keeping the active set small (typically << M).
    All active point Jacobians are computed in one batched pinocchio pass.

    Per-carrier normalization 1/N_k (N_k = number of correspondence points on
    link k) is applied so each link contributes equal weight regardless of its
    point density.

    Object-channel residuals are expressed in the object-local frame:
        J_loc = R_obj.T @ J_world  (object driven => object fixed in the per-frame solve)
    Floor-channel residuals are expressed in the world frame.

    When ``dxi`` is provided (a cvxpy Variable of shape (6,) — the object SE(3)
    tangent step in the world frame), the object-channel residuals become
    bilateral: the object-local relative displacement of each control point
    gains the object rigid-motion contribution:

        Bobj_i = R_obj.T @ [I_3 | -skew(p_i)]      (3, 6)

    where ``p_i`` is the world position of the control point (NOT p_i - t_obj;
    see derivation note in docs). The object-DOF contribution is subtracted:
        relative_disp_local = R_obj.T @ (J_i @ dqa - [I, -skew(p_i)] @ dxi)
                            = (R_obj.T @ J_i) @ dqa  -  Bobj_i @ dxi

    The floor channel is unchanged (floor is not a movable entity).

    Args:
        rt: TestSocpRetargeter instance with correspondence, object_sdf, pin, smplx_ground_probe.
        q_pin: Pinocchio configuration vector.
        dqa: cvxpy Variable of shape (nv_a,) — the active-tangent step.
        t: Frame index (for frame_references).
        obj_pose: (7,) [qw, qx, qy, qz, x, y, z] object pose.
        lambda_D: Weight for the normal-proximity (D) term.
        lambda_X: Weight for the tangential-placement (X) term.
        dxi: Optional cvxpy Variable of shape (6,) — the object SE(3) tangent
            step (world-frame left perturbation). When None (default), the
            object channel is robot-only (current behaviour).

    Returns:
        List of cvxpy scalar expressions (one per active point per entity per
        channel). May be empty if no points are active.
    """
    import cvxpy as cp

    corr = rt.correspondence
    M = corr.link_idx.shape[0]
    L = rt.smplx_ground_probe.margin

    # Precompute per-link point counts for 1/N_k normalization.
    n_links = len(corr.link_names)
    link_counts = np.zeros(n_links, dtype=float)
    for li in range(n_links):
        link_counts[li] = float(np.sum(corr.link_idx == li))
    # N_k for each point i.
    Nk = link_counts[corr.link_idx]  # (M,)

    # World positions of all robot control points.
    P = robot_control_points(rt, q_pin)

    # Robot-side field queries.
    fobj, fflr = query_entities(rt, P, obj_pose, margin=L)

    # Source-side references (object-local frame / world frame).
    # p_ref (world source points) is used only by build_p_terms; ignored here.
    d_obj_ref, x_obj_ref, d_flr_ref, x_flr_ref, _p_ref = frame_references(rt, t)

    # Object-to-world rotation (Robj[:, j] = j-th object axis in world).
    Robj = _robj_from_pose(obj_pose)

    # --- Active-set selection ---
    # Object channel: alpha > 0 AND field marked active.
    alpha_obj = np.array([_activation(d_obj_ref[i], L) for i in range(M)])
    active_obj = (alpha_obj > 0) & np.asarray(fobj.active, dtype=bool)

    # Floor channel: alpha > 0 AND field marked active.
    alpha_flr = np.array([_activation(d_flr_ref[i], L) for i in range(M)])
    active_flr = (alpha_flr > 0) & np.asarray(fflr.active, dtype=bool)

    # Union of active indices for a single batched Jacobian pass.
    active_union = np.where(active_obj | active_flr)[0]

    if active_union.size == 0:
        return []

    # Batched Jacobian computation (one computeJointJacobians pass).
    link_names_active = [corr.link_names[corr.link_idx[i]] for i in active_union]
    offsets_active = corr.offset_local[active_union]  # (K, 3)
    jacs_full = rt.pin.point_jacobians(q_pin, link_names_active, offsets_active)
    # Slice to active tangent indices (nv_a columns).
    jacs = [J[:, rt.v_a_indices] for J in jacs_full]

    # Map global index -> position in active_union for Jacobian lookup.
    idx_to_pos = {int(active_union[k]): k for k in range(len(active_union))}

    I3 = np.eye(3)

    # Separate object-channel and floor-channel rows so the dxi block can be
    # added only to the object channel (floor is not movable).
    # D channel: one scalar row per active object point.
    A_d_obj_rows, c_d_obj_rows = [], []   # robot-DOF columns (nv_a)
    Adxi_d_rows = []                       # object-DOF columns (6); only when dxi given
    # X channel: three rows per active object point.
    B_x_obj_rows, r_x_obj_rows = [], []
    Bdxi_x_rows = []

    # Floor D / X rows (robot DOF only; floor is not movable).
    A_d_flr_rows, c_d_flr_rows = [], []
    B_x_flr_rows, r_x_flr_rows = [], []

    # --- Object channel (object-local frame) ---
    for i in np.where(active_obj)[0]:
        alpha = alpha_obj[i]
        w = alpha / (L ** 2 * Nk[i])
        Ji = jacs[idx_to_pos[i]]
        Jloc = Robj.T @ Ji                   # (3, nv_a): object-local point Jacobian
        n0 = np.asarray(fobj.direction[i], dtype=float)   # object-local unit normal
        d0 = float(fobj.distance[i])
        x0 = np.asarray(fobj.witness[i], dtype=float)     # object-local witness

        # Object-DOF block for bilateral coupling (when dxi is given).
        # Bobj_i = R_obj.T @ [I_3, -skew(p_i)]  where p_i = P[i] is the world
        # position of the control point.  This block is SUBTRACTED from the
        # robot-DOF Jacobian contribution (see docstring derivation).
        if dxi is not None:
            Bobj_i = Robj.T @ np.hstack([I3, -_skew(P[i])])  # (3, 6)

        # D: sqrt(lambda_D * w) * n0^T Jloc  vs  sqrt(lambda_D * w) * (d_ref - d0)
        if lambda_D > 0:
            sw = float(np.sqrt(lambda_D * w))
            A_d_obj_rows.append(sw * (n0 @ Jloc))             # (nv_a,)
            c_d_obj_rows.append(sw * float(d_obj_ref[i] - d0))
            if dxi is not None:
                Adxi_d_rows.append(sw * (n0 @ Bobj_i))        # (6,)

        # X: sqrt(lambda_X * w) * Pi0 Jloc  vs  sqrt(lambda_X * w) * Pi0(x_ref - x0)
        if lambda_X > 0:
            sw = float(np.sqrt(lambda_X * w))
            Pi0 = I3 - np.outer(n0, n0)
            B_x_obj_rows.append(sw * (Pi0 @ Jloc))            # (3, nv_a)
            r_x_obj_rows.append(sw * Pi0 @ (np.asarray(x_obj_ref[i], dtype=float) - x0))
            if dxi is not None:
                Bdxi_x_rows.append(sw * (Pi0 @ Bobj_i))       # (3, 6)

    # --- Floor channel (world frame) ---
    for i in np.where(active_flr)[0]:
        alpha = alpha_flr[i]
        w = alpha / (L ** 2 * Nk[i])
        Ji = jacs[idx_to_pos[i]]              # (3, nv_a): world-frame point Jacobian
        n0 = np.asarray(fflr.direction[i], dtype=float)   # world-frame unit normal
        d0 = float(fflr.distance[i])
        x0 = np.asarray(fflr.witness[i], dtype=float)     # world-frame floor witness

        # D: sqrt(lambda_D * w) * n0^T Ji  vs  sqrt(lambda_D * w) * (d_ref - d0)
        if lambda_D > 0:
            sw = float(np.sqrt(lambda_D * w))
            A_d_flr_rows.append(sw * (n0 @ Ji))               # (nv_a,)
            c_d_flr_rows.append(sw * float(d_flr_ref[i] - d0))

        # X: sqrt(lambda_X * w) * Pi0 Ji  vs  sqrt(lambda_X * w) * Pi0(x_ref - x0)
        if lambda_X > 0:
            sw = float(np.sqrt(lambda_X * w))
            Pi0 = I3 - np.outer(n0, n0)
            B_x_flr_rows.append(sw * (Pi0 @ Ji))              # (3, nv_a)
            r_x_flr_rows.append(sw * Pi0 @ (np.asarray(x_flr_ref[i], dtype=float) - x0))

    # Build one cvxpy expression per non-empty channel.
    # Object channel: bilateral when dxi is given (A_full=[A_dqa | -A_dxi],
    # variable = cp.hstack([dqa, dxi])), or robot-only otherwise.
    terms = []

    # Object D rows.
    if A_d_obj_rows:
        A_d_obj = np.array(A_d_obj_rows)    # (K_d_obj, nv_a)
        c_d_obj = np.array(c_d_obj_rows)    # (K_d_obj,)
        if dxi is not None and Adxi_d_rows:
            Adxi_d = np.array(Adxi_d_rows)  # (K_d_obj, 6)
            # residual = A_dqa @ dqa - Adxi_d @ dxi - c_d_obj
            terms.append(cp.sum_squares(A_d_obj @ dqa - Adxi_d @ dxi - c_d_obj))
        else:
            terms.append(cp.sum_squares(A_d_obj @ dqa - c_d_obj))

    # Object X rows.
    if B_x_obj_rows:
        B_x_obj = np.vstack(B_x_obj_rows)  # (3*K_x_obj, nv_a)
        r_x_obj = np.concatenate(r_x_obj_rows)
        if dxi is not None and Bdxi_x_rows:
            Bdxi_x = np.vstack(Bdxi_x_rows)  # (3*K_x_obj, 6)
            terms.append(cp.sum_squares(B_x_obj @ dqa - Bdxi_x @ dxi - r_x_obj))
        else:
            terms.append(cp.sum_squares(B_x_obj @ dqa - r_x_obj))

    # Floor D rows (robot-only; floor not movable).
    if A_d_flr_rows:
        A_d_flr = np.array(A_d_flr_rows)   # (K_d_flr, nv_a)
        c_d_flr = np.array(c_d_flr_rows)
        terms.append(cp.sum_squares(A_d_flr @ dqa - c_d_flr))

    # Floor X rows.
    if B_x_flr_rows:
        B_x_flr = np.vstack(B_x_flr_rows)  # (3*K_x_flr, nv_a)
        r_x_flr = np.concatenate(r_x_flr_rows)
        terms.append(cp.sum_squares(B_x_flr @ dqa - r_x_flr))

    return terms


# ---------------------------------------------------------------------------
# P (contact persistence) residual assembly
# ---------------------------------------------------------------------------

def build_p_terms(rt, q_pin: np.ndarray, dqa, t: int,
                  obj_pose: np.ndarray,
                  lambda_P: float, sigma_v: float, dt: float) -> list:
    """Build cvxpy contact persistence (P) residual terms.

    P penalises tangential slip at contact points across consecutive frames.
    The residual is the mismatch between the robot's tangential displacement
    Δp_i and the source's tangential displacement Δp_i^ref, projected by
    Π0 = I − n0 n0ᵀ (the tangential projector at the current field normal).

    Activation γ_i = min(α_i^t, α_i^{t-1}, α̂_i^{t-1}), where:
        α_i^t:     source activation at the current frame (from d_obj/flr_ref[i]).
        α_i^{t-1}: source activation at the previous frame (stored in rt._p_state).
        α̂_i^{t-1}: previous SOLVED robot-side activation (from rt._p_state d_prev).
    Points with γ ≤ 0 or inactive robot-side field are skipped.

    Object channel: residuals in the object-local frame (consistent with how the
    object SDF field direction is expressed).
    Floor channel: residuals in the world frame.

    Args:
        rt: TestSocpRetargeter instance.
        q_pin: Pinocchio configuration vector at the current linearisation point.
        dqa: cvxpy Variable of shape (nv_a,).
        t: Current frame index (must be >= 1).
        obj_pose: (7,) [qw, qx, qy, qz, x, y, z] current object pose.
        lambda_P: Weight for the persistence term.
        sigma_v: Accepted for API compatibility; no longer used in the scale (see
            the normalization note below).
        dt: Accepted for API compatibility; no longer used in the scale.

    Returns:
        List of cvxpy scalar expressions. May be empty if no points have γ > 0.

    Normalization (deliberate divergence from the paper). The paper normalizes P
    by the characteristic per-frame slide (sigma_v * dt)^2. With sigma_v = 0.05 and
    dt = 1/30 that scale is ~1.7 mm, giving a weight ~3.6e5 — ~3600x the D/X terms
    (normalized by the field range L^2 ~ 0.01), which wrecks CLARABEL's
    conditioning and makes the solve fail once P engages. P is a tangential
    meter-residual exactly like X, so we normalize it by the SAME L^2: lambda_P is
    then directly comparable to lambda_X, and no-slip is enforced as a gentle
    tangential prior (the per-frame slide is already bounded by the SQP trust
    region). This keeps the paper's intent (reproduce the source's tangential
    slide) while staying numerically well-conditioned.
    """
    import cvxpy as cp

    state = rt._p_state
    corr = rt.correspondence
    M = corr.link_idx.shape[0]
    L = rt.smplx_ground_probe.margin
    scale_sq = (lambda_P / L ** 2)

    # Per-link point counts for 1/N_k normalisation.
    n_links = len(corr.link_names)
    link_counts = np.zeros(n_links, dtype=float)
    for li in range(n_links):
        link_counts[li] = float(np.sum(corr.link_idx == li))
    Nk = link_counts[corr.link_idx]  # (M,)

    # Current and previous object-to-world rotation matrices.
    Robj_t = _robj_from_pose(obj_pose)              # current frame
    obj_t = np.asarray(obj_pose[4:7], dtype=float)
    obj_prev_pose = state["obj_prev"]
    Robj_tm1 = _robj_from_pose(obj_prev_pose)       # previous frame
    obj_tm1 = np.asarray(obj_prev_pose[4:7], dtype=float)

    # Current world positions of robot control points.
    P = robot_control_points(rt, q_pin)             # (M, 3)

    # Robot-side field at current config.
    fobj, fflr = query_entities(rt, P, obj_pose, margin=L)

    # Source references at t and t-1.
    d_obj_ref_t,  _,  d_flr_ref_t,  _, p_ref_t   = frame_references(rt, t)
    d_obj_ref_tm1, _, d_flr_ref_tm1, _, p_ref_tm1 = frame_references(rt, t - 1)

    # Previous solved robot-side distances (for α̂^{t-1}).
    d_prev_obj = state["d_prev_obj"]   # (M,)
    d_prev_flr = state["d_prev_flr"]  # (M,)
    # Previous source activations α^{t-1}.
    a_prev_obj = state["a_prev_obj"]   # (M,)
    a_prev_flr = state["a_prev_flr"]  # (M,)
    # Previous solved robot-point world positions.
    p_prev_world = state["p_prev_world"]  # (M, 3)

    # Source point displacement in world frame: Δp_ref = p_ref_t - p_ref_{t-1}.
    dp_ref_world = p_ref_t - p_ref_tm1   # (M, 3)

    # --- Activation masks ---
    alpha_obj_t = np.array([_activation(d_obj_ref_t[i], L) for i in range(M)])
    alpha_flr_t = np.array([_activation(d_flr_ref_t[i], L) for i in range(M)])

    def _hat(d_prev_i):
        """Previous solved robot-side activation α̂^{t-1}."""
        return _activation(float(d_prev_i), L)

    # Active sets: γ > 0 AND current robot-side field is marked active.
    gamma_obj = np.minimum(np.minimum(alpha_obj_t, a_prev_obj),
                           np.array([_hat(d_prev_obj[i]) for i in range(M)]))
    active_obj = (gamma_obj > 0) & np.asarray(fobj.active, dtype=bool)

    gamma_flr = np.minimum(np.minimum(alpha_flr_t, a_prev_flr),
                           np.array([_hat(d_prev_flr[i]) for i in range(M)]))
    active_flr = (gamma_flr > 0) & np.asarray(fflr.active, dtype=bool)

    active_union = np.where(active_obj | active_flr)[0]
    if active_union.size == 0:
        return []

    # Batched Jacobian computation.
    link_names_active = [corr.link_names[corr.link_idx[i]] for i in active_union]
    offsets_active = corr.offset_local[active_union]
    jacs_full = rt.pin.point_jacobians(q_pin, link_names_active, offsets_active)
    jacs = [J[:, rt.v_a_indices] for J in jacs_full]
    idx_to_pos = {int(active_union[k]): k for k in range(len(active_union))}

    I3 = np.eye(3)

    # Collect rows for the stacked B_p matrix and r_p vector.
    # Each active point contributes 3 rows: sqrt(scale_sq * w) * Pi0 * Jcoeff
    B_p_rows = []
    r_p_rows = []

    # --- Object channel (object-local frame) ---
    # Robot displacement (object-local):
    #   Δp_local = Robj_t.T @ (P_i + Ji@dqa − obj_t) − p_prev_local_i
    # where p_prev_local_i = Robj_{t-1}.T @ (p_prev_world_i − obj_{t-1}).
    # The constant (non-dqa) part: Robj_t.T @ (P_i − obj_t) − p_prev_local_i.
    # The linear part: Robj_t.T @ Ji @ dqa.
    #
    # Source displacement (object-local):
    #   Δp_ref_local = Robj_t.T @ (p_ref_t_i − obj_t) − Robj_{t-1}.T @ (p_ref_{t-1}_i − obj_{t-1}).
    for i in np.where(active_obj)[0]:
        gamma = gamma_obj[i]
        w = gamma / Nk[i]
        Ji = jacs[idx_to_pos[i]]            # (3, nv_a) world-frame Jacobian
        Jloc = Robj_t.T @ Ji               # (3, nv_a) object-local Jacobian

        n0 = np.asarray(fobj.direction[i], dtype=float)   # object-local unit normal
        Pi0 = I3 - np.outer(n0, n0)

        p_prev_local_i = Robj_tm1.T @ (p_prev_world[i] - obj_tm1)
        const_i = Robj_t.T @ (P[i] - obj_t) - p_prev_local_i  # (3,) constant offset

        p_ref_t_loc = Robj_t.T @ (p_ref_t[i] - obj_t)
        p_ref_tm1_loc = Robj_tm1.T @ (p_ref_tm1[i] - obj_tm1)
        dp_ref_loc = p_ref_t_loc - p_ref_tm1_loc              # (3,)

        # residual = Π0 @ (const_i + Jloc@dqa - dp_ref_loc)
        # rhs = Π0 @ (dp_ref_loc - const_i)
        rhs_const = Pi0 @ (dp_ref_loc - const_i)              # (3,)
        sw = float(np.sqrt(scale_sq * w))
        B_p_rows.append(sw * (Pi0 @ Jloc))                    # (3, nv_a)
        r_p_rows.append(sw * rhs_const)                        # (3,)

    # --- Floor channel (world frame) ---
    # Robot displacement (world): Δp = (P_i - p_prev_world_i) + Ji@dqa.
    # Source displacement (world): Δp_ref = p_ref_t_i - p_ref_{t-1}_i.
    for i in np.where(active_flr)[0]:
        gamma = gamma_flr[i]
        w = gamma / Nk[i]
        Ji = jacs[idx_to_pos[i]]            # (3, nv_a) world-frame Jacobian
        n0 = np.asarray(fflr.direction[i], dtype=float)   # world-frame unit normal
        Pi0 = I3 - np.outer(n0, n0)

        const_i   = P[i] - p_prev_world[i]                  # (3,) constant offset
        dp_ref_i  = dp_ref_world[i]                          # (3,)
        rhs_const = Pi0 @ (dp_ref_i - const_i)              # (3,)

        sw = float(np.sqrt(scale_sq * w))
        B_p_rows.append(sw * (Pi0 @ Ji))                    # (3, nv_a)
        r_p_rows.append(sw * rhs_const)                     # (3,)

    if not B_p_rows:
        return []

    B_p = np.vstack(B_p_rows)          # (3*K, nv_a)
    r_p = np.concatenate(r_p_rows)     # (3*K,)
    return [cp.sum_squares(B_p @ dqa - r_p)]
