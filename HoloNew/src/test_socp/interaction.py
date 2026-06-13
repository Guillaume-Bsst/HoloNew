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
        Tuple (d_obj_ref, x_obj_ref, d_flr_ref, x_flr_ref) where:
            d_obj_ref (M,): object signed distance at each human correspondence point
                (object-local frame), indexed by correspondence.human_idx.
            x_obj_ref (M, 3): object closest-surface witness at each human correspondence
                point (object-local frame).
            d_flr_ref (M,): floor signed distance at each probe world point, indexed by
                correspondence.human_idx.
            x_flr_ref (M, 3): floor closest-surface witness at each probe world point.
    """
    pf = rt.smplx_ground_probe(t, rt.human_quat[t], rt.gmr_grounded[:, 0][t])
    hi = rt.correspondence.human_idx
    fflr = floor_field(pf.points, rt.smplx_ground_probe.margin)
    return (pf.field.distance[hi], pf.field.witness[hi],
            fflr.distance[hi], fflr.witness[hi])


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

def build_dx_terms(rt, q_pin: np.ndarray, dqa, t: int,
                   obj_pose: np.ndarray,
                   lambda_D: float, lambda_X: float) -> list:
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

    Args:
        rt: TestSocpRetargeter instance with correspondence, object_sdf, pin, smplx_ground_probe.
        q_pin: Pinocchio configuration vector.
        dqa: cvxpy Variable of shape (nv_a,) — the active-tangent step.
        t: Frame index (for frame_references).
        obj_pose: (7,) [qw, qx, qy, qz, x, y, z] object pose.
        lambda_D: Weight for the normal-proximity (D) term.
        lambda_X: Weight for the tangential-placement (X) term.

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
    d_obj_ref, x_obj_ref, d_flr_ref, x_flr_ref = frame_references(rt, t)

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

    terms = []
    I3 = np.eye(3)

    # --- Object channel (object-local frame) ---
    for i in np.where(active_obj)[0]:
        alpha = alpha_obj[i]
        w = alpha / (L ** 2 * Nk[i])
        Ji = jacs[idx_to_pos[i]]
        Jloc = Robj.T @ Ji                   # (3, nv_a): object-local point Jacobian
        n0 = np.asarray(fobj.direction[i], dtype=float)   # object-local unit normal
        d0 = float(fobj.distance[i])
        x0 = np.asarray(fobj.witness[i], dtype=float)     # object-local witness

        # D term: (n0^T Jloc dqa - (d_ref - d0))^2
        if lambda_D > 0:
            res_d = n0 @ (Jloc @ dqa) - float(d_obj_ref[i] - d0)
            terms.append((lambda_D * w) * cp.square(res_d))

        # X term: || Pi0 (Jloc dqa) - Pi0 (x_ref - x0) ||^2
        if lambda_X > 0:
            Pi0 = I3 - np.outer(n0, n0)
            rhs_x = Pi0 @ (np.asarray(x_obj_ref[i], dtype=float) - x0)
            res_x = Pi0 @ (Jloc @ dqa) - rhs_x
            terms.append((lambda_X * w) * cp.sum_squares(res_x))

    # --- Floor channel (world frame) ---
    for i in np.where(active_flr)[0]:
        alpha = alpha_flr[i]
        w = alpha / (L ** 2 * Nk[i])
        Ji = jacs[idx_to_pos[i]]              # (3, nv_a): world-frame point Jacobian
        n0 = np.asarray(fflr.direction[i], dtype=float)   # world-frame unit normal
        d0 = float(fflr.distance[i])
        x0 = np.asarray(fflr.witness[i], dtype=float)     # world-frame floor witness

        # D term: (n0^T Ji dqa - (d_ref - d0))^2
        if lambda_D > 0:
            res_d = n0 @ (Ji @ dqa) - float(d_flr_ref[i] - d0)
            terms.append((lambda_D * w) * cp.square(res_d))

        # X term: || Pi0 (Ji dqa) - Pi0 (x_ref - x0) ||^2
        if lambda_X > 0:
            Pi0 = I3 - np.outer(n0, n0)
            rhs_x = Pi0 @ (np.asarray(x_flr_ref[i], dtype=float) - x0)
            res_x = Pi0 @ (Ji @ dqa) - rhs_x
            terms.append((lambda_X * w) * cp.sum_squares(res_x))

    return terms
