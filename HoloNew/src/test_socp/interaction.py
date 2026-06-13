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
