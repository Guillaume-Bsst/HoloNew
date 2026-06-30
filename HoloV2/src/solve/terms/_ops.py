"""Complex ops AT THE EXCLUSIVE SERVICE OF THE RESIDUALS — the ``solve``-specific contractions / frame
maps / manifold logs that the spec keeps OUT of the (ref-free) ``targets`` evaluator. Pure numpy
(float64), no I/O, no mutation. Shared by C and CO (rule #8 homogeneity): one ``dist_jac`` contraction,
one ``world_normal`` frame map, one ``so3_log``.

Conventions (locked by ``targets``):
  * Robot point Jacobians (``point_jac``, ``jac_pos``, ``jac_rot``) are WORLD / LOCAL_WORLD_ALIGNED.
  * OBJECT-channel ``direction``/``witness``/geodesic gradients are OBJECT-LOCAL -> map to world with
    ``world_normal(R_i, …)`` before contracting with a world Jacobian; contract with the RAW local
    vector against the object tangent Jacobian ``probe_jac_obj`` (object-local).
  * Orientation residual is the WORLD-frame log ``log(R_cur·R_refᵀ)`` to pair with the world angular
    Jacobian ``jac_rot`` (``omega_world = jac_rot·v``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...prepare.contracts import GeodesicTable


def world_normal(R: np.ndarray, n_local: np.ndarray) -> np.ndarray:
    """Map an object-LOCAL direction/normal/gradient to WORLD: ``n_world = R · n_local``.
    ``R`` (3,3) [one frame] or (M,3,3) [per row]; ``n_local`` (...,3). Ground channel: ``R = I``."""
    R = np.asarray(R, np.float64)
    n = np.asarray(n_local, np.float64)
    return np.einsum("...ij,...j->...i", R, n)


def dist_jac(direction: np.ndarray, jac: np.ndarray) -> np.ndarray:
    """``∂(directionᵀ·point)/∂step`` = ``directionᵀ·jac`` row-wise. ``direction`` (M,3), ``jac``
    (M,3,K) -> (M,K). The signed-distance gradient w.r.t. the point is the contact unit normal, so
    this gives ``∂d/∂step`` for both the robot tangent (K=nv, ``point_jac``) and the object tangent
    (K=6, ``probe_jac_obj`` / ``cloud_jac_self``)."""
    direction = np.asarray(direction, np.float64)
    jac = np.asarray(jac, np.float64)
    return np.einsum("mi,mij->mj", direction, jac)


def geo_chain(grad: np.ndarray, jac: np.ndarray) -> np.ndarray:
    """``∂geo/∂step`` = ``gradᵀ·jac`` — the SAME contraction as ``dist_jac`` (the geodesic gradient is
    tangent to the surface; its normal component, if any, is annihilated by the tangent Jacobian).
    Kept as a named op for builder readability (rule #8)."""
    return dist_jac(grad, jac)


def quat_to_rot(wxyz: np.ndarray) -> np.ndarray:
    """Unit quaternion(s) ``wxyz`` (...,4) -> rotation matrix (...,3,3). Normalises defensively."""
    q = np.asarray(wxyz, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), np.float64)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z); R[..., 0, 1] = 2 * (x * y - z * w); R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w); R[..., 1, 1] = 1 - 2 * (x * x + z * z); R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w); R[..., 2, 1] = 2 * (y * z + x * w); R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _log_one(E: np.ndarray) -> np.ndarray:
    """SO(3) log of a single rotation matrix -> rotation vector (3,). Robust near 0 and π."""
    cos = np.clip((np.trace(E) - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos)
    if theta < 1e-7:                                    # near identity: first-order
        return 0.5 * np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
    if np.pi - theta < 1e-4:                            # near π: axis from the symmetric part
        Aerr = (E + np.eye(3)) * 0.5
        axis = np.sqrt(np.clip(np.diag(Aerr), 0.0, None))
        # fix signs from the off-diagonal of (E - Eᵀ)
        s = np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
        axis = np.where(s < 0, -axis, axis)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return theta * axis
    w = theta / (2.0 * np.sin(theta))
    return w * np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])


def so3_log(R_ref: np.ndarray, R_cur: np.ndarray) -> np.ndarray:
    """World-frame orientation error per row: ``log(R_cur·R_refᵀ)`` (L,3,3),(L,3,3) -> (L,3).
    Gauss-Newton residual ``c = so3_log(R_ref, R_cur)``; the EXACT first-order Jacobian of a world
    (left) bump is ``A = J_l⁻¹(c)·jac_rot``, NOT the raw ``jac_rot`` — the inverse-left-Jacobian factor
    matters at finite error (see ``style._so3_left_jac_inv``; omitting it fails the S-rot FD test)."""
    R_ref = np.asarray(R_ref, np.float64); R_cur = np.asarray(R_cur, np.float64)
    E = np.einsum("lij,lkj->lik", R_cur, R_ref)         # R_cur · R_refᵀ
    return np.stack([_log_one(E[l]) for l in range(E.shape[0])])


def se3_log_world(R_ref: np.ndarray, p_ref: np.ndarray,
                  R_cur: np.ndarray, p_cur: np.ndarray) -> np.ndarray:
    """World-aligned SE(3) error per object: ``[p_cur − p_ref, log(R_cur·R_refᵀ)]`` (N,6). Matches the
    world-aligned object tangent ``δξ = (δt, δθ)`` (the O term anchors the object to its observed pose)."""
    p_ref = np.asarray(p_ref, np.float64); p_cur = np.asarray(p_cur, np.float64)
    out = np.empty((p_ref.shape[0], 6), np.float64)
    out[:, :3] = p_cur - p_ref
    out[:, 3:] = so3_log(R_ref, R_cur)
    return out


def scatter_obj(block: np.ndarray, object_idx: int, n_obj: int) -> np.ndarray:
    """Place a per-object ``(m,6)`` Jacobian block into the full ``(m, n_obj*6)`` object coupling
    matrix (sparse: zeros for the other objects). ``object_idx`` in ``[0, n_obj)``."""
    block = np.asarray(block, np.float64)
    m = block.shape[0]
    A_obj = np.zeros((m, n_obj * 6), np.float64)
    A_obj[:, object_idx * 6:(object_idx + 1) * 6] = block
    return A_obj


@dataclass(frozen=True)
class GeoField:
    """Per-channel geodesic tables + channel WORLD frames — the bundle ``build_contact`` reads as its
    ``geo`` argument. Assembled by Plan C from ``InteractionContext`` (the geodesic tables) +
    ``FrameTargets.object_rot/pos``. Lets ``build_contact`` (a) read the geodesic field per channel and
    (b) map object-LOCAL field directions/gradients to world via ``world_normal(rot[c], …)`` —
    uniformly across channels (ground frame = identity). See plan Assumption 3."""

    tables: tuple[GeodesicTable | None, ...]  # (C,) per-channel geodesic table; None -> no C-X row
    rot: np.ndarray                           # (C, 3, 3) per-channel world rotation (ground = I)
    pos: np.ndarray                           # (C, 3)    per-channel world translation (ground = 0)
    object_idx: tuple[int, ...]               # (C,) channel -> object index (-1 for ground)
