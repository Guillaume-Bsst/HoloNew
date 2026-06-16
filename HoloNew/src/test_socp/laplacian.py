"""W^lap: native-Holosoma interaction-mesh Laplacian deformation term for TEST-SOCP.

The interaction mesh is the mapped robot bodies (and, later, object points). The mesh
topology (Delaunay adjacency) and the target Laplacian coordinates come from the SOURCE
reference shape (the GMR-grounded targets). With uniform Laplacian weights L depends
only on the adjacency, so L is constant per frame and the residual

    sqrt(lambda) * (L @ v(q) - target_lap)

is exactly affine in dqa via J_L = kron(L, I3) @ J_V, J_V the stacked point Jacobians.
"""
from __future__ import annotations

import numpy as np

from HoloNew.src.holosoma.interaction_mesh import (
    calculate_laplacian_coordinates,
    calculate_laplacian_matrix,
    create_interaction_mesh,
    get_adjacency_list,
)
from HoloNew.src.test_socp.centroidal import mapped_frame_masses_and_names


def _mesh_frames(rt):
    """Robot link names forming the interaction-mesh vertices (mapped-body order)."""
    frames, _ = mapped_frame_masses_and_names(rt)
    return frames


def laplacian_pieces(rt, q_pin, frame_idx):
    """Per-frame constants + Jacobian for the Laplacian residual.

    Returns (L, target_lap, robot_vertices, J_V) where L is (K, K), target_lap (K, 3),
    robot_vertices (K, 3) at q_pin, and J_V (3K, nv_a) the stacked point Jacobians.
    """
    frames = _mesh_frames(rt)
    K = len(frames)
    # Source reference shape (GMR-grounded targets) -> adjacency + target coordinates.
    source = np.asarray(rt.gmr_ground["pos"][frame_idx][:K], dtype=np.float64)
    _, tets = create_interaction_mesh(source)
    adj = get_adjacency_list(tets, K)
    L = calculate_laplacian_matrix(source, adj)          # (K, K); uniform -> adj-only
    target_lap = calculate_laplacian_coordinates(source, adj)  # (K, 3)

    robot_vertices = np.array([rt.pin.body_position(q_pin, f) for f in frames])  # (K, 3)
    J_V = np.zeros((3 * K, rt.nv_a))
    for i, f in enumerate(frames):
        J_V[3 * i:3 * i + 3, :] = rt.pin.frame_translational_jacobian(q_pin, f)[:, rt.v_a_indices]
    return L, target_lap, robot_vertices, J_V


def build_laplacian_terms(rt, q_pin, dqa, frame_idx, lambda_lap):
    """Assemble the W^lap residual as a list with one cp.sum_squares term."""
    import cvxpy as cp

    if lambda_lap <= 0:
        return []
    L, target_lap, robot_vertices, J_V = laplacian_pieces(rt, q_pin, frame_idx)
    K = L.shape[0]
    Kron = np.kron(L, np.eye(3))                         # (3K, 3K)
    J_L = Kron @ J_V                                     # (3K, nv_a)
    lap0 = (L @ robot_vertices).reshape(-1)              # (3K,)
    b = lap0 - target_lap.reshape(-1)                    # (3K,)
    sw = float(np.sqrt(lambda_lap))
    return [cp.sum_squares(sw * (J_L @ dqa + b))]
