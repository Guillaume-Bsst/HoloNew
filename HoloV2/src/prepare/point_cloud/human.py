"""Bakes the SMPL surface into a ``PointCloud`` carrying its own sparse LBS skinning.

Sampled ONCE on the subject's rest mesh, reusing the shared ``SurfaceSampling`` (so the point order
matches the correspondence's ``smpl_idx``). Each point stores its K dominant SMPL bones, their
normalised skinning weights, and its offset in each bone's REST-local frame; posing is then
mesh-free and torch-free (``targets/interaction/pose_cloud``), closing joint creases that a single
rigid bone would tear (LBS-on-cloud). The heavy SMPL model is touched only here, at bake time.

Why the offset is a plain native difference: at rest every SMPL bone's world transform is the pure
frame rotation Q (native Y-up -> world Z-up; ``load/smpl.py``), identical for all bones, so in
``p = sum_k w_k (R[k] @ offset_k + t[k])`` it cancels and ``offset_k = rest_point - rest_joint[k]``
in the native frame (the frame ``rest_vertices`` / ``rest_joints`` / ``lbs_weights`` all share).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ...contracts import PointCloud, SmplParams
from config_types import CloudConfig
from ..load.smpl import SmplBody
from .sampling import SurfaceSampling
from .store import load_cloud, save_cloud


# =============================================================================
# Pure functions (no I/O, no mutation)
# =============================================================================
def _top_k(weights_dense: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep each row's K largest skinning weights, renormalised to sum 1.
    Returns ``parts (N, K)`` bone indices and ``weights (N, K)`` (order within a row is irrelevant —
    ``pose_cloud`` sums over K). ``K`` is clamped to the number of bones."""
    n, j = weights_dense.shape
    k = min(k, j)
    idx = np.argpartition(-weights_dense, k - 1, axis=1)[:, :k]      # (N, K) the K dominant bones
    w = np.take_along_axis(weights_dense, idx, axis=1)               # (N, K)
    w = w / w.sum(axis=1, keepdims=True)
    return idx.astype(np.int64), w.astype(np.float32)


def bake_skinned_cloud(rest_verts: np.ndarray, faces: np.ndarray, lbs_weights: np.ndarray,
                       rest_joints: np.ndarray, sampling: SurfaceSampling, k: int) -> PointCloud:
    """Sparse-skinned ``PointCloud`` from a rest mesh + its LBS weights, sampled at ``sampling``.

    ``rest_verts (V,3)`` / ``rest_joints (J,3)`` / ``lbs_weights (V,J)`` are all in the body's
    native rest frame. Each sample is the barycentric blend of its triangle's vertices (point and
    weight vector alike); the K dominant bones give the sparse skinning."""
    rv = np.asarray(rest_verts, np.float64)
    bc = np.asarray(sampling.bary, np.float64)                      # (N, 3)
    tri_v = np.asarray(faces)[np.asarray(sampling.tri_idx)]         # (N, 3) vertex ids
    pts = np.einsum("nij,ni->nj", rv[tri_v], bc)                    # (N, 3) rest surface points
    w_dense = np.einsum("nij,ni->nj", np.asarray(lbs_weights, np.float64)[tri_v], bc)  # (N, J)
    parts, weights = _top_k(w_dense, k)                             # (N, K)
    offsets = pts[:, None, :] - np.asarray(rest_joints, np.float64)[parts]  # (N, K, 3) bone-local
    return PointCloud(parts=parts, weights=weights, offsets=offsets.astype(np.float32),
                      sampling_id=sampling.sampling_id)


def build_human_cloud(body: SmplBody, sampling: SurfaceSampling, config: CloudConfig) -> PointCloud:
    """Bake the subject's human cloud (pulls the rest arrays from ``body``)."""
    return bake_skinned_cloud(body.rest_vertices(None), body.faces, body.lbs_weights,
                              body.rest_joints, sampling, config.k_influences)


# =============================================================================
# HumanCloudBuilder — the AssetBuilder for this deliverable (build / cache)
# =============================================================================
class HumanCloudBuilder:
    """``AssetBuilder`` producing the subject's human ``PointCloud``. Scoped per SUBJECT: the cloud
    is the subject's rest geometry sampled at the shared ``SurfaceSampling`` (so its ``sampling_id``
    binds to the correspondence). The runner wraps ``build``/``load`` in a ``prof.span``."""

    def cache_key(self, config: CloudConfig, params: SmplParams, sampling: SurfaceSampling) -> str:
        """Hash of everything the cloud depends on: the K of the sparse skinning, the shared
        sampling identity (density/seed/topology), and the subject (betas + gender). No SMPL forward
        needed — the rest geometry is fully determined by these."""
        h = hashlib.sha1()
        h.update(f"{config.k_influences}|{sampling.sampling_id}|{params.gender}".encode())
        h.update(np.ascontiguousarray(params.betas, np.float32).tobytes())
        return h.hexdigest()

    def build(self, config: CloudConfig, body: SmplBody, sampling: SurfaceSampling) -> PointCloud:
        return build_human_cloud(body, sampling, config)

    def save(self, cloud: PointCloud, path: Path) -> None:
        save_cloud(cloud, path)

    def load(self, path: Path) -> PointCloud:
        return load_cloud(path)
