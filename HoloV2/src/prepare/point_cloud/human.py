"""Crée la surface SMPL dans un ``PointCloud`` portant son propre skinning LBS creux.

Échantillonné UNE FOIS sur le maillage au repos du sujet, réutilisant l'``SurfaceSampling`` partagé
(pour que l'ordre des points corresponde à ``smpl_idx`` de la correspondance). Chaque point stocke
ses K os SMPL dominants, leurs poids de skinning normalisés et son décalage dans le repère local au
repos de chaque os ; la pose est alors sans maillage et sans torch (``targets/interaction/pose_cloud``),
comblant les plis articulaires qu'un seul os rigide déchirerait (LBS-on-cloud). Le modèle SMPL lourd
n'est touché que ici, au moment de la création.

Pourquoi le décalage est une simple différence native : au repos, la transformation mondiale de chaque
os SMPL est la pure rotation de repère Q (natif Y-up → mondial Z-up ; ``load/smpl.py``), identique pour
tous les os, donc dans ``p = sum_k w_k (R[k] @ offset_k + t[k])`` elle s'annule et
``offset_k = rest_point - rest_joint[k]`` dans le repère natif (le repère que ``rest_vertices`` /
``rest_joints`` / ``lbs_weights`` partagent tous).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..contracts import PointCloud, SmplParams
from ..config import CloudConfig
from ..load.smpl import SmplBody
from .sampling import SurfaceSampling
from .cache import load_cloud, save_cloud


# =============================================================================
# Fonctions pures (pas d'I/O, pas de mutation)
# =============================================================================
def _top_k(weights_dense: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Conserve les K poids de skinning les plus grands de chaque ligne, renormalisés pour sommer 1.
    Retourne ``parts (N, K)`` indices d'os et ``weights (N, K)`` (l'ordre dans une ligne est sans importance —
    ``pose_cloud`` somme sur K). ``K`` est limité au nombre d'os."""
    n, j = weights_dense.shape
    k = min(k, j)
    idx = np.argpartition(-weights_dense, k - 1, axis=1)[:, :k]      # (N, K) les K bones dominants
    w = np.take_along_axis(weights_dense, idx, axis=1)               # (N, K)
    w = w / w.sum(axis=1, keepdims=True)
    return idx.astype(np.int64), w.astype(np.float32)


def bake_skinned_cloud(rest_verts: np.ndarray, faces: np.ndarray, lbs_weights: np.ndarray,
                       rest_joints: np.ndarray, sampling: SurfaceSampling, k: int) -> PointCloud:
    """``PointCloud`` avec skinning creux à partir d'un maillage au repos + ses poids LBS,
    échantillonné à ``sampling``.

    ``rest_verts (V,3)`` / ``rest_joints (J,3)`` / ``lbs_weights (V,J)`` sont tous dans le repère
    natif au repos du corps. Chaque échantillon est le mélange barycentrique des sommets de son
    triangle (point et vecteur de poids) ; les K os dominants donnent le skinning creux."""
    rv = np.asarray(rest_verts, np.float64)
    bc = np.asarray(sampling.bary, np.float64)                      # (N, 3)
    tri_v = np.asarray(faces)[np.asarray(sampling.tri_idx)]         # (N, 3) ids de vertex
    pts = np.einsum("nij,ni->nj", rv[tri_v], bc)                    # (N, 3) rest surface points
    w_dense = np.einsum("nij,ni->nj", np.asarray(lbs_weights, np.float64)[tri_v], bc)  # (N, J)
    parts, weights = _top_k(w_dense, k)                             # (N, K)
    offsets = pts[:, None, :] - np.asarray(rest_joints, np.float64)[parts]  # (N, K, 3) bone-local
    return PointCloud(parts=parts, weights=weights, offsets=offsets.astype(np.float32),
                      sampling_id=sampling.sampling_id)


def build_human_cloud(body: SmplBody, sampling: SurfaceSampling, config: CloudConfig) -> PointCloud:
    """Crée le nuage humain du sujet (récupère les tableaux au repos de ``body``)."""
    return bake_skinned_cloud(body.rest_vertices(None), body.faces, body.lbs_weights,
                              body.rest_joints, sampling, config.k_influences)


# =============================================================================
# HumanCloudBuilder — l'AssetBuilder pour ce livrable (build / cache)
# =============================================================================
class HumanCloudBuilder:
    """``AssetBuilder`` produisant le ``PointCloud`` humain du sujet. Limité par SUJET : le nuage
    est la géométrie au repos du sujet échantillonnée à l'``SurfaceSampling`` partagé (de sorte que
    son ``sampling_id`` se lie à la correspondance). Le runner encapsule ``build``/``load`` dans un
    ``prof.span``."""

    def cache_key(self, config: CloudConfig, params: SmplParams, sampling: SurfaceSampling) -> str:
        """Hash de tout ce dont le nuage dépend : le K du skinning creux, l'identité d'échantillonnage
        partagée (densité/seed/topologie) et le sujet (betas + genre). Aucun forward SMPL nécessaire —
        la géométrie au repos est entièrement déterminée par ceux-ci."""
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
