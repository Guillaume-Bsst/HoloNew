"""Pose un ``PointCloud`` pour une frame à partir de ses transformations du monde des parties — l'op unique
partagée par tout type de nuage (humain K~4, objet K=1, robot K=1), donc pas de chemin de code par type
(homogénéité).

``p[i] = sum_k weights[i,k] * (R[parts[i,k]] @ offsets[i,k] + t[parts[i,k]])``

L'appelant fournit les transformations par partie qui correspondent au nuage : les parties du nuage humain
sont des os SMPL (``BodyModel.bone_transforms``), la partie unique d'un nuage objet est la pose objet
(``R[None]``, ``t[None]``), les parties d'un nuage robot sont des liens (FK). Pur, vectorisé, pas de boucle
Python sur les points ; sans torch et sans mesh.
"""
from __future__ import annotations

import numpy as np

from ...prepare.contracts import PointCloud


def pose_cloud(cloud: PointCloud, part_rot: np.ndarray, part_pos: np.ndarray) -> np.ndarray:
    """(P, 3) points mondiaux. ``part_rot (J,3,3)`` et ``part_pos (J,3)`` sont la transformation du monde
    de chaque partie que le nuage peut référencer (os / objet / liens) ; ``cloud.parts`` rassemble par point."""
    rot = part_rot[cloud.parts]                                     # (P, K, 3, 3)
    pos = part_pos[cloud.parts]                                     # (P, K, 3)
    contrib = np.einsum("pkij,pkj->pki", rot, cloud.offsets) + pos  # (P, K, 3) placement de chaque os
    return np.einsum("pk,pki->pi", cloud.weights, contrib)         # (P, 3) blend de skinning
