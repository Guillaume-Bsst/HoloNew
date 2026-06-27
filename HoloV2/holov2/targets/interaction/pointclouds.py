"""Poses a ``PointCloud`` for one frame from its parts' world transforms — the single op shared by
every cloud kind (human K~4, object K=1, robot K=1), so there is no per-kind code path (homogeneity).

``p[i] = sum_k weights[i,k] * (R[parts[i,k]] @ offsets[i,k] + t[parts[i,k]])``

The caller supplies the per-part transforms that match the cloud: the human cloud's parts are SMPL
bones (``BodyModel.bone_transforms``), an object cloud's single part is the object pose
(``R[None]``, ``t[None]``), a robot cloud's parts are links (FK). Pure, vectorised, no Python loop
over points; torch-free and mesh-free.
"""
from __future__ import annotations

import numpy as np

from ...contracts import PointCloud


def pose_cloud(cloud: PointCloud, part_rot: np.ndarray, part_pos: np.ndarray) -> np.ndarray:
    """(P, 3) world points. ``part_rot (J,3,3)`` and ``part_pos (J,3)`` are the world transform of
    every part the cloud can reference (bones / object / links); ``cloud.parts`` gathers per point."""
    rot = part_rot[cloud.parts]                                     # (P, K, 3, 3)
    pos = part_pos[cloud.parts]                                     # (P, K, 3)
    contrib = np.einsum("pkij,pkj->pki", rot, cloud.offsets) + pos  # (P, K, 3) each bone's placement
    return np.einsum("pk,pki->pi", cloud.weights, contrib)         # (P, 3) skinning blend
