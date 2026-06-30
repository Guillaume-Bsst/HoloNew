"""L'identité d'échantillonnage de surface partagée qui lie le nuage humain à la correspondance.

``CorrespondenceTable.smpl_idx`` indexe l'ORDRE des points du nuage humain. Cette correspondance est
construite une fois sur un corps de modèle NEUTRE, tandis que le nuage à l'exécution est celui du
SUJET ; pour que ``smpl_idx`` reste valide, les deux doivent partager exactement le même
échantillonnage ``(tri_idx, bary)``. L'échantillonnage est donc l'identité canonique : la
correspondance le porte (intégré dans ``corr_neutral.npz``), et le nuage humain le RÉUTILISE au
lieu de le rééchantillonner — il recompile seulement le skinning par sujet par-dessus.

``SurfaceSampling`` est un intermédiaire de construction uniquement (il n'atteint jamais ``targets``/
``solve``), il réside donc ici, local à ``point_cloud/``, pas dans le paquet ``contracts/``.
``sampling_id`` est le hash stable estampillé sur le nuage (``PointCloud.sampling_id``) et la table
(``CorrespondenceTable.smpl_sampling_id``) ; le runner affirme qu'ils correspondent.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SurfaceSampling:
    """Un ensemble fixe d'échantillons de surface comme emplacements barycentriques sur les triangles
    d'un maillage, stable en ORDRE. Lié à la topologie (``tri_idx`` référence les faces SMPL), donc
    indépendant des betas : le même ``(tri_idx, bary)`` désigne un point valide sur n'importe quel
    maillage SMPL-X de la même topologie."""

    tri_idx: np.ndarray   # (N,)    triangle index per sample
    bary: np.ndarray      # (N, 3)  poids barycentriques, lignes sommant à 1
    sampling_id: str      # stable identity (see ``sampling_id``)

    @property
    def n_points(self) -> int:
        return self.tri_idx.shape[0]


def sampling_id(tri_idx: np.ndarray, bary: np.ndarray) -> str:
    """Hash stable d'un échantillonnage ``(tri_idx, bary)`` — la clé de liaison entre le nuage et la table.

    Dépend uniquement des emplacements échantillonnés (triangle + poids barycentriques), donc il est
    identique pour le modèle neutre et tous les sujets qui réutilisent le même échantillonnage, et
    change chaque fois que l'échantillonnage le fait. Hashé dans les dtype canoniques pour que la
    valeur soit reproductible sur les machines."""
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(tri_idx, np.int64).tobytes())
    h.update(np.ascontiguousarray(bary, np.float32).tobytes())
    return h.hexdigest()[:16]
