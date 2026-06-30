"""Charge un mesh d'objet/terrain d'un chemin dans une géométrie local-frame (appuyé par trimesh).

Point d'entrée unique pour les consommateurs de géométrie hors ligne — ``prepare/sdf`` (construit l'SDF)
et ``prepare/point_cloud/objects`` (échantillonne la surface). Les deux doivent lire le MÊME frame local,
donc le mesh est chargé UNE FOIS ici. Les chargeurs de dataset (``prepare/load/datasets/*``) ont déjà
centré + caché chaque mesh sur le centroïde dont ses poses sont calibrées, donc c'est une simple lecture
de géométrie.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Mesh au ``path`` -> (vertices (V, 3) float64, faces (F, 3) int64), frame local, géométrie uniquement.

    ``skip_materials`` supprime les textures (non pertinent pour l'SDF/nuage) ; ``process=True`` soude
    les sommets dupliqués pour que ``mesh.contains`` ait une surface étanche à signer. Le chargeur de
    dataset a déjà fixé le centrage/frame, que le soudage préserve (il déduplique les positions, ne les
    déplace jamais)."""
    import trimesh
    m = trimesh.load(str(path), force="mesh", process=True, skip_materials=True)
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)
