"""Crée la surface de chaque objet dans un ``PointCloud`` rigide (K=1).

Un objet est un corps rigide unique, donc son nuage est le cas dégénéré à influence unique du même
contrat que le nuage humain : chaque point a une partie (le corps, indice 0), poids 1 et un décalage
qui EST le point dans le repère local de l'objet. La pose en ligne est alors le même ``pose_cloud``
avec la transformation mondiale par image de l'objet passée comme partie unique — aucun chemin de
code spécifique à l'objet.

La géométrie provient de ``prepare/load/mesh.load_mesh`` (le point d'entrée trimesh partagé), donc
le nuage et le SDF de l'objet lisent le même repère local exact. L'échantillonnage est déterministe
dans la seed de la config.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..contracts import PointCloud
from ..config import CloudConfig
from .cache import load_cloud, save_cloud


# =============================================================================
# Fonctions pures (déterministes ; pas d'I/O, pas de mutation des inputs)
# =============================================================================
def sample_object_surface(vertices: np.ndarray, faces: np.ndarray, density: float,
                          seed: int) -> np.ndarray:
    """Échantillonne uniformément ``~area * density`` points (min 64) sur la surface → (P, 3) objet-local.
    Déterministe dans ``seed`` (échantillonnage uniforme par disque de Poisson), donc la création est reproductible."""
    import trimesh
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices, np.float64), faces=np.asarray(faces),
                           process=False)
    n = max(64, int(float(mesh.area) * density))
    pts, _ = trimesh.sample.sample_surface_even(mesh, n, seed=seed)
    return np.asarray(pts, np.float64)


def assemble_rigid_cloud(points_local: np.ndarray) -> PointCloud:
    """Enveloppe les points de surface locaux à l'objet dans un ``PointCloud`` K=1 : une partie
    (indice 0), poids 1, le point lui-même comme décalage au repos-local. Les objets ne portent pas
    de correspondance, donc ``sampling_id`` est vide (il lie uniquement le nuage humain à son
    échantillonnage)."""
    pts = np.asarray(points_local, np.float64)
    p = pts.shape[0]
    return PointCloud(parts=np.zeros((p, 1), np.int64), weights=np.ones((p, 1), np.float32),
                      offsets=pts[:, None, :].astype(np.float32), sampling_id="")


def build_object_cloud(vertices: np.ndarray, faces: np.ndarray, config: CloudConfig) -> PointCloud:
    """Échantillonne la surface de l'objet et l'enveloppe dans un ``PointCloud`` rigide."""
    pts = sample_object_surface(vertices, faces, config.object_density, config.seed)
    return assemble_rigid_cloud(pts)


# =============================================================================
# ObjectCloudBuilder — l'AssetBuilder pour ce livrable (build / cache)
# =============================================================================
class ObjectCloudBuilder:
    """``AssetBuilder`` produisant un ``PointCloud`` rigide d'objet. Limité par GÉOMÉTRIE (partagé
    par chaque scène utilisant le même objet) : la clé de cache hash le maillage local + densité + seed."""

    def cache_key(self, config: CloudConfig, vertices: np.ndarray, faces: np.ndarray) -> str:
        h = hashlib.sha1()
        h.update(f"{config.object_density}|{config.seed}".encode())
        h.update(np.ascontiguousarray(vertices, np.float32).tobytes())
        h.update(np.ascontiguousarray(faces, np.int64).tobytes())
        return h.hexdigest()

    def build(self, config: CloudConfig, vertices: np.ndarray, faces: np.ndarray) -> PointCloud:
        return build_object_cloud(vertices, faces, config)

    def save(self, cloud: PointCloud, path: Path) -> None:
        save_cloud(cloud, path)

    def load(self, path: Path) -> PointCloud:
        return load_cloud(path)
