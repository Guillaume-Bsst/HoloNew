"""Bakes each object surface into a rigid ``PointCloud`` (K=1).

An object is one rigid body, so its cloud is the degenerate single-influence case of the same
contract as the human cloud: every point has one part (the body, index 0), weight 1, and an offset
that IS the point in the object-local frame. Posing online is then the very same ``pose_cloud`` with
the object's per-frame world transform passed as the single part — no object-specific code path.

Geometry comes from ``prepare/load/mesh.load_mesh`` (the shared trimesh entry point), so the cloud
and the object's SDF read the exact same local frame. Sampling is deterministic in the config seed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ...contracts import PointCloud
from config_types import CloudConfig
from .cache import load_cloud, save_cloud


# =============================================================================
# Pure functions (deterministic; no I/O, no mutation of inputs)
# =============================================================================
def sample_object_surface(vertices: np.ndarray, faces: np.ndarray, density: float,
                          seed: int) -> np.ndarray:
    """Evenly sample ``~area * density`` points (min 64) on the surface -> (P, 3) object-local.
    Deterministic in ``seed`` (even Poisson-disk sampling), so the bake is reproducible."""
    import trimesh
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices, np.float64), faces=np.asarray(faces),
                           process=False)
    n = max(64, int(float(mesh.area) * density))
    pts, _ = trimesh.sample.sample_surface_even(mesh, n, seed=seed)
    return np.asarray(pts, np.float64)


def assemble_rigid_cloud(points_local: np.ndarray) -> PointCloud:
    """Wrap object-local surface points into a K=1 ``PointCloud``: one part (index 0), weight 1, the
    point itself as the rest-local offset. Objects carry no correspondence, so ``sampling_id`` is
    empty (it only binds the human cloud to its sampling)."""
    pts = np.asarray(points_local, np.float64)
    p = pts.shape[0]
    return PointCloud(parts=np.zeros((p, 1), np.int64), weights=np.ones((p, 1), np.float32),
                      offsets=pts[:, None, :].astype(np.float32), sampling_id="")


def build_object_cloud(vertices: np.ndarray, faces: np.ndarray, config: CloudConfig) -> PointCloud:
    """Sample the object surface and wrap it into a rigid ``PointCloud``."""
    pts = sample_object_surface(vertices, faces, config.object_density, config.seed)
    return assemble_rigid_cloud(pts)


# =============================================================================
# ObjectCloudBuilder — the AssetBuilder for this deliverable (build / cache)
# =============================================================================
class ObjectCloudBuilder:
    """``AssetBuilder`` producing an object's rigid ``PointCloud``. Scoped per GEOMETRY (shared by
    every scene that uses the same object): the cache key hashes the local mesh + density + seed."""

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
