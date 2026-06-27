"""Loads an object/terrain mesh from a path into local-frame geometry (trimesh-backed).

Single trimesh entry point for the offline geometry consumers — ``prepare/sdf`` (builds the SDF) and
``prepare/point_cloud/objects`` (samples the surface). Both must read the SAME local frame, so the
mesh is loaded ONCE here. The dataset loaders (``prepare/load/datasets/*``) already centred + cached
each mesh on the centroid its poses are calibrated against, so this is a plain geometry read.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Mesh at ``path`` -> (vertices (V, 3) float64, faces (F, 3) int64), local frame, geometry only.

    ``skip_materials`` drops textures (irrelevant to the SDF/cloud); ``process=True`` welds duplicate
    vertices so ``mesh.contains`` has a watertight surface to sign against. The dataset loader already
    fixed the centring/frame, which welding preserves (it dedupes positions, never moves them)."""
    import trimesh
    m = trimesh.load(str(path), force="mesh", process=True, skip_materials=True)
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)
