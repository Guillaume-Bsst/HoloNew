# src/test_pipe_retargeting/test_pipe_retargeting/fields/object_input.py
from __future__ import annotations

from pathlib import Path

import numpy as np


def parse_obj_name(seq: str) -> str:
    """'sub10_clothesstand_000' → 'clothesstand'."""
    parts = seq.split("_")
    return parts[1] if len(parts) >= 2 else seq


def get_obj_scale(omomo_dir: Path, obj_name: str) -> float:
    """Extract obj_scale[0] for obj_name from OMOMO raw .p files."""
    import joblib
    p_files = [
        omomo_dir / "data" / "train_diffusion_manip_seq_joints24.p",
        omomo_dir / "data" / "test_diffusion_manip_seq_joints24.p",
    ]
    for p_file in p_files:
        if not p_file.exists():
            continue
        data = joblib.load(str(p_file))
        for idx in data:
            seq_name = data[idx].get("seq_name", "")
            if seq_name.split("_")[1] == obj_name:
                return float(np.asarray(data[idx]["obj_scale"]).flat[0])
    return 1.0


def load_mesh(omomo_dir: Path, obj_name: str) -> tuple | None:
    """Return (mesh, obj_scale, mesh_origin) or None.

    obj_trans in the .pt files is calibrated for the vertex mean (centroid) of the
    scaled canonical mesh. Centering the mesh at this origin aligns it correctly.
    """
    try:
        import trimesh
    except ImportError:
        return None
    obj_path = omomo_dir / "data" / "captured_objects" / f"{obj_name}_cleaned_simplified.obj"
    if not obj_path.exists():
        return None
    scale = get_obj_scale(omomo_dir, obj_name)
    print(f"  obj_scale={scale:.4f} (from OMOMO .p)")
    mesh = trimesh.load(str(obj_path), force="mesh", process=False)
    mesh.vertices *= scale
    mesh_origin = np.mean(mesh.vertices, axis=0)
    mesh.vertices -= mesh_origin
    return mesh, scale, mesh_origin
