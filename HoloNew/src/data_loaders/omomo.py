"""OMOMO loader: motion from the InterMimic .pt (new), betas from the OMOMO pickle (non-new)."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import smplx
import torch

from HoloNew.src.data_loaders.base import MotionLoader, register_loader
from HoloNew.src.utils import load_intermimic_data

import tempfile

# Disk cache for the recentred+scaled captured object meshes (one per sequence stem).
_OMOMO_MESH_CACHE = Path(tempfile.gettempdir()) / "holonew_omomo_meshes"


def _omomo_obj_name(seq_name: str) -> str:
    """Object token = 2nd '_'-segment of the sequence name (sub3_largebox_003 -> largebox)."""
    parts = str(seq_name).split("_")
    return parts[1] if len(parts) >= 2 else str(seq_name)


def resolve_omomo_object_mesh(seq_name, omomo_dir=None, cache_dir=None):
    """Object mesh path for an OMOMO sequence, in the centroid-centred frame the .pt
    poses expect.

    1. Bundled <pkg>/models/<obj>/<obj>.obj (already recentred + pre-scaled): returned
       as-is. Package-anchored, so independent of the current working directory.
    2. Else the captured unit mesh data/captured_objects/<obj>_cleaned_simplified.obj,
       recentred on its vertex mean and scaled by the per-sequence obj_scale, written to
       a derived .obj in cache_dir and returned (mirrors the HODome mesh cache).
    3. Else None.

    Raises ValueError if a captured mesh exists but obj_scale is unavailable (a wrong
    size is worse than a clear failure).
    """
    import trimesh
    obj = _omomo_obj_name(seq_name)
    pkg_models = Path(__file__).resolve().parents[2] / "models"
    bundled = pkg_models / obj / f"{obj}.obj"
    if bundled.exists():
        return bundled
    if omomo_dir is None:
        return None
    captured = (Path(omomo_dir) / "data" / "captured_objects"
                / f"{obj}_cleaned_simplified.obj")
    if not captured.exists():
        return None
    from HoloNew.src.test_socp.correspondence.human_metadata import load_object_scale
    scale = load_object_scale(Path(omomo_dir), str(seq_name))
    if scale is None:
        raise ValueError(
            f"obj_scale missing for {seq_name}: captured mesh {captured.name} is "
            f"off-origin and unscaled, so it cannot be sized correctly.")
    cache_dir = Path(cache_dir) if cache_dir is not None else _OMOMO_MESH_CACHE
    out = cache_dir / f"{seq_name}.obj"
    if out.exists():
        return out
    mesh = trimesh.load(str(captured), force="mesh", process=False)
    verts = np.asarray(mesh.vertices, np.float64)
    verts = (verts - verts.mean(0)) * float(scale)
    cache_dir.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=verts, faces=np.asarray(mesh.faces), process=False).export(str(out))
    return out


def omomo_height_from_betas(betas: np.ndarray, gender: str, model_dir: Path) -> float:
    """Stature (m) from SMPL-H betas via rest-pose forward kinematics.

    Runs the body model with zero pose/orient/transl and measures the vertical
    (SMPL native Y-up) extent of the posed mesh vertices.
    """
    betas = np.asarray(betas, np.float32).reshape(1, -1)
    model = smplx.SMPLH(model_path=str(model_dir), gender=gender, ext="pkl", use_pca=False)
    # The shared SMPL+H model exposes a fixed number of shape coefficients (10);
    # OMOMO stores 16. Truncate/pad to the model's capacity — the leading betas
    # carry the dominant stature variation, which is all the scale factor needs.
    nb = model.num_betas
    fitted = np.zeros((1, nb), np.float32)
    k = min(nb, betas.shape[-1])
    fitted[0, :k] = betas[0, :k]
    out = model(betas=torch.from_numpy(fitted))
    verts = out.vertices.detach().numpy()[0]
    return float(verts[:, 1].max() - verts[:, 1].min())


def _betas_for_seq(pickle_path: Path, seq_name: str) -> tuple[np.ndarray, str]:
    data = joblib.load(str(pickle_path))
    for entry in data.values():
        if str(entry["seq_name"]) == seq_name:
            return np.asarray(entry["betas"], np.float32), str(entry.get("gender", "neutral"))
    raise KeyError(f"seq_name {seq_name!r} not found in {pickle_path}")


@register_loader("omomo")
class OmomoMixedLoader(MotionLoader):
    def load(self, *, model_path, motion_path, obj_path, task_type,
             constants, motion_data_config, smpl_model_dir=None):
        if smpl_model_dir is None:
            raise ValueError(
                "OMOMO height needs the SMPL-H body model: pass --smpl-model-dir "
                "(e.g. <models>/smplh). No default is assumed.")

        human_joints, object_poses = load_intermimic_data(str(motion_path))
        if task_type == "robot_only":
            n = human_joints.shape[0]
            object_poses = np.tile(np.array([[1, 0, 0, 0, 0, 0, 0]]), (n, 1))

        betas, gender = _betas_for_seq(Path(model_path), Path(motion_path).stem)
        height = omomo_height_from_betas(betas, gender, Path(smpl_model_dir))
        smpl_scale = float(constants.ROBOT_HEIGHT) / height
        return human_joints, object_poses, smpl_scale

    def object_source(self, *, motion_path, obj_path, model_path, task_type,
                      constants, motion_data_config, smpl_model_dir=None):
        if task_type == "robot_only" or motion_path is None:
            return []
        from HoloNew.src.data_loaders.base import ObjectSource
        from HoloNew.src.utils import load_intermimic_data
        seq = Path(motion_path).stem
        # OMOMO release root (holds data/captured_objects + the betas/obj_scale pickle)
        # is two levels up from the pickle passed as model_path. None when unavailable
        # (then only the bundled mesh can resolve).
        omomo_dir = Path(model_path).parent.parent if model_path is not None else None
        try:
            mesh_path = resolve_omomo_object_mesh(seq, omomo_dir)
        except ValueError:
            # Captured mesh present but no obj_scale: degrade to no-object (floor-only),
            # consistent with the builder's "object mesh not found" behaviour.
            return []
        if mesh_path is None:
            return []
        _, poses = load_intermimic_data(str(motion_path))
        return [ObjectSource(mesh_path=Path(mesh_path), poses_raw=np.asarray(poses, np.float64))]
