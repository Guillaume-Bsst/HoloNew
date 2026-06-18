"""OMOMO loader: motion from the InterMimic .pt (new), betas from the OMOMO pickle (non-new)."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import smplx
import torch

from HoloNew.src.data_loaders.base import (
    MotionLoader, SMPLH_MODEL_DIR_DEFAULT, register_loader,
)
from HoloNew.src.utils import load_intermimic_data


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
             constants, motion_data_config):
        human_joints, object_poses = load_intermimic_data(str(motion_path))
        if task_type == "robot_only":
            n = human_joints.shape[0]
            object_poses = np.tile(np.array([[1, 0, 0, 0, 0, 0, 0]]), (n, 1))

        betas, gender = _betas_for_seq(Path(model_path), Path(motion_path).stem)
        height = omomo_height_from_betas(betas, gender, SMPLH_MODEL_DIR_DEFAULT)
        smpl_scale = float(constants.ROBOT_HEIGHT) / height
        return human_joints, object_poses, smpl_scale
