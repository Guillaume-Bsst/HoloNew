"""HOI-M3 (HODome) loader: raw SMPL-X params -> Z-up joints; object R/T -> poses."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation as R

from HoloNew.src.data_loaders.base import MotionLoader, register_loader
from HoloNew.src.utils import transform_y_up_to_z_up


def hoim3_fk(npz_path: Path, model_dir: Path) -> tuple[np.ndarray, float]:
    """FK raw SMPL-X params to (T, 22, 3) Z-up joints; return (joints, rest_height_m)."""
    d = np.load(str(npz_path), allow_pickle=True)
    T = d["body_pose"].shape[0]
    model = smplx.SMPLX(model_path=str(model_dir), gender=str(d["gender"]), ext="npz",
                        num_betas=d["betas"].shape[-1], use_pca=False)
    betas = torch.from_numpy(np.asarray(d["betas"][:1], np.float32)).repeat(T, 1)
    out = model(
        betas=betas,
        global_orient=torch.from_numpy(np.asarray(d["global_orient"], np.float32)),
        body_pose=torch.from_numpy(np.asarray(d["body_pose"], np.float32)),
        transl=torch.from_numpy(np.asarray(d["transl"], np.float32)),
    )
    joints = out.joints.detach().numpy()[:, :22, :]          # SMPL-X body order
    joints = transform_y_up_to_z_up(joints)                  # dataset is Y-up

    rest = model(betas=betas[:1])
    rv = rest.vertices.detach().numpy()[0]
    height = float(rv[:, 1].max() - rv[:, 1].min())          # SMPL native Y-up stature
    return joints, height


def hoim3_object_poses(npz_path: Path) -> np.ndarray:
    """Object 6DoF (T,7) [qw,qx,qy,qz,x,y,z] from object_R (T,3,3) + object_T (T,1,3)."""
    d = np.load(str(npz_path), allow_pickle=True)
    rot = np.asarray(d["object_R"], np.float64)
    trans = np.asarray(d["object_T"], np.float64).reshape(-1, 3)
    quat_xyzw = R.from_matrix(rot).as_quat()                 # (T,4) xyzw
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    return np.concatenate([quat_wxyz, trans], axis=1)


@register_loader("hoim3")
class HoiM3Loader(MotionLoader):
    def load(self, *, model_path, motion_path, obj_path, task_type,
             constants, motion_data_config):
        human_joints, height = hoim3_fk(Path(motion_path), Path(model_path))
        smpl_scale = float(constants.ROBOT_HEIGHT) / height

        n = human_joints.shape[0]
        if task_type == "robot_only" or obj_path is None:
            object_poses = np.tile(np.array([[1, 0, 0, 0, 0, 0, 0]]), (n, 1))
        else:
            object_poses = hoim3_object_poses(Path(obj_path))[:n]
        return human_joints, object_poses, smpl_scale
