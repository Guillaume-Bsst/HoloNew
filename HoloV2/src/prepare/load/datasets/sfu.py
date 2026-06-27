"""SFU loader (AMASS-style SMPL-X): per-joint GLOBAL orientations + positions -> RawMotion.

The .npz stores 22 global SMPL-X body-joint orientations + positions (already Z-up, floor~0),
plus betas and gender -- NO local pose, NO hands, NO objects. We reconstruct the local
``SmplParams`` the ``BodyModel`` needs (``local_rotvecs_from_global``; hands set to zero) and keep
the global positions as the demo joints. Body-only: interaction would be human-vs-ground only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ...contracts import RawMotion, SceneSpec, SmplParams
from ..base import register_loader
from ..smpl import SMPLX_BODY_JOINTS, local_rotvecs_from_global, rest_body_model


@register_loader("sfu")
class SfuLoader:
    """SceneSpec -> RawMotion for an SFU SMPL-X sequence (body-only, no objects)."""

    def load(self, spec: SceneSpec) -> RawMotion:
        if spec.smpl_model_dir is None:
            raise ValueError("SFU needs spec.smpl_model_dir (the SMPL-X model directory)")
        d = np.load(str(spec.motion_path), allow_pickle=True)
        betas = np.asarray(d["betas"], np.float32).reshape(-1)
        gender = str(d["gender"])
        # SFU quats are wxyz (validated against positions).
        quats = np.asarray(d["global_joint_orientations"], np.float64)   # (T, 22, 4) global, Z-up
        pos = np.asarray(d["global_joint_positions"], np.float32)        # (T, 22, 3) Z-up
        T, J = pos.shape[0], pos.shape[1]

        rest = rest_body_model(betas, gender, Path(spec.smpl_model_dir))
        local, transl = local_rotvecs_from_global(
            quats, pos[:, 0], rest.parents[:J], rest.rest_joints[0])
        z = np.zeros((T, 45), np.float32)
        params = SmplParams(betas=betas, global_orient=local[:, 0],
                            body_pose=local[:, 1:J].reshape(T, -1), left_hand_pose=z,
                            right_hand_pose=z, transl=transl, gender=gender, model_type="smplx")
        return RawMotion(joint_pos=pos, joint_names=SMPLX_BODY_JOINTS, fps=30.0, source_format="sfu",
                         object_poses_raw=(), object_mesh_paths=(), smpl_params=params)
