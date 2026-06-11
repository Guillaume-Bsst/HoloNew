"""Build per-frame SE3 body targets for the GMR-SOCP objective.

For each robot frame in an IK match table, produce the world-frame target
position and rotation (with the table's pos_offset / rot_offset applied) plus the
position and orientation weights. Quaternions are wxyz throughout.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .tables import HUMAN_BODY_TO_IDX


def _wxyz_to_R(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def build_frame_targets(joint_pos: np.ndarray, joint_quat_wxyz: np.ndarray, table: dict):
    """joint_pos: (J,3) world positions; joint_quat_wxyz: (J,4) wxyz per joint.

    Returns dict: robot_frame -> (p_target(3,), R_target(3,3), pos_weight, rot_weight).
    pos_offset is applied in the re-oriented (rot_offset-composed) body frame;
    rot_offset is composed onto the human orientation (per GMR's tables).
    """
    out = {}
    for frame, (human, pos_w, rot_w, pos_off, rot_off) in table.items():
        idx = HUMAN_BODY_TO_IDX[human]
        R_h = _wxyz_to_R(joint_quat_wxyz[idx])
        R_off = _wxyz_to_R(np.asarray(rot_off, dtype=float))
        R_target = R_h @ R_off
        p_target = joint_pos[idx] + R_target @ np.asarray(pos_off, dtype=float)
        out[frame] = (np.asarray(p_target, float), R_target, float(pos_w), float(rot_w))
    return out


# Intermimic's "upright_start" correction quaternion (xyzw). The .pt files from
# interact2mimic.py bake a right-rotation by Q^-1 into each stored joint quaternion
# (interact2mimic.py:795). We undo it by right-multiplying by Q to recover the true
# SMPL-X global orientations expected by the GMR tables.
_UPRIGHT_START_FIX_XYZW = np.array([0.5, 0.5, 0.5, 0.5])


def _undo_upright_start(quats_xyzw: np.ndarray) -> np.ndarray:
    """Right-multiply (T, J, 4) xyzw quats by the upright_start fix quaternion."""
    T, J, _ = quats_xyzw.shape
    fixed = Rotation.from_quat(quats_xyzw.reshape(-1, 4)) * Rotation.from_quat(_UPRIGHT_START_FIX_XYZW)
    return fixed.as_quat().reshape(T, J, 4)


def load_pt_quaternions(pt_path: str | Path) -> np.ndarray:
    """Load per-joint quaternions (T, J, 4) wxyz from an OMOMO .pt file.

    .pt layout (InterAct interact2mimic.py — raw tensor of shape (T, 591)):
      [383 : 383 + 52*4]  per-joint global quats (52, 4) stored in xyzw order.

    The stored quats have intermimic's upright_start convention baked in
    (each true SMPL-X global orientation is post-rotated by Q^-1 where
    Q = [0.5, 0.5, 0.5, 0.5] xyzw). We undo it by right-multiplying with Q,
    then convert from xyzw -> wxyz.

    Returns: (T, 52, 4) float32 numpy array, wxyz quaternion convention.
    """
    import torch
    data = torch.load(pt_path, map_location="cpu", weights_only=False).detach().numpy()
    # Extract per-joint quaternions: stored as xyzw in the flat slice
    quats_xyzw = data[:, 383 : 383 + 52 * 4].reshape(-1, 52, 4)
    # Undo intermimic's upright_start bake-in to recover true SMPL-X global orientations
    quats_xyzw = _undo_upright_start(quats_xyzw)
    # Convert xyzw -> wxyz by reordering the last axis: [x,y,z,w] -> [w,x,y,z]
    quats_wxyz = quats_xyzw[:, :, [3, 0, 1, 2]].astype(np.float32)
    return quats_wxyz
