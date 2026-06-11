"""Load human-object motion from an OMOMO/InterAct .pt file.

.pt layout (InterAct interact2mimic.py — raw tensor of shape (T, 591)):
  [162:162+52*3]    52 joint positions (mujoco order), xyz
  [318:325]         object pose as [tx, ty, tz, qx, qy, qz, qw]  (trans first, xyzw quat)
  [383:383+52*4]    per-joint global quats (52, 4) stored as xyzw, with
                    intermimic's upright_start convention baked in

obj_poses reorder: raw columns [6, 3, 4, 5, 0, 1, 2] → [qw, qx, qy, qz, x, y, z]
  i.e. the stored layout is [t(0:3), q_xyzw(3:7)] and we output [qw, qx, qy, qz, tx, ty, tz].

Joint quaternions: intermimic's upright_start fix is undone by right-multiplying each
  stored xyzw quaternion by Q = [0.5, 0.5, 0.5, 0.5] (xyzw), then converted to wxyz.
  See HoloNew.src.gmr_socp.targets._undo_upright_start for the original derivation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_UPRIGHT_START_FIX_XYZW = np.array([0.5, 0.5, 0.5, 0.5])


def _undo_upright_start(quats_xyzw: np.ndarray) -> np.ndarray:
    """Right-multiply (T, J, 4) xyzw quats by the upright_start fix quaternion."""
    T, J, _ = quats_xyzw.shape
    fixed = Rotation.from_quat(quats_xyzw.reshape(-1, 4)) * Rotation.from_quat(_UPRIGHT_START_FIX_XYZW)
    return fixed.as_quat().reshape(T, J, 4)


def load_pt_motion(
    pt_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load joint positions, object poses, and joint quaternions from a .pt file.

    Parameters
    ----------
    pt_path:
        Path to the OMOMO .pt motion file.

    Returns
    -------
    joints : (T, 52, 3) float32
        Joint world positions in mujoco order.
    obj_poses : (T, 7) float32
        Object pose per frame as [qw, qx, qy, qz, x, y, z].
        Source layout is [tx, ty, tz, qx, qy, qz, qw]; column reorder applied:
        raw[:, [6, 3, 4, 5, 0, 1, 2]].
    quats : (T, 52, 4) float32
        Per-joint global quaternions in wxyz order, with intermimic's
        upright_start convention removed.
    """
    import torch

    data = torch.load(str(Path(pt_path)), map_location="cpu", weights_only=False).detach().numpy()

    # Joint positions: columns [162 : 162+52*3], shape (T, 52, 3)
    joints = data[:, 162 : 162 + 52 * 3].reshape(-1, 52, 3).astype(np.float32)

    # Object pose: columns [318:325] = [tx, ty, tz, qx, qy, qz, qw]
    # Reorder to [qw, qx, qy, qz, tx, ty, tz] via index [6, 3, 4, 5, 0, 1, 2]
    raw = data[:, 318:325]
    obj_poses = raw[:, [6, 3, 4, 5, 0, 1, 2]].astype(np.float32)

    # Per-joint quaternions: columns [383 : 383+52*4], stored as xyzw with upright_start baked in
    quats_xyzw = data[:, 383 : 383 + 52 * 4].reshape(-1, 52, 4)
    quats_xyzw = _undo_upright_start(quats_xyzw)
    # Convert xyzw -> wxyz: [x, y, z, w] -> [w, x, y, z]
    quats = quats_xyzw[:, :, [3, 0, 1, 2]].astype(np.float32)

    return joints, obj_poses, quats
