"""Assembles a GroundedScene by applying the Calibration to the loaded motion and object poses.

The grounding (``Calibration.floor_offset``) is a single world z-shift applied to the WHOLE scene
at once — the demo joints, the object poses AND the human params — so the captured human<->object
contact geometry is preserved (everyone drops by the same amount). The scene stays at HUMAN scale:
the human->robot scale is NOT applied here — it is a (human, robot) quantity composed downstream by
the correspondence/transport layer from ``Calibration.human_stature`` and the robot height.

``root_frame`` is identity for now (provisional), so only the z-shift is applied. When a non-trivial
framing is introduced, apply it to the world arrays here (and rebase the native params accordingly).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..contracts import Calibration, GroundedScene, RawMotion, SmplParams

# SMPL params are in the model's NATIVE Y-up frame; the body model maps native Y -> world Z (the Q
# rotation in load/smpl.py). So a world z-drop of ``floor_offset`` is a native y-drop of the root
# translation by the same amount — posing the grounded params then yields the grounded world.
_NATIVE_UP_AXIS = 1   # transl column carrying world height


def _drop_object_z(pose: np.ndarray, dz: float) -> np.ndarray:
    """Lower an object's per-frame pose (T,7) [x,y,z,qw,qx,qy,qz] by ``dz`` in world z."""
    out = np.asarray(pose, np.float32).copy()
    out[:, 2] -= dz
    return out


def _ground_params(params: SmplParams, dz: float) -> SmplParams:
    """Lower the human by ``dz`` in world z via the native root translation (see module note)."""
    transl = np.asarray(params.transl).copy()
    transl[:, _NATIVE_UP_AXIS] -= dz
    return replace(params, transl=transl)


def assemble(raw: RawMotion, calib: Calibration) -> GroundedScene:
    """Apply ``calib`` to a loaded ``RawMotion`` -> ``GroundedScene`` (grounded, human-scale)."""
    dz = float(calib.floor_offset)
    joints = np.asarray(raw.joint_pos, np.float32).copy()
    joints[:, :, 2] -= dz
    objects = tuple(_drop_object_z(p, dz) for p in raw.object_poses_raw)
    params = _ground_params(raw.smpl_params, dz) if raw.is_parametric else None
    return GroundedScene(
        joint_pos=joints, joint_names=raw.joint_names, object_poses=objects,
        object_mesh_paths=raw.object_mesh_paths, calibration=calib, fps=raw.fps,
        smpl_params=params,
    )
