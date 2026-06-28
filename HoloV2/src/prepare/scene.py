"""Assembles a GroundedScene by applying the Calibration to the loaded motion and object poses.

Grounding is PER ENTITY (single-human / multi-object): the human (demo joints + SMPL params) drops
by ``Calibration.human_offset``, and ALL objects drop together by the single ``Calibration.object_offset``
— the human and the objects can sit at different heights in the raw capture (e.g. the human floats
while the objects already rest on the floor), so one shared scene shift would push the objects through
the floor; the objects share one offset so their inter-object geometry is kept. The scene stays at
HUMAN scale: the human->robot scale is NOT applied here — it is
a (human, robot) quantity composed downstream by the correspondence/transport layer from
``body.stature`` and the robot height.

The subject's ``body`` (built once upstream by the runner) is carried THROUGH onto the
``GroundedScene`` so the per-frame treatment can pose the human cloud (``body.bone_transforms``); it
is ``None`` for a positions-only source. ``assemble`` only stores the reference — it stays pure
(numpy-only), the torch-backed body is built in ``load/smpl.py``.

``root_frame`` is identity for now (provisional), so only the z-shifts are applied. When a non-trivial
framing is introduced, apply it to the world arrays here (and rebase the native params accordingly).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .contracts import BodyModel, Calibration, GroundedScene, RawMotion, SmplParams

# SMPL params are in the model's NATIVE Y-up frame; the body model maps native Y -> world Z (the Q
# rotation in load/smpl.py). So a world z-drop of ``human_offset`` is a native y-drop of the root
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


def assemble(raw: RawMotion, calib: Calibration, body: BodyModel | None = None) -> GroundedScene:
    """Apply ``calib`` to a loaded ``RawMotion`` -> ``GroundedScene`` (grounded, human-scale).

    The human drops by ``human_offset``; all objects drop together by the shared ``object_offset``.
    ``body`` (the subject's posing engine, built once upstream) is carried through unchanged; pass
    ``None`` for a positions-only source."""
    dz = float(calib.human_offset)
    joints = np.asarray(raw.joint_pos, np.float32).copy()
    joints[:, :, 2] -= dz
    objects = tuple(_drop_object_z(p, calib.object_offset) for p in raw.object_poses_raw)
    params = _ground_params(raw.smpl_params, dz) if raw.is_parametric else None
    return GroundedScene(
        joint_pos=joints, joint_names=raw.joint_names, object_poses=objects,
        object_mesh_paths=raw.object_mesh_paths, calibration=calib, fps=raw.fps,
        smpl_params=params, body=body,
    )
