"""style.build — per-frame SMPL bones -> ``StyleTargets`` (GMR posture/style tracking, G1-ready).

The object-agnostic "how the body should move" channel: each tracked robot link is pulled toward ONE
SMPL body after a morphological SCALE (the subject's proportions, in pelvis-local frame) and a per-link
frame OFFSET (SMPL-X vs robot zero-pose alignment + a small reference-point shift). Ported from HoloNew
``test_socp/preprocess`` (``scale`` then ``offset``); the robot-specific recipe is the robot-keyed
``style_table`` in ``tables.py``.

Reads each mapped body's bone (R, t) from the shared ``FramePose`` (J_bones, SMPL-X order) — NOT the
demo joints: the bone WORLD transforms already carry the joint orientation the style term tracks, and
sharing ``FramePose`` avoids a recompute. Pure, float64, torch-free (scipy only); no I/O, no input
mutation, frozen numpy output.

V1's clip-wide FLOOR drop (subtract the lowest mapped-body z over the WHOLE sequence) is intentionally
NOT applied here: style stays strictly per-frame. If the feet float, a precomputed clip drop is to be
re-added upstream later — not as a per-frame pass.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..contracts import FramePose, StyleTargets
from ...prepare.contracts import RobotSpec
from .tables import (GROUND_HEIGHT, HUMAN_HEIGHT_ASSUMPTION, HUMAN_SCALE_TABLE, ROOT_BODY,
                     SMPL_BODY_INDEX, style_table)


def _quat_wxyz_to_R(quat_wxyz) -> np.ndarray:
    """(4,) wxyz quaternion -> (3, 3) rotation matrix (scipy is xyzw; kept torch-free)."""
    q = np.asarray(quat_wxyz, np.float64)
    return R.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def _R_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    """(3, 3) rotation matrix -> (4,) wxyz quaternion."""
    q = R.from_matrix(np.asarray(rot, np.float64)).as_quat()           # xyzw
    return q[[3, 0, 1, 2]]


def build(pose: FramePose, robot: RobotSpec, stature: float) -> StyleTargets:
    """One frame of SMPL bones -> ``StyleTargets`` for ``robot``'s tracked links.

    ``pose`` carries the per-frame SMPL bone (R, t) (J_bones, SMPL-X order); ``robot.name`` selects the
    GMR style table; ``stature`` (the subject's rest height, ``GroundedScene.body.stature``) sets the
    morphological ``ratio = stature / HUMAN_HEIGHT_ASSUMPTION``. Per tracked link: SCALE the source body
    in pelvis-local frame, then OFFSET (compose rot_offset, add pos_offset in the re-oriented frame).
    """
    table = style_table(robot.name)
    ratio = stature / HUMAN_HEIGHT_ASSUMPTION

    bone_rot = np.asarray(pose.bone_rot, np.float64)                   # (J_bones, 3, 3)
    bone_pos = np.asarray(pose.bone_pos, np.float64)                   # (J_bones, 3)

    # SCALE anchor: the root (pelvis) world position rigidly places the whole scaled skeleton. V1-TEST
    # defaults: scale_xy = 1.0 (x, y kept native) and scale_z = None (z scaled morphologically by the
    # root's ``SCALE[pelvis] * ratio``). Body proportions (pelvis-local) are unchanged otherwise.
    root_pos = bone_pos[SMPL_BODY_INDEX[ROOT_BODY]]                    # (3,)
    base = HUMAN_SCALE_TABLE[ROOT_BODY] * ratio
    scaled_root = np.array([root_pos[0], root_pos[1], root_pos[2] * base])   # sx = sy = 1.0, sz = base
    ground = np.array([0.0, 0.0, GROUND_HEIGHT])

    links = tuple(table.keys())
    L = len(links)
    position = np.empty((L, 3), np.float64)
    orientation = np.empty((L, 4), np.float64)
    weight_pos = np.empty(L, np.float64)
    weight_rot = np.empty(L, np.float64)

    for i, link in enumerate(links):
        body, w_p, w_r, pos_off, rot_off = table[link]
        idx = SMPL_BODY_INDEX[body]
        src_rot = bone_rot[idx]                                        # (3, 3) world orientation
        src_pos = bone_pos[idx]                                        # (3,)

        # SCALE: morph the body's pelvis-local vector by ``SCALE[body] * ratio``, re-anchored on the
        # scaled root (the orientation is untouched by scale).
        if body == ROOT_BODY:
            scaled_pos = scaled_root
        else:
            s = HUMAN_SCALE_TABLE[body] * ratio
            scaled_pos = (src_pos - root_pos) * s + scaled_root

        # OFFSET: compose rot_offset onto the orientation, then add pos_offset expressed in the
        # re-oriented body frame (minus GROUND_HEIGHT*z, a no-op at ground 0). Mirrors V1 ``offset``.
        updated_rot = src_rot @ _quat_wxyz_to_R(rot_off)              # (3, 3)
        world_pos = scaled_pos + updated_rot @ (np.asarray(pos_off, np.float64) - ground)

        position[i] = world_pos
        orientation[i] = _R_to_quat_wxyz(updated_rot)
        weight_pos[i] = w_p
        weight_rot[i] = w_r

    for a in (position, orientation, weight_pos, weight_rot):
        a.flags.writeable = False
    return StyleTargets(link_names=links, position=position, weight_pos=weight_pos,
                        weight_rot=weight_rot, orientation=orientation)
