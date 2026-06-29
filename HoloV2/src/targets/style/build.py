"""style.build — per-frame SMPL bones -> ``StyleTargets`` (GMR posture/style tracking, G1-ready).

The object-agnostic "how the body should move" channel: each tracked robot link is pulled toward ONE
SMPL body after a morphological SCALE (the subject's proportions, in pelvis-local frame) and a per-link
frame OFFSET (SMPL-X vs robot zero-pose alignment + a small reference-point shift). Ported from HoloNew
``test_socp/preprocess`` (``scale`` then ``offset``); the tunable scalars are ``targets/config``'s
``StyleConfig`` and the robot-specific recipe is the robot-keyed ``style_table`` (also ``targets/config``).

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

from ..config import ARM_BODIES, ROOT_BODY, SMPL_BODY_INDEX, SceneScaleConfig, StyleConfig, style_table
from ..scale import resolve_scale
from ..contracts import FramePose, StyleTargets
from ...prepare.contracts import RobotSpec


def _quat_wxyz_to_R(quat_wxyz) -> np.ndarray:
    """(4,) wxyz quaternion -> (3, 3) rotation matrix (scipy is xyzw; kept torch-free)."""
    q = np.asarray(quat_wxyz, np.float64)
    return R.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def _R_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    """(3, 3) rotation matrix -> (4,) wxyz quaternion."""
    q = R.from_matrix(np.asarray(rot, np.float64)).as_quat()           # xyzw
    return q[[3, 0, 1, 2]]


def build(pose: FramePose, robot: RobotSpec, stature: float,
          cfg: StyleConfig = StyleConfig(), scene: SceneScaleConfig = SceneScaleConfig()) -> StyleTargets:
    """One frame of SMPL bones -> ``StyleTargets`` for ``robot``'s tracked links.

    ``pose`` carries the per-frame SMPL bone (R, t) (J_bones, SMPL-X order); ``robot.name`` selects the
    GMR style RECIPE (``config.style_table``); ``stature`` (the subject's rest height,
    ``GroundedScene.body.stature``) sets the morphological ``ratio = stature /
    cfg.human_height_assumption``. ``cfg`` (``StyleConfig``) carries the tunable SCALE values + heights.
    Per tracked link: SCALE the source body in pelvis-local frame, then OFFSET (compose rot_offset, add
    pos_offset in the re-oriented frame).
    """
    table = style_table(robot.name)
    ratio = stature / cfg.human_height_assumption

    bone_rot = np.asarray(pose.bone_rot, np.float64)                   # (J_bones, 3, 3)
    bone_pos = np.asarray(pose.bone_pos, np.float64)                   # (J_bones, 3)

    # SCALE anchor: the root (pelvis) world position rigidly places the whole scaled skeleton. V1-TEST
    # defaults: scale_xy = 1.0 (x, y kept native) and scale_z = None (z scaled morphologically by the
    # root's ``SCALE[pelvis] * ratio``). Body proportions (pelvis-local) are unchanged otherwise.
    root_pos = bone_pos[SMPL_BODY_INDEX[ROOT_BODY]]                    # (3,)
    # PLACEMENT du root via l'échelle de scène (None -> ratio) ; xy autour de l'origine, z autour du
    # sol. Le morphologique du pelvis (scale_torso_legs, le pelvis est torse/jambes) reste sur z.
    # scale_xy=1.0, scale_z=None reproduit le natif : xy brut, z = scale_torso_legs * ratio.
    s_xy, s_z = resolve_scale(scene, ratio)
    base_z = cfg.scale_torso_legs * s_z
    scaled_root = np.array([root_pos[0] * s_xy, root_pos[1] * s_xy,
                            cfg.ground_height + (root_pos[2] - cfg.ground_height) * base_z])
    ground = np.array([0.0, 0.0, cfg.ground_height])

    links = tuple(table.keys())
    L = len(links)
    position = np.empty((L, 3), np.float64)
    orientation = np.empty((L, 4), np.float64)

    for i, link in enumerate(links):
        body, pos_off, rot_off = table[link]
        idx = SMPL_BODY_INDEX[body]
        src_rot = bone_rot[idx]                                        # (3, 3) world orientation
        src_pos = bone_pos[idx]                                        # (3,)

        # SCALE: morph the body's pelvis-local vector by ``SCALE[body] * ratio``, re-anchored on the
        # scaled root (the orientation is untouched by scale).
        if body == ROOT_BODY:
            scaled_pos = scaled_root
        else:
            s = (cfg.scale_arms if body in ARM_BODIES else cfg.scale_torso_legs) * ratio
            scaled_pos = (src_pos - root_pos) * s + scaled_root

        # OFFSET: compose rot_offset onto the orientation, then add pos_offset expressed in the
        # re-oriented body frame (minus cfg.ground_height*z, a no-op at ground 0). Mirrors V1 ``offset``.
        updated_rot = src_rot @ _quat_wxyz_to_R(rot_off)              # (3, 3)
        world_pos = scaled_pos + updated_rot @ (np.asarray(pos_off, np.float64) - ground)

        position[i] = world_pos
        orientation[i] = _R_to_quat_wxyz(updated_rot)

    return StyleTargets(link_names=links, position=position, orientation=orientation)
