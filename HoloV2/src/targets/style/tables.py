"""Robot-keyed GMR style tables + the fixed SMPL-bone-index map (torch-free, module-level data).

The STYLE objective ports the GMR (General Motion Retargeting, YanjieZe/GMR) per-link target recipe:
each tracked robot link is pulled toward ONE SMPL body, after a morphological SCALE and a per-link
frame OFFSET (see ``style/build.py``). That recipe is robot-SPECIFIC data — a name-keyed table,
exactly like the correspondence rest pose (``prepare/load/robot.CORRESPONDENCE_REST_POSE``) — NOT a
config knob: adding a robot is a data entry, never a code change.

Ported from HoloNew ``test_socp/tables.py`` (``IK_MATCH_TABLE1`` + ``HUMAN_SCALE_TABLE``). The GMR
human bodies are addressed by NAME and resolved to OUR SMPL-X bone order (``SMPL_BODY_INDEX``, derived
from ``prepare/load/smpl.SMPLX_BODY_JOINTS``) — NOT the 52-joint MuJoCo layout V1 indexed into. Kept
numpy-free here (plain Python data, importable anywhere); ``build.py`` does the float64 math.
"""
from __future__ import annotations

# A fixed SMPL-skeleton fact (like the frame conventions): GMR human-body NAME -> bone INDEX in OUR
# SMPL-X bone order (``prepare/load/smpl.SMPLX_BODY_JOINTS``, i.e. the order of ``FramePose.bone_rot``
# / ``bone_pos``). GMR "left_foot" / "right_foot" are the leg-chain ANKLE joints (the toe is reached
# via the table's pos_offset), so they resolve to L_Ankle (7) / R_Ankle (8), NOT the SMPL foot/toe
# bones (10 / 11). Hardcoded so ``targets`` never imports the torch-pulling ``load/smpl``.
SMPL_BODY_INDEX: dict[str, int] = {
    "pelvis": 0,
    "left_hip": 1, "right_hip": 2,
    "left_knee": 4, "right_knee": 5,
    "left_foot": 7, "right_foot": 8,        # L_Ankle / R_Ankle
    "spine3": 9,
    "left_shoulder": 16, "right_shoulder": 17,
    "left_elbow": 18, "right_elbow": 19,
    "left_wrist": 20, "right_wrist": 21,
}

ROOT_BODY = "pelvis"            # the SCALE anchor (V1 ``HUMAN_ROOT_NAME``)
HUMAN_HEIGHT_ASSUMPTION = 1.8   # GMR's reference subject height (m): ratio = stature / this. A fact.
GROUND_HEIGHT = 0.0             # GMR target grounding plane z (V1 ``GROUND_HEIGHT``) — a fact, not a knob.

# Per-SMPL-body morphological scale (V1 ``HUMAN_SCALE_TABLE``, GMR smplx_to_g1): a body's pelvis-local
# vector is scaled by ``SCALE[body] * ratio``. Torso/legs 0.9, arms 0.8.
HUMAN_SCALE_TABLE: dict[str, float] = {
    "pelvis": 0.9, "spine3": 0.9,
    "left_hip": 0.9, "right_hip": 0.9,
    "left_knee": 0.9, "right_knee": 0.9,
    "left_foot": 0.9, "right_foot": 0.9,
    "left_shoulder": 0.8, "right_shoulder": 0.8,
    "left_elbow": 0.8, "right_elbow": 0.8,
    "left_wrist": 0.8, "right_wrist": 0.8,
}

# G1 GMR style table (ported from V1 ``IK_MATCH_TABLE1``): robot_link -> (smpl_body, w_p, w_r,
# pos_offset xyz, rot_offset wxyz). ``w_p`` / ``w_r`` are the position / orientation tracking weights
# (V1 FrameTask position_cost / orientation_cost): 100 = planted (pelvis + feet), 0 = position
# unconstrained (limb tracked by orientation only). ``pos_offset`` is added in the RE-ORIENTED body
# frame (corrects the human reference point vs the robot link point); ``rot_offset`` (w FIRST) is
# composed onto the body orientation (SMPL-X vs G1 zero-pose alignment per body). Insertion order is
# the link order of ``StyleTargets`` (14 tracked G1 links).
_H = 0.4267755048530407
_K = 0.5637931078484661
_STYLE_TABLE: dict[str, dict[str, tuple]] = {
    "g1": {
        "pelvis":                  ("pelvis",        100, 10, (0.0,  0.0,  0.0), (0.5, -0.5, -0.5, -0.5)),
        "left_hip_roll_link":      ("left_hip",        0, 10, (0.0,  0.0,  0.0), (_H, -_K, -_K, -_H)),
        "left_knee_link":          ("left_knee",       0, 10, (0.0,  0.0,  0.0), (0.5, -0.5, -0.5, -0.5)),
        "left_toe_link":           ("left_foot",     100, 10, (0.0,  0.02, 0.0), (0.5, -0.5, -0.5, -0.5)),
        "right_hip_roll_link":     ("right_hip",       0, 10, (0.0,  0.0,  0.0), (_H, -_K, -_K, -_H)),
        "right_knee_link":         ("right_knee",      0, 10, (0.0,  0.0,  0.0), (0.5, -0.5, -0.5, -0.5)),
        "right_toe_link":          ("right_foot",    100, 10, (0.0, -0.02, 0.0), (0.5, -0.5, -0.5, -0.5)),
        "torso_link":              ("spine3",          0, 10, (0.0,  0.0,  0.0), (0.5, -0.5, -0.5, -0.5)),
        "left_shoulder_yaw_link":  ("left_shoulder",   0, 10, (0.0,  0.0,  0.0), (0.70710678, 0.0, -0.70710678, 0.0)),
        "left_elbow_link":         ("left_elbow",      0, 10, (0.0,  0.0,  0.0), (1.0, 0.0, 0.0, 0.0)),
        "left_wrist_yaw_link":     ("left_wrist",      0, 10, (0.0,  0.0,  0.0), (1.0, 0.0, 0.0, 0.0)),
        "right_shoulder_yaw_link": ("right_shoulder",  0, 10, (0.0,  0.0,  0.0), (0.0, 0.70710678, 0.0, 0.70710678)),
        "right_elbow_link":        ("right_elbow",     0, 10, (0.0,  0.0,  0.0), (0.0, 0.0, 0.0, -1.0)),
        "right_wrist_yaw_link":    ("right_wrist",     0, 10, (0.0,  0.0,  0.0), (0.0, 0.0, 0.0, -1.0)),
    },
}


def style_table(robot_name: str) -> dict[str, tuple]:
    """GMR style table for ``robot_name`` (robot_link -> (smpl_body, w_p, w_r, pos_offset, rot_offset);
    raises if undefined). Mirrors ``load/robot.correspondence_rest_angles``."""
    try:
        return _STYLE_TABLE[robot_name]
    except KeyError:
        raise ValueError(f"no style table for robot {robot_name!r} — add an entry to "
                         f"_STYLE_TABLE") from None
