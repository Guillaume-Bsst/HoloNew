# src/gmr_socp/tables.py
"""Mapping & scale tables copied verbatim from GMR (YanjieZe/GMR),
general_motion_retargeting/ik_configs/smplx_to_g1.json.

Credit: General Motion Retargeting (GMR). Only the *tables* are reused here.
The operations applied to them (scale / offset / ground) are reimplemented
independently in preprocess.py.

What lives here, and what it is for:

  IK_MATCH_TABLE1 / IK_MATCH_TABLE2
      The *definition of the IK optimisation problem*. Each row attaches one
      mink.FrameTask to a robot body and tells the solver which human target it
      tracks, how strongly (the cost weights), and how the target is recalibrated
      (the offsets). The two tables are the two cost-weight sets of GMR's
      two-pass solve (see problem.py / backends.py). See the column legend below.

  HUMAN_SCALE_TABLE, HUMAN_HEIGHT_ASSUMPTION, *_ROOT_NAME, GROUND_HEIGHT
      Pre-IK target *preprocessing* inputs (morphological scaling + grounding),
      consumed by preprocess.py — not part of the cost function itself.

  HUMAN_BODY_TO_IDX, MAPPED_BODY_NAMES, MAPPED_BODY_BONES
      Indexing into the 52-joint SMPL-X array and bone topology for *rendering*.
"""
from __future__ import annotations

HUMAN_ROOT_NAME = "pelvis"
HUMAN_HEIGHT_ASSUMPTION = 1.8
GROUND_HEIGHT = 0.0

HUMAN_SCALE_TABLE: dict[str, float] = {
    "pelvis": 0.9, "spine3": 0.9,
    "left_hip": 0.9, "right_hip": 0.9,
    "left_knee": 0.9, "right_knee": 0.9,
    "left_foot": 0.9, "right_foot": 0.9,
    "left_shoulder": 0.8, "right_shoulder": 0.8,
    "left_elbow": 0.8, "right_elbow": 0.8,
    "left_wrist": 0.8, "right_wrist": 0.8,
}

# ── Column legend (a row defines one mink.FrameTask) ─────────────────────────
# robot_frame -> (human_body, pos_weight, rot_weight, pos_offset[xyz], rot_offset[wxyz])
#   robot_frame : KEY. The G1 body the FrameTask is attached to (frame_type="body").
#   human_body  : the Ground-stage human body this frame is pulled toward (the target;
#                 set_targets reads human_data[human_body]).
#   pos_weight  : -> FrameTask.position_cost. Weight of POSITION error in the cost.
#                 0 = position unconstrained (only orientation matters for this frame);
#                 100 = hard (pelvis + feet — the "planted" points). A row with both
#                 weights 0 is dropped (no task created — see problem.py).
#   rot_weight  : -> FrameTask.orientation_cost. Weight of ORIENTATION (quaternion) error.
#   pos_offset  : xyz shift added to the target, expressed in the body's (re-oriented)
#                 frame. Corrects the human-reference point vs the robot link point.
#   rot_offset  : wxyz quaternion (w FIRST) composed onto the human orientation. Recalibrates
#                 the SMPL-X vs G1 frame conventions (a "zero pose" alignment per body).
#   pos_offset/rot_offset are applied to the targets in preprocess.offset(); pos_weight/
#   rot_weight feed the cost function of the solve.
#
# Uses ik_match_table1 — this is what GMR's pre-IK scaled_human_data is built from.
# table2 differs in IK weights *and* in the foot entries (a different toe rot_offset
# and a zeroed toe pos_offset); none of that affects these stages, since table2 is
# consumed only during the (deferred) IK solve.
_H = 0.4267755048530407
_K = 0.5637931078484661
IK_MATCH_TABLE1: dict[str, tuple] = {
    "pelvis":                 ("pelvis",        100, 10, [0.0, 0.0,  0.0], [0.5, -0.5, -0.5, -0.5]),
    "left_hip_roll_link":     ("left_hip",        0, 10, [0.0, 0.0,  0.0], [_H, -_K, -_K, -_H]),
    "left_knee_link":         ("left_knee",       0, 10, [0.0, 0.0,  0.0], [0.5, -0.5, -0.5, -0.5]),
    "left_toe_link":          ("left_foot",     100, 10, [0.0, 0.02, 0.0], [0.5, -0.5, -0.5, -0.5]),
    "right_hip_roll_link":    ("right_hip",       0, 10, [0.0, 0.0,  0.0], [_H, -_K, -_K, -_H]),
    "right_knee_link":        ("right_knee",      0, 10, [0.0, 0.0,  0.0], [0.5, -0.5, -0.5, -0.5]),
    "right_toe_link":         ("right_foot",    100, 10, [0.0, -0.02,0.0], [0.5, -0.5, -0.5, -0.5]),
    "torso_link":             ("spine3",          0, 10, [0.0, 0.0,  0.0], [0.5, -0.5, -0.5, -0.5]),
    "left_shoulder_yaw_link": ("left_shoulder",   0, 10, [0.0, 0.0,  0.0], [0.70710678, 0.0, -0.70710678, 0.0]),
    "left_elbow_link":        ("left_elbow",      0, 10, [0.0, 0.0,  0.0], [1.0, 0.0, 0.0, 0.0]),
    "left_wrist_yaw_link":    ("left_wrist",      0, 10, [0.0, 0.0,  0.0], [1.0, 0.0, 0.0, 0.0]),
    "right_shoulder_yaw_link":("right_shoulder",  0, 10, [0.0, 0.0,  0.0], [0.0, 0.70710678, 0.0, 0.70710678]),
    "right_elbow_link":       ("right_elbow",     0, 10, [0.0, 0.0,  0.0], [0.0, 0.0, 0.0, -1.0]),
    "right_wrist_yaw_link":   ("right_wrist",     0, 10, [0.0, 0.0,  0.0], [0.0, 0.0, 0.0, -1.0]),
}

ROBOT_ROOT_NAME = "pelvis"
USE_IK_MATCH_TABLE1 = True
USE_IK_MATCH_TABLE2 = True

# Same column layout as IK_MATCH_TABLE1 (see legend above).
# ik_match_table2 from smplx_to_g1.json. NOTE: GMR only ever applies table1's
# offsets to the human targets; table2's offset columns are loaded but never
# applied. The two passes differ ONLY in these IK weights (e.g. positions are now
# enabled on the limbs at 10, and foot orientation is tightened to 50). Kept
# verbatim for fidelity; set_targets must use table1-offset targets for both groups.
IK_MATCH_TABLE2: dict[str, tuple] = {
    "pelvis":                 ("pelvis",        100, 5, [0.0, 0.0, 0.0], [0.5, -0.5, -0.5, -0.5]),
    "left_hip_roll_link":     ("left_hip",       10, 5, [0.0, 0.0, 0.0], [_H, -_K, -_K, -_H]),
    "left_knee_link":         ("left_knee",      10, 5, [0.0, 0.0, 0.0], [0.5, -0.5, -0.5, -0.5]),
    "left_toe_link":          ("left_foot",     100, 50, [0.0, 0.0, 0.0], [-0.5, 0.5, 0.5, 0.5]),
    "right_hip_roll_link":    ("right_hip",      10, 5, [0.0, 0.0, 0.0], [_H, -_K, -_K, -_H]),
    "right_knee_link":        ("right_knee",     10, 5, [0.0, 0.0, 0.0], [0.5, -0.5, -0.5, -0.5]),
    "right_toe_link":         ("right_foot",    100, 50, [0.0, 0.0, 0.0], [-0.5, 0.5, 0.5, 0.5]),
    "torso_link":             ("spine3",          0, 10, [0.0, 0.0, 0.0], [0.5, -0.5, -0.5, -0.5]),
    "left_shoulder_yaw_link": ("left_shoulder",  10, 5, [0.0, 0.0, 0.0], [0.70710678, 0.0, -0.70710678, 0.0]),
    "left_elbow_link":        ("left_elbow",     10, 5, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
    "left_wrist_yaw_link":    ("left_wrist",     10, 5, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
    "right_shoulder_yaw_link":("right_shoulder", 10, 5, [0.0, 0.0, 0.0], [0.0, 0.70710678, 0.0, 0.70710678]),
    "right_elbow_link":       ("right_elbow",    10, 5, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, -1.0]),
    "right_wrist_yaw_link":   ("right_wrist",    10, 5, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, -1.0]),
}

# human body name -> index into the 52-joint mujoco-order array (constants.JOINT_NAMES).
# Validated by the GMR parity tests (tests/test_gmr_parity.py, test_gmr_qpos_parity.py).
HUMAN_BODY_TO_IDX: dict[str, int] = {
    "pelvis": 0,
    "left_hip": 1, "left_knee": 2, "left_foot": 3,
    "right_hip": 5, "right_knee": 6, "right_foot": 7,
    "spine3": 11,
    "left_shoulder": 15, "left_elbow": 16, "left_wrist": 17,
    "right_shoulder": 34, "right_elbow": 35, "right_wrist": 36,
}

MAPPED_BODY_NAMES: list[str] = list(HUMAN_BODY_TO_IDX.keys())

# bones connecting mapped bodies, for stage-skeleton rendering (human-body-name pairs)
MAPPED_BODY_BONES: list[tuple[str, str]] = [
    ("pelvis", "spine3"),
    ("pelvis", "left_hip"), ("left_hip", "left_knee"), ("left_knee", "left_foot"),
    ("pelvis", "right_hip"), ("right_hip", "right_knee"), ("right_knee", "right_foot"),
    ("spine3", "left_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("spine3", "right_shoulder"), ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
]
