"""Unit tests for the segment maps (pure, no deps): the anatomical link/joint -> segment rule and
the per-sample labelling that keeps the per-segment OT hand->hand, foot->foot."""
import numpy as np

from holov2.prepare.point_cloud.correspondence.segments import (
    SEGMENTS, SMPLX_JOINT_TO_SEGMENT, link_to_segment, point_segments, seg_index)


def test_link_to_segment_real_g1_links():
    cases = {
        "pelvis": "pelvis", "pelvis_contour_link": "pelvis",
        "waist_yaw_link": "torso", "torso_link": "torso", "head_link": "head",
        "left_hip_pitch_link": "left_thigh", "left_knee_link": "left_shank",
        "left_ankle_roll_link": "left_foot",
        "right_shoulder_pitch_link": "right_upperarm", "right_elbow_link": "right_forearm",
        "left_wrist_yaw_link": "left_forearm", "right_rubber_hand_link": "right_hand",
        "left_thumb_link": "left_hand",
    }
    for link, seg in cases.items():
        assert link_to_segment(link) == seg, f"{link} -> {link_to_segment(link)} (want {seg})"


def test_smplx_joint_segments_cover_all_55():
    assert set(SMPLX_JOINT_TO_SEGMENT) == set(range(55))            # every SMPL-X joint labelled
    assert all(s in SEGMENTS for s in SMPLX_JOINT_TO_SEGMENT.values())
    assert SMPLX_JOINT_TO_SEGMENT[0] == "pelvis"
    assert SMPLX_JOINT_TO_SEGMENT[20] == "left_forearm"            # wrist -> forearm
    assert SMPLX_JOINT_TO_SEGMENT[25] == "left_hand" and SMPLX_JOINT_TO_SEGMENT[54] == "right_hand"


def test_point_segments_takes_dominant_corner():
    # 2 triangles (verts 0,1,2 and 3,4,5); lbs one-hot so vertex i -> joint i.
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    lbs = np.zeros((6, 55))
    lbs[0, 0] = lbs[1, 0] = lbs[2, 0] = 1.0                         # tri 0 verts -> pelvis (joint 0)
    lbs[3, 20] = lbs[4, 20] = lbs[5, 25] = 1.0                      # tri 1: v3,v4 wrist, v5 hand
    tri_idx = np.array([0, 1])
    bary = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])            # dom corner: 0 then 2
    seg = point_segments(lbs, faces, tri_idx, bary)
    assert seg[0] == seg_index("pelvis")                           # dom corner of tri 0 = pelvis
    assert seg[1] == seg_index("left_hand")                        # dom corner of tri 1 = v5 (hand)
