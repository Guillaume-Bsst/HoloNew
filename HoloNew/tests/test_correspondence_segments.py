import numpy as np
from HoloNew.src.test_socp.correspondence.segments import (
    SEGMENTS, g1_link_to_segment, point_segments, SMPLX_JOINT_TO_SEGMENT,
)

def test_g1_link_segments():
    assert g1_link_to_segment("pelvis") == "pelvis"
    assert g1_link_to_segment("left_knee_link") == "left_shank"
    assert g1_link_to_segment("right_ankle_roll_link") == "right_foot"
    assert g1_link_to_segment("left_shoulder_pitch_link") == "left_upperarm"
    assert g1_link_to_segment("torso_link") == "torso"

def test_segments_count():
    assert len(SEGMENTS) == 15

def test_point_segments_picks_dominant_corner():
    V = 16
    lbs = np.zeros((V, 55)); lbs[0, 0] = 1.0; lbs[1, 15] = 1.0
    faces = np.array([[0, 1, 0]])
    tri_idx = np.array([0])
    bary = np.array([[0.1, 0.8, 0.1]])  # dominant corner = vertex 1 (head)
    seg = point_segments(lbs, faces, tri_idx, bary)
    assert SEGMENTS[seg[0]] == "head"
