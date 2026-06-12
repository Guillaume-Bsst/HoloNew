# tests/test_skeleton.py
import numpy as np
from HoloNew.config_types.data_type import SMPLH_DEMO_JOINTS
from HoloNew.src import skeleton


def test_bone_and_joint_indices_in_range():
    n = len(SMPLH_DEMO_JOINTS)  # 52
    all_bones = skeleton.BODY_BONES + skeleton.FINGER_BONES
    for a, b in all_bones:
        assert 0 <= a < n and 0 <= b < n
    for i in skeleton.BODY_JOINT_INDICES + skeleton.FINGER_JOINT_INDICES:
        assert 0 <= i < n


def test_body_and_finger_joint_sets_disjoint_and_cover():
    body = set(skeleton.BODY_JOINT_INDICES)
    finger = set(skeleton.FINGER_JOINT_INDICES)
    assert body.isdisjoint(finger)
    assert body | finger == set(range(len(SMPLH_DEMO_JOINTS)))


def test_bones_for_subset_simple_chain():
    # Pelvis(0)-L_Hip(1)-L_Knee(2)-L_Ankle(3): a straight kinematic chain.
    assert skeleton.bones_for_subset([0, 1, 2, 3]) == [(0, 1), (1, 2), (2, 3)]


def test_bones_for_subset_links_to_nearest_present_ancestor():
    # With the spine absent, L_Shoulder(15) attaches to the pelvis(0); the subset
    # root (pelvis) contributes no bone.
    bones = skeleton.bones_for_subset([0, 15, 16])  # pelvis, L_Shoulder, L_Elbow
    assert bones == [(0, 1), (1, 2)]


def test_colors_are_uint8_rgb():
    for c in (skeleton.COLOR_BODY, skeleton.COLOR_FINGER,
              skeleton.COLOR_GHOST_BODY, skeleton.COLOR_GHOST_FINGER,
              skeleton.COLOR_STAGE, skeleton.COLOR_GHOST_STAGE):
        assert c.dtype == np.uint8 and c.shape == (3,)
