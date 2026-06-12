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


def test_colors_are_uint8_rgb():
    for c in (skeleton.COLOR_BODY, skeleton.COLOR_FINGER,
              skeleton.COLOR_GHOST_BODY, skeleton.COLOR_GHOST_FINGER,
              skeleton.COLOR_STAGE, skeleton.COLOR_GHOST_STAGE):
        assert c.dtype == np.uint8 and c.shape == (3,)
