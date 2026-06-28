"""targets/pipeline tests: the implemented per-frame state + the trivial target assemblers (the pure
ops eval/transport/style are still stubs). Synthetic, torch-free — a fake BodyModel feeds canned bone
transforms so ``frame_pose`` is exercised without SMPL."""
from pathlib import Path

import numpy as np
import pytest

from src.prepare.contracts import Calibration, GroundedScene
from src.targets.contracts import MultiChannelField
from src.targets.interaction import (environment_interaction_targets, robot_interaction_targets)
from src.targets.pipeline import _pose7_to_Rt, frame_pose


class _FakeBody:
    """Minimal BodyModel: canned bone transforms (frame index leaks into the z of bone 0)."""
    faces = np.zeros((1, 3), np.int64)
    n_bones = 2
    stature = 1.7

    def bone_transforms(self, params, t):
        rot = np.stack([np.eye(3), np.eye(3)])              # (2, 3, 3)
        pos = np.array([[0.0, 0.0, float(t)], [1.0, 0.0, 0.0]])   # (2, 3)
        return rot, pos


def _grounded(body, n_obj: int = 1, T: int = 3) -> GroundedScene:
    obj = np.tile([1.0, 2.0, 3.0, 1, 0, 0, 0], (T, 1)).astype(np.float32)   # identity-quat pose
    return GroundedScene(joint_pos=np.zeros((T, 4, 3), np.float32), joint_names=("a", "b", "c", "d"),
                         object_poses=(obj,) * n_obj, object_mesh_paths=(Path("o.obj"),) * n_obj,
                         calibration=Calibration(0.0, 0.0, np.eye(4)), fps=30.0,
                         smpl_params=None, body=body)


def test_pose7_to_Rt_identity_and_z_rotation():
    rot, t = _pose7_to_Rt([1.0, 2.0, 3.0, 1, 0, 0, 0])             # identity quaternion
    assert np.allclose(rot, np.eye(3)) and np.allclose(t, [1.0, 2.0, 3.0])
    s = np.sqrt(0.5)
    rot, _ = _pose7_to_Rt([0, 0, 0, s, 0, 0, s])                   # +90 deg about z: x -> y
    assert np.allclose(rot @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-7)


def test_frame_pose_with_body_poses_bones_and_objects():
    g = _grounded(_FakeBody(), n_obj=2)
    pose = frame_pose(g, f=2)
    assert pose.bone_rot.shape == (2, 3, 3) and pose.bone_pos.shape == (2, 3)
    assert np.allclose(pose.bone_pos[0], [0.0, 0.0, 2.0])          # frame index leaked into bone 0 z
    assert pose.object_rot.shape == (2, 3, 3) and pose.object_pos.shape == (2, 3)
    assert np.allclose(pose.object_rot[0], np.eye(3))              # identity-quat object
    assert np.allclose(pose.object_pos[1], [1.0, 2.0, 3.0])


def test_frame_pose_without_body_has_no_bones_but_keeps_objects():
    g = _grounded(body=None, n_obj=1)
    pose = frame_pose(g, f=0)
    assert pose.bone_rot.shape == (0, 3, 3) and pose.bone_pos.shape == (0, 3)
    assert pose.object_pos.shape == (1, 3) and np.allclose(pose.object_pos[0], [1.0, 2.0, 3.0])


def _field(c: int, p: int) -> MultiChannelField:
    return MultiChannelField(distance=np.zeros((c, p)), direction=np.zeros((c, p, 3)),
                             witness=np.zeros((c, p, 3)), active=np.zeros((c, p), bool),
                             channels=tuple(f"ch{i}" for i in range(c)))


def test_target_assemblers_wrap_fields():
    rf = _field(2, 5)
    assert robot_interaction_targets(rf).field is rf
    of = (_field(2, 3), _field(2, 4))
    assert environment_interaction_targets(of).per_object == of
