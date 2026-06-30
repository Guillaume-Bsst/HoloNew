"""init : le seed f=0 place la base free-flyer à la cible pelvis du style (pos + orientation), joints
neutres, objets à leur pose observée ; warm_start recopie l'état de f-1."""
import types

import numpy as np

from src.solve.init import compute_q_init, warm_start
from src.solve.retract import so3_exp, mat_to_quat_wxyz


class _StubRobot:
    """RobotModel free-flyer minimal : neutral() = base identité (quat xyzw, qw à l'index 6) + joints 0."""

    def __init__(self, n_joints):
        self.nq = 7 + n_joints
        self.nv = 6 + n_joints

    def neutral(self):
        q = np.zeros(self.nq)
        q[6] = 1.0                                       # quat xyzw identité : qw = 1
        return q


def _ft(link_names, position, orientation, object_rot, object_pos):
    return types.SimpleNamespace(
        style=types.SimpleNamespace(link_names=link_names, position=position, orientation=orientation),
        object_rot=object_rot, object_pos=object_pos)


def test_base_placed_at_pelvis_target_no_objects():
    robot = _StubRobot(n_joints=2)
    quat_pelvis = mat_to_quat_wxyz(so3_exp(np.array([0.0, 0.0, np.pi / 2])))   # +90° z, wxyz
    ft0 = _ft(link_names=("pelvis", "torso_link"),
              position=np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.9]]),
              orientation=np.array([quat_pelvis, [1.0, 0.0, 0.0, 0.0]]),
              object_rot=np.zeros((0, 3, 3)), object_pos=np.zeros((0, 3)))
    q, poses = compute_q_init(ft0, robot)
    assert np.allclose(q[0:3], [1.0, 2.0, 3.0])                                  # base pos = cible pelvis
    assert np.allclose(q[3:7], [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)], atol=1e-10)  # xyzw
    assert np.allclose(q[7:], 0.0)                                               # joints neutres
    assert poses.shape == (0, 7)


def test_base_keeps_identity_when_orientation_none():
    robot = _StubRobot(n_joints=1)
    ft0 = _ft(link_names=("pelvis",), position=np.array([[5.0, 6.0, 7.0]]), orientation=None,
              object_rot=np.zeros((0, 3, 3)), object_pos=np.zeros((0, 3)))
    q, _ = compute_q_init(ft0, robot)
    assert np.allclose(q[0:3], [5.0, 6.0, 7.0])
    assert np.allclose(q[3:7], [0.0, 0.0, 0.0, 1.0])                             # identité xyzw conservée


def test_objects_seeded_at_observed_pose():
    robot = _StubRobot(n_joints=0)
    ft0 = _ft(link_names=("pelvis",), position=np.zeros((1, 3)),
              orientation=np.array([[1.0, 0.0, 0.0, 0.0]]),
              object_rot=np.stack([np.eye(3)]), object_pos=np.array([[8.0, 9.0, 10.0]]))
    _, poses = compute_q_init(ft0, robot)
    assert poses.shape == (1, 7)
    assert np.allclose(poses[0], [8.0, 9.0, 10.0, 1.0, 0.0, 0.0, 0.0])           # pos + quat wxyz identité


def test_warm_start_copies_state():
    q = np.arange(8.0)
    poses = np.array([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]])
    q2, p2 = warm_start(q, poses)
    assert np.allclose(q2, q) and np.allclose(p2, poses)
    q2[0] = 99.0                                                                 # copie défensive
    assert q[0] == 0.0
