"""retract : délégation robot.integrate (round-trip) + exp SE(3) objet (rotation/translation connues),
et les helpers quaternion/Rodrigues (vs valeurs analytiques)."""
import numpy as np

from src.solve.contracts import Step
from src.solve.retract import (mat_to_quat_wxyz, quat_wxyz_to_mat, quat_wxyz_to_xyzw, retract, so3_exp)


class _StubRobot:
    """RobotModel minimal, Euclidien (integrate additif) — teste la DÉLÉGATION de retract, pas la FK
    free-flyer (couverte par les tests pinocchio de prepare)."""

    def __init__(self, n):
        self.nq = self.nv = n

    def integrate(self, q, v):
        return np.asarray(q, np.float64) + np.asarray(v, np.float64)


def test_so3_exp_known_rotation():
    R = so3_exp(np.array([0.0, 0.0, np.pi / 2]))          # +90° autour de z
    assert np.allclose(R, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)


def test_so3_exp_small_angle_is_near_identity():
    R = so3_exp(np.array([1e-10, 0.0, 0.0]))
    assert np.allclose(R, np.eye(3), atol=1e-8)


def test_quat_mat_round_trip():
    R = so3_exp(np.array([0.3, -0.7, 1.1]))
    assert np.allclose(quat_wxyz_to_mat(mat_to_quat_wxyz(R)), R, atol=1e-12)


def test_quat_wxyz_to_xyzw_reorders():
    assert np.allclose(quat_wxyz_to_xyzw(np.array([0.1, 0.2, 0.3, 0.4])), [0.2, 0.3, 0.4, 0.1])


def test_retract_robot_round_trip():
    robot = _StubRobot(3)
    q0 = np.array([0.1, 0.2, 0.3])
    poses = np.zeros((0, 7))
    q1, p1 = retract(q0, poses, Step(dv=np.array([1.0, 1.0, 1.0]), dxi=None, value=0.0, status="optimal"), robot)
    assert np.allclose(q1, [1.1, 1.2, 1.3])
    q2, _ = retract(q1, p1, Step(dv=np.array([-1.0, -1.0, -1.0]), dxi=None, value=0.0, status="optimal"), robot)
    assert np.allclose(q2, q0)                            # round-trip exact (stub Euclidien)
    assert p1.shape == (0, 7)


def test_retract_object_exp_known_motion():
    robot = _StubRobot(2)
    poses = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])   # pose identité (quat wxyz)
    dxi = np.array([[0.1, 0.2, 0.3, 0.0, 0.0, np.pi / 2]])    # δt + δθ (+90° z)
    _, p = retract(np.zeros(2), poses, Step(dv=np.zeros(2), dxi=dxi, value=0.0, status="optimal"), robot)
    assert np.allclose(p[0, :3], [0.1, 0.2, 0.3], atol=1e-12)
    assert np.allclose(p[0, 3:7], [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)], atol=1e-10)


def test_retract_object_pure_translation_keeps_orientation():
    robot = _StubRobot(2)
    poses = np.array([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
    dxi = np.array([[2.0, 0.0, -1.0, 0.0, 0.0, 0.0]])
    _, p = retract(np.zeros(2), poses, Step(dv=np.zeros(2), dxi=dxi, value=0.0, status="optimal"), robot)
    assert np.allclose(p[0], [3.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0], atol=1e-12)
