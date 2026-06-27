"""Robot kinematics (UrdfRobot): generic URDF FK + the robot-keyed correspondence rest pose.
Skips when the G1 URDF is not present."""
from pathlib import Path

import numpy as np
import pytest

from src.contracts import RobotSpec
from src.prepare.load.robot import build_robot_model, correspondence_rest_angles

_URDF = Path(__file__).resolve().parent.parent / "models" / "g1" / "g1_29dof.urdf"


def test_correspondence_rest_angles_is_robot_keyed():
    assert "left_elbow_joint" in correspondence_rest_angles("g1")   # G1 defined
    with pytest.raises(ValueError):
        correspondence_rest_angles("no_such_robot")                 # unknown robot -> clear error


@pytest.mark.skipif(not _URDF.exists(), reason="G1 URDF not available")
def test_urdf_robot_fk():
    robot = build_robot_model(RobotSpec(name="g1", urdf_path=_URDF, link_names=(), dof=29, height=1.3))
    assert robot.dof == 29
    assert "pelvis" in robot.link_names
    n = len(robot.link_names)

    rot, pos = robot.rest_transforms()
    assert rot.shape == (n, 3, 3) and pos.shape == (n, 3)
    assert np.allclose(rot[0] @ rot[0].T, np.eye(3), atol=1e-6)      # orthonormal rotation

    q = np.zeros(29)
    _, pos0 = robot.link_transforms(q)
    q[3] = 0.8                                                       # bend the left knee
    _, pos1 = robot.link_transforms(q)
    assert not np.allclose(pos0, pos1)                              # some link relocated
