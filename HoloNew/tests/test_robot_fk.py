import numpy as np
from HoloNew.src.robot_fk import robot_link_positions

_MJCF = "models/g1/g1_29dof.xml"


def test_pelvis_follows_base_and_knee_offset():
    qpos = np.zeros((2, 36), dtype=np.float32)
    qpos[:, 3] = 1.0                  # identity base quaternion (wxyz)
    qpos[1, :3] = [1.0, 2.0, 3.0]     # translate the base on frame 1
    pos = robot_link_positions(_MJCF, ["pelvis", "left_knee_link"], qpos)
    assert pos.shape == (2, 2, 3)
    # The free-base body (pelvis) sits at the base position.
    np.testing.assert_allclose(pos[0, 0], [0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(pos[1, 0], [1, 2, 3], atol=1e-6)
    # The knee is a distinct link, offset from the pelvis.
    assert not np.allclose(pos[0, 1], pos[0, 0])
