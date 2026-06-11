import numpy as np
from HoloNew.src.viewer import Viewer

def test_viewer_creates_named_robot_root(robot_urdf):
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None)
    assert "socp" in v.robots
    v.close()

def test_draw_q_sets_base_pose(robot_urdf):
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None)
    dof = v.robots["socp"].dof
    q = np.zeros(7 + dof)
    q[:3] = [1.0, 2.0, 3.0]
    q[3:7] = [1.0, 0.0, 0.0, 0.0]
    v.draw_q(q, stage="socp")
    np.testing.assert_allclose(v.robots["socp"].base.position, [1.0, 2.0, 3.0])
    v.close()
