import numpy as np
from HoloNew.src.stages import METHODS
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

def test_builds_one_robot_per_method(robot_urdf):
    keys = tuple(m.robot_key for m in METHODS)
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None, stage_keys=keys)
    assert set(v.robots) == set(keys)
    v.close()

def test_bind_methods_builds_method_and_stage(robot_urdf):
    import numpy as np
    from HoloNew.src.viewer import Viewer, MethodViz
    m = MethodViz(label="GMR-SOCP v1", robot_key="gmr_socp_v1",
                  qpos=np.zeros((3, 36)),
                  stages={"Original": np.zeros((3, 5, 3)), "Mapped": np.zeros((3, 5, 3))})
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None, stage_keys=("gmr_socp_v1",))
    v.bind_methods([m])
    assert v._method_dd.value == "GMR-SOCP v1"
    # selecting a skeleton stage and the Robot stage both redraw without error
    v._method_dd.value = "GMR-SOCP v1"; v._stage_dd.value = "Mapped"; v._redraw(0)
    v._stage_dd.value = "Robot"; v._redraw(0)
    v.close()
