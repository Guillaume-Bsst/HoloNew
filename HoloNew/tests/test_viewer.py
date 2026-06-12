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
    m = MethodViz(label="GMR-SOCP", robot_key="gmr_socp",
                  qpos=np.zeros((3, 36)),
                  stages={"Original": np.zeros((3, 5, 3)), "Mapped": np.zeros((3, 5, 3))})
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None, stage_keys=("gmr_socp",))
    v.bind_methods([m])
    assert v._method_dd.value == "GMR-SOCP"
    # selecting a skeleton stage and the Robot stage both redraw without error
    v._method_dd.value = "GMR-SOCP"; v._stage_dd.value = "Mapped"; v._redraw(0)
    v._stage_dd.value = "Robot"; v._redraw(0)
    v.close()

def test_viewer_stores_original_motion(robot_urdf):
    import numpy as np
    from HoloNew.src.viewer import Viewer
    oj = np.zeros((4, 52, 3), dtype=np.float32)
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None,
               original_joints=oj)
    assert v.original_joints.shape == (4, 52, 3)
    assert v.original_quats is None and v.human_body is None
    v.close()

def test_original_stage_renders_with_toggles(robot_urdf):
    import numpy as np
    from HoloNew.src.viewer import Viewer, MethodViz
    oj = np.zeros((3, 52, 3), dtype=np.float32)
    m = MethodViz(label="GMR-SOCP", robot_key="gmr_socp",
                  qpos=np.zeros((3, 36)),
                  stages={"Original": oj, "Mapped": np.zeros((3, 14, 3))})
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None,
               stage_keys=("gmr_socp",), original_joints=oj)
    v.bind_methods([m])
    for cb in (v._tog_body_bones, v._tog_finger_bones,
               v._tog_body_joints, v._tog_finger_joints):
        assert cb.value in (True, False)
    v._stage_dd.value = "Original"; v._redraw(0)
    v._tog_finger_bones.value = False; v._redraw(0)
    v._stage_dd.value = "Mapped"; v._redraw(0)
    v.close()

def test_smplx_toggle_noop_without_body(robot_urdf):
    import numpy as np
    from HoloNew.src.viewer import Viewer, MethodViz
    oj = np.zeros((3, 52, 3), dtype=np.float32)
    m = MethodViz(label="GMR-SOCP", robot_key="gmr_socp",
                  qpos=np.zeros((3, 36)), stages={"Original": oj})
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None,
               stage_keys=("gmr_socp",), original_joints=oj,
               original_quats=None, human_body=None)
    v.bind_methods([m])
    v._tog_smplx.value = True      # no human_body -> must not raise
    v._stage_dd.value = "Original"; v._redraw(0)
    assert v._smplx_handle is None
    v.close()

def test_ghost_overlays_skeleton_stage(robot_urdf):
    import numpy as np
    from HoloNew.src.viewer import Viewer, MethodViz
    oj = np.zeros((3, 52, 3), dtype=np.float32)
    m1 = MethodViz(label="GMR-SOCP", robot_key="gmr_socp",
                   qpos=np.zeros((3, 36)),
                   stages={"Original": oj, "Mapped": np.zeros((3, 14, 3))})
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None,
               stage_keys=("gmr_socp",), original_joints=oj)
    v.bind_methods([m1])
    assert v._ghost_stage_dd.value == "Off"
    assert "Robot" not in v._ghost_stage_dd.options
    # Ghost Method offers only bound methods, so an unsolved method cannot be
    # selected (would KeyError in _redraw).
    assert list(v._ghost_method_dd.options) == ["GMR-SOCP"]
    v._ghost_method_dd.value = "GMR-SOCP"
    v._ghost_stage_dd.value = "Mapped"
    v._redraw(0)   # must not raise
    assert len(v._dynamic_handles) > 0   # ghost actually drew something
    v._ghost_stage_dd.value = "Off"
    v._redraw(0)
    v.close()


def test_playback_controls_and_advance(robot_urdf):
    import numpy as np
    from HoloNew.src.viewer import Viewer, MethodViz
    oj = np.zeros((4, 52, 3), dtype=np.float32)
    m = MethodViz(label="GMR-SOCP", robot_key="gmr_socp",
                  qpos=np.zeros((4, 36)), stages={"Original": oj})
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None,
               stage_keys=("gmr_socp",), original_joints=oj)
    v.bind_methods([m])
    assert hasattr(v, "_play_btn") and hasattr(v, "_fps_in")
    assert v._playing is False
    # _advance_frame wraps from the last frame back to 0
    v._slider.value = v._n_frames - 1
    assert v._advance_frame() == 0
    assert int(v._slider.value) == 0
    v.close()


def test_mapped_stage_bones_and_robot_skeleton_with_urdf_toggle(robot_urdf):
    import numpy as np
    from HoloNew.src.viewer import Viewer, MethodViz
    oj = np.zeros((3, 52, 3), dtype=np.float32)
    rs = np.zeros((3, 14, 3), dtype=np.float32)
    bones = [(0, 1), (1, 2)]
    m = MethodViz(label="GMR-SOCP", robot_key="gmr_socp",
                  qpos=np.zeros((3, 36)),
                  stages={"Original": oj, "Mapped": np.zeros((3, 14, 3))},
                  stage_bones={"Mapped": bones, "Robot": bones},
                  robot_skeleton=rs)
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None,
               stage_keys=("gmr_socp",), original_joints=oj)
    v.bind_methods([m])
    assert v._tog_urdf.value is True
    # A mapped stage now draws bones + joints (2 handles), not just points.
    v._stage_dd.value = "Mapped"; v._redraw(0)
    assert len(v._dynamic_handles) == 2
    # Robot stage: hiding the URDF leaves the mesh invisible but the solved-robot
    # skeleton (bones + joints) is drawn underneath.
    v._stage_dd.value = "Robot"; v._tog_urdf.value = False; v._redraw(0)
    assert v.robots["gmr_socp"].urdf.show_visual is False
    assert len(v._dynamic_handles) == 2
    v.close()


def test_joint_frames_axes_toggle(robot_urdf):
    import numpy as np
    from HoloNew.src.viewer import Viewer, MethodViz
    oj = np.zeros((3, 52, 3), dtype=np.float32)
    quat = np.zeros((3, 14, 4), dtype=np.float32)
    quat[..., 0] = 1.0   # identity wxyz
    m = MethodViz(label="GMR-SOCP", robot_key="gmr_socp",
                  qpos=np.zeros((3, 36)),
                  stages={"Original": oj, "Mapped": np.zeros((3, 14, 3))},
                  stage_bones={"Mapped": [(0, 1)]},
                  stage_quats={"Mapped": quat})
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None,
               stage_keys=("gmr_socp",), original_joints=oj)
    v.bind_methods([m])
    assert hasattr(v, "_tog_axes") and hasattr(v, "_axis_size")
    v._stage_dd.value = "Mapped"
    v._tog_axes.value = False; v._redraw(0); n_off = len(v._dynamic_handles)
    v._tog_axes.value = True;  v._redraw(0); n_on = len(v._dynamic_handles)
    assert n_on == n_off + 1   # one extra handle for the joint-frame axes
    v.close()


def test_holosoma_g1_points_toggle(robot_urdf):
    import numpy as np
    from HoloNew.src.viewer import Viewer, MethodViz
    oj = np.zeros((3, 52, 3), dtype=np.float32)
    g1 = np.zeros((3, 15, 3), dtype=np.float32)
    m = MethodViz(label="GMR-SOCP", robot_key="gmr_socp",
                  qpos=np.zeros((3, 36)), stages={"Original": oj}, g1_points=g1)
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None,
               stage_keys=("gmr_socp",), original_joints=oj)
    v.bind_methods([m])
    assert hasattr(v, "_tog_g1_pts")
    assert v._g1_pts_handle is None
    v._tog_g1_pts.value = True; v._redraw(0)
    assert v._g1_pts_handle is not None and v._g1_pts_handle.visible
    v._tog_g1_pts.value = False; v._redraw(0)
    assert v._g1_pts_handle.visible is False
    v.close()


def test_sdf_object_toggle(robot_urdf):
    import numpy as np
    from HoloNew.src.viewer import Viewer, MethodViz
    oj = np.zeros((3, 52, 3), dtype=np.float32)
    band = np.zeros((5, 3), dtype=np.float32)
    cols = np.zeros((5, 3), dtype=np.uint8)
    pose = np.zeros((3, 7), dtype=np.float32)
    pose[:, 3] = 1.0   # identity quaternion (wxyz)
    m = MethodViz(label="GMR-SOCP", robot_key="gmr_socp",
                  qpos=np.zeros((3, 36)), stages={"Original": oj})
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None,
               stage_keys=("gmr_socp",), original_joints=oj,
               object_sdf_pts=band, object_sdf_cols=cols, object_pose_raw=pose)
    v.bind_methods([m])
    assert hasattr(v, "_tog_sdf")
    assert v._sdf_handle is None
    v._tog_sdf.value = True; v._redraw(0)
    assert v._sdf_handle is not None and v._sdf_handle.visible
    v._tog_sdf.value = False; v._redraw(0)
    assert v._sdf_handle.visible is False
    v.close()


def test_interaction_toggles_and_gating(robot_urdf):
    import numpy as np
    from HoloNew.src.viewer import Viewer, MethodViz
    oj = np.zeros((3, 52, 3), dtype=np.float32)
    human = np.zeros((3, 6, 3), dtype=np.float32)
    g1 = np.zeros((3, 4, 3), dtype=np.float32)
    dist = np.zeros((3, 6), dtype=np.float32)
    m = MethodViz(label="TEST-SOCP", robot_key="test_socp",
                  qpos=np.zeros((3, 36)),
                  stages={"Original": oj, "Grounded": oj},
                  human_probe_pts=human, human_dist=dist,
                  g1_transport_pts=g1, g1_dist=dist[:, :4])
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None,
               stage_keys=("test_socp",), original_joints=oj)
    v.bind_methods([m])
    assert hasattr(v, "_tog_human") and hasattr(v, "_tog_g1_transport")
    # Human contact gated to the Grounded stage.
    v._tog_human.value = True
    v._stage_dd.value = "Robot"; v._redraw(0)
    assert v._human_handle is None or v._human_handle.visible is False
    v._stage_dd.value = "Grounded"; v._redraw(0)
    assert v._human_handle is not None and v._human_handle.visible
    # G1 transport gated to the Robot stage.
    v._tog_g1_transport.value = True
    v._stage_dd.value = "Robot"; v._redraw(0)
    assert v._g1_transport_handle is not None and v._g1_transport_handle.visible
    v.close()


def test_object_floor_footprints_grounded_stage(robot_urdf):
    import numpy as np
    from HoloNew.src.viewer import Viewer, MethodViz
    oj = np.zeros((3, 52, 3), dtype=np.float32)
    N = 6
    probes = np.zeros((3, N, 3), dtype=np.float32)
    wit = np.zeros((3, N, 3), dtype=np.float32)
    obj_d = np.zeros((3, N), dtype=np.float32)   # all probes active (0 < margin)
    flr_d = np.zeros((3, N), dtype=np.float32)
    m = MethodViz(label="TEST-SOCP", robot_key="test_socp", qpos=np.zeros((3, 36)),
                  stages={"Original": oj, "Grounded": oj}, human_probe_pts=probes,
                  human_witness=wit, human_obj_dist=obj_d, human_flr_dist=flr_d)
    pose = np.zeros((3, 7), dtype=np.float32)
    pose[:, 3] = 1.0   # identity quaternion (wxyz), needed to lift the object-local witness
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None,
               stage_keys=("test_socp",), original_joints=oj, object_pose_raw=pose)
    v.bind_methods([m])
    assert hasattr(v, "_tog_object_contact") and hasattr(v, "_tog_floor_contact")
    v._tog_object_contact.value = True
    v._tog_floor_contact.value = True
    # Gated to Grounded: hidden on Robot, shown on Grounded.
    v._stage_dd.value = "Robot"; v._redraw(0)
    assert v._object_contact_handle is None or v._object_contact_handle.visible is False
    v._stage_dd.value = "Grounded"; v._redraw(0)
    assert v._object_contact_handle is not None and v._object_contact_handle.visible
    assert v._floor_contact_handle is not None and v._floor_contact_handle.visible
    v.close()
