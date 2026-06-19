import numpy as np

from HoloNew.src.viewer import MethodViz, Viewer


def _cube():
    verts = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], np.uint32)
    return verts, faces


def _obj_kwargs(T=3):
    verts, faces = _cube()
    pose_raw = np.tile([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], (T, 1)).astype(np.float32)
    pose_raw[:, :3] = [2.0, 3.0, 1.0]
    pose_scaled = pose_raw.copy()
    pose_scaled[:, :3] = [1.0, 1.5, 0.5]   # centred
    return dict(object_mesh_verts=verts, object_mesh_faces=faces,
                object_points_local=verts.copy(), object_pose_raw=pose_raw,
                object_pose_scaled=pose_scaled,
                object_scaled_stages=("Scaled", "Robot"))


def test_object_pose_selects_scaled_vs_raw(robot_urdf):
    kw = _obj_kwargs()
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None, **kw)
    # Unscaled stage -> raw pose; scaled stages -> centred pose. Size never changes.
    np.testing.assert_array_equal(v._object_pose("Original"), kw["object_pose_raw"])
    np.testing.assert_array_equal(v._object_pose("Scaled"), kw["object_pose_scaled"])
    np.testing.assert_array_equal(v._object_pose("Robot"), kw["object_pose_scaled"])
    v.close()


def test_object_pose_uses_active_method_scaled(robot_urdf):
    # On a scaled stage the object follows the ACTIVE method's own placement, so a
    # method that keeps the object raw (TEST-SOCP: scale_*_object=1.0) shows the raw
    # trajectory, not the global holosoma/GMR-centred pose. Unscaled stages stay raw.
    kw = _obj_kwargs()
    # GMR places the object at the centred pose; TEST keeps it raw.
    gmr = MethodViz(label="GMR-SOCP", robot_key="gmr_socp", qpos=np.zeros((3, 36)),
                    stages={"Original": np.zeros((3, 5, 3)), "Scaled": np.zeros((3, 5, 3))},
                    object_pose_scaled=kw["object_pose_scaled"])
    test = MethodViz(label="TEST-SOCP", robot_key="test_socp", qpos=np.zeros((3, 36)),
                     stages={"Original": np.zeros((3, 5, 3)), "Scaled": np.zeros((3, 5, 3))},
                     object_pose_scaled=kw["object_pose_raw"])   # raw == no centring
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None,
               stage_keys=("gmr_socp", "test_socp"), **kw)
    v.bind_methods([gmr, test])
    v._method_dd.value = "GMR-SOCP"
    np.testing.assert_array_equal(v._object_pose("Scaled"), kw["object_pose_scaled"])
    v._method_dd.value = "TEST-SOCP"
    np.testing.assert_array_equal(v._object_pose("Scaled"), kw["object_pose_raw"])
    np.testing.assert_array_equal(v._object_pose("Original"), kw["object_pose_raw"])
    v.close()


def test_redraw_with_object_runs_for_each_stage(robot_urdf):
    kw = _obj_kwargs()
    m = MethodViz(label="GMR-SOCP", robot_key="gmr_socp",
                  qpos=np.zeros((3, 36)),
                  stages={"Original": np.zeros((3, 5, 3)), "Scaled": np.zeros((3, 5, 3))})
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None,
               stage_keys=("gmr_socp",), **kw)
    v.bind_methods([m])
    v._tog_object.value = True
    v._tog_object_pts.value = True
    for stage in ("Original", "Scaled", "Robot"):
        v._stage_dd.value = stage
        v._redraw(0)   # mesh + points drawn without error
    v.close()


def test_object_frame_drawn_at_object_pose(robot_urdf):
    # The "Object frame" triad sits at the object's pose (the same pose the mesh uses), so
    # the mesh's local origin is at the triad origin. Toggling off hides it.
    kw = _obj_kwargs()
    m = MethodViz(label="GMR-SOCP", robot_key="gmr_socp", qpos=np.zeros((3, 36)),
                  stages={"Original": np.zeros((3, 5, 3))})
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None,
               stage_keys=("gmr_socp",), **kw)
    v.bind_methods([m])
    v._stage_dd.value = "Original"
    v._tog_object_frame.value = True
    v._redraw(0)
    h = v._object_frame_handle
    assert h is not None and h.visible
    np.testing.assert_array_equal(h.position, kw["object_pose_raw"][0, :3])   # frame origin = pose trans
    np.testing.assert_array_equal(h.wxyz, kw["object_pose_raw"][0, 3:7])      # frame orient = pose quat
    v._tog_object_frame.value = False
    v._redraw(0)
    assert v._object_frame_handle.visible is False
    v.close()


def test_object_handles_are_persistent_not_recreated(robot_urdf):
    # The object must not flicker: its mesh/points handles are created once and updated
    # in place across redraws, never appended to the per-frame _dynamic_handles.
    kw = _obj_kwargs()
    m = MethodViz(label="GMR-SOCP", robot_key="gmr_socp",
                  qpos=np.zeros((3, 36)),
                  stages={"Original": np.zeros((3, 5, 3)), "Scaled": np.zeros((3, 5, 3))})
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None,
               stage_keys=("gmr_socp",), **kw)
    v.bind_methods([m])
    v._tog_object.value = True
    v._tog_object_pts.value = True
    v._stage_dd.value = "Scaled"
    v._redraw(0)
    mesh_h, pts_h = v._object_mesh_handle, v._object_pts_handle
    assert mesh_h is not None and pts_h is not None
    v._redraw(1)   # same handles reused (identity), not recreated
    assert v._object_mesh_handle is mesh_h
    assert v._object_pts_handle is pts_h
    # And they live outside the per-frame _dynamic_handles, so _clear_dynamic never
    # removes them (identity check; `in` would do elementwise numpy comparison).
    assert not any(h is mesh_h or h is pts_h for h in v._dynamic_handles)
    v.close()


def test_smplx_follows_active_52joint_stage_pelvis(robot_urdf):
    # The SMPL-X mesh follows the displayed skeleton: on a full 52-joint stage it uses
    # that stage's pelvis (Original -> raw/in the air, Grounded -> lowered); otherwise raw.
    oj = np.zeros((3, 52, 3), np.float32)
    oj[:, 0] = [1.0, 2.0, 3.0]                 # raw pelvis (in the air)
    grounded = oj.copy()
    grounded[:, :, 2] -= 3.0                    # lowered onto the floor
    grounded[:, 0] = [1.0, 2.0, 0.0]
    m = MethodViz(label="GMR-SOCP", robot_key="gmr_socp",
                  qpos=np.zeros((3, 36)),
                  stages={"Original": oj, "Grounded": grounded,
                          "Mapped": np.zeros((3, 14, 3))})
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None,
               stage_keys=("gmr_socp",), original_joints=oj)
    v.bind_methods([m])
    v._stage_dd.value = "Original"
    np.testing.assert_array_equal(v._active_human_pelvis(0), [1.0, 2.0, 3.0])
    v._stage_dd.value = "Grounded"
    np.testing.assert_array_equal(v._active_human_pelvis(0), [1.0, 2.0, 0.0])
    v._stage_dd.value = "Mapped"   # not 52-joint -> falls back to raw pelvis
    np.testing.assert_array_equal(v._active_human_pelvis(0), [1.0, 2.0, 3.0])
    v.close()


class _FakeBody:
    faces = np.array([[0, 1, 2]], np.uint32)

    def placed_verts(self, quats, pelvis, frame_idx=None):
        return (np.zeros((3, 3), np.float32) + pelvis).astype(np.float32)


def test_smplx_mesh_only_visible_on_original_and_grounded(robot_urdf):
    T = 3
    oj = np.zeros((T, 52, 3), np.float32)
    grounded = oj.copy(); grounded[:, :, 2] -= 1.0
    m = MethodViz(label="GMR-SOCP", robot_key="gmr_socp",
                  qpos=np.zeros((T, 36)),
                  stages={"Original": oj, "Grounded": grounded,
                          "Mapped": np.zeros((T, 14, 3))})
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None,
               stage_keys=("gmr_socp",), original_joints=oj,
               original_quats=np.zeros((T, 52, 4), np.float32), human_body=_FakeBody())
    v.bind_methods([m])
    v._tog_smplx.value = True
    for stage, visible in (("Original", True), ("Grounded", True), ("Mapped", False), ("Robot", False)):
        v._stage_dd.value = stage
        v._redraw(0)
        assert bool(v._smplx_handle.visible) is visible, stage
    v.close()


def test_g1_points_only_visible_on_robot_stage(robot_urdf):
    T = 3
    m = MethodViz(label="GMR-SOCP", robot_key="gmr_socp",
                  qpos=np.zeros((T, 36)),
                  stages={"Original": np.zeros((T, 52, 3)), "Mapped": np.zeros((T, 14, 3))},
                  g1_points=np.zeros((T, 8, 3), np.float32))
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None,
               stage_keys=("gmr_socp",))
    v.bind_methods([m])
    v._tog_g1_pts.value = True
    for stage, visible in (("Robot", True), ("Original", False), ("Mapped", False)):
        v._stage_dd.value = stage
        v._redraw(0)
        assert bool(v._g1_pts_handle.visible) is visible, stage
    v.close()


def test_object_absent_is_noop(robot_urdf):
    m = MethodViz(label="GMR-SOCP", robot_key="gmr_socp",
                  qpos=np.zeros((3, 36)), stages={"Original": np.zeros((3, 5, 3))})
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None,
               stage_keys=("gmr_socp",))
    v.bind_methods([m])
    v._tog_object.value = True
    v._tog_object_pts.value = True
    v._stage_dd.value = "Original"
    v._redraw(0)   # no object data -> _draw_object is a no-op, must not raise
    v.close()
