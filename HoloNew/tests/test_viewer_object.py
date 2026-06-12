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


def test_redraw_with_object_runs_for_each_stage(robot_urdf):
    kw = _obj_kwargs()
    m = MethodViz(label="GMR-SOCP v1", robot_key="gmr_socp_v1",
                  qpos=np.zeros((3, 36)),
                  stages={"Original": np.zeros((3, 5, 3)), "Scaled": np.zeros((3, 5, 3))})
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None,
               stage_keys=("gmr_socp_v1",), **kw)
    v.bind_methods([m])
    v._tog_object.value = True
    v._tog_object_pts.value = True
    for stage in ("Original", "Scaled", "Robot"):
        v._stage_dd.value = stage
        v._redraw(0)   # mesh + points drawn without error
    v.close()


def test_object_absent_is_noop(robot_urdf):
    m = MethodViz(label="GMR-SOCP v1", robot_key="gmr_socp_v1",
                  qpos=np.zeros((3, 36)), stages={"Original": np.zeros((3, 5, 3))})
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None,
               stage_keys=("gmr_socp_v1",))
    v.bind_methods([m])
    v._tog_object.value = True
    v._tog_object_pts.value = True
    v._stage_dd.value = "Original"
    v._redraw(0)   # no object data -> _draw_object is a no-op, must not raise
    v.close()
