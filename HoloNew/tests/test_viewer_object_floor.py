"""Viewer object-as-carrier <-> floor channel (Object surface pts / Object->Floor
directions / Object->Floor contact). The object-local surface samples are lifted by the
object pose, which must honour the 'Solved object pose' toggle exactly like the box and
the robot/box direction channels: solved pose when on, stage reference pose when off."""
import numpy as np

from HoloNew.src.viewer import MethodViz, Viewer


def _method(T=2):
    # One surface point at the object origin; the object pose alone sets its world height.
    surf = np.array([[0.0, 0.0, 0.0]], np.float32)
    # Solved pose [qw,qx,qy,qz,x,y,z] places the point NEAR the floor (z=0.05 < 0.10 margin).
    solved = np.tile([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05], (T, 1)).astype(np.float32)
    return MethodViz(label="TEST-SOCP", robot_key="test_socp", qpos=np.zeros((T, 36)),
                     stages={"Original": np.zeros((T, 5, 3)), "Robot": np.zeros((T, 5, 3))},
                     object_surface_local=surf, solved_object_poses=solved)


def _viewer(robot_urdf, m, T=2):
    # Reference (raw) pose [x,y,z,qw,qx,qy,qz] places the point HIGH (z=1.0, off the floor).
    pose_raw = np.tile([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], (T, 1)).astype(np.float32)
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None,
               stage_keys=("test_socp",), object_pose_raw=pose_raw, object_pose_scaled=pose_raw)
    v.bind_methods([m])
    return v


def test_solved_or_ref_pose_follows_toggle(robot_urdf):
    v = _viewer(robot_urdf, _method())
    v._stage_dd.value = "Original"
    v._tog_solved_obj.value = True
    _, t, ok = v._solved_or_ref_object_pose(0, "Original")
    assert ok and t[2] == 0.05          # solved height
    v._tog_solved_obj.value = False
    _, t, ok = v._solved_or_ref_object_pose(0, "Original")
    assert ok and t[2] == 1.0           # reference (raw) height
    v.close()


def test_object_floor_contact_follows_solved_vs_reference(robot_urdf):
    # The near-floor contact footprint must appear under the SOLVED pose (point at z=0.05)
    # and vanish under the REFERENCE pose (point at z=1.0): the channel tracks
    # solved-vs-origin like the box, per the requirement.
    v = _viewer(robot_urdf, _method())
    v._stage_dd.value = "Original"
    v._tog_obj_floor_contact.value = True
    v._tog_solved_obj.value = True
    v._redraw(0)
    assert v._object_floor_contact_handle is not None
    assert bool(v._object_floor_contact_handle.visible) is True
    v._tog_solved_obj.value = False     # reference pose -> point high -> no contact
    v._redraw(0)
    assert bool(v._object_floor_contact_handle.visible) is False
    v.close()


def test_object_floor_directions_drawn_when_near(robot_urdf):
    # A near-floor point under the solved pose adds a direction segment; the high reference
    # point draws none.
    v = _viewer(robot_urdf, _method())
    v._stage_dd.value = "Original"
    v._tog_dir_obj_floor.value = True
    v._tog_solved_obj.value = True
    v._redraw(0)
    n_with = len(v._dynamic_handles)    # one batched object->floor segment handle, near pose
    v._tog_solved_obj.value = False
    v._redraw(0)                        # reference point high -> no object/floor segment
    n_without = len(v._dynamic_handles)
    assert n_with == n_without + 1      # the only per-frame difference is the direction segment
    v.close()


def test_object_surface_points_visible_and_persistent(robot_urdf):
    v = _viewer(robot_urdf, _method())
    v._stage_dd.value = "Robot"
    v._tog_obj_surface.value = True
    v._redraw(0)
    h = v._object_surface_handle
    assert h is not None and bool(h.visible) is True
    v._redraw(1)                        # persistent handle reused, lives outside _dynamic_handles
    assert v._object_surface_handle is h
    assert not any(x is h for x in v._dynamic_handles)
    v.close()


def test_object_floor_absent_is_noop(robot_urdf):
    # No object_surface_local on the method -> the channel is a no-op, must not raise.
    m = MethodViz(label="TEST-SOCP", robot_key="test_socp", qpos=np.zeros((2, 36)),
                  stages={"Original": np.zeros((2, 5, 3))})
    v = _viewer(robot_urdf, m)
    v._stage_dd.value = "Original"
    v._tog_obj_surface.value = True
    v._tog_dir_obj_floor.value = True
    v._tog_obj_floor_contact.value = True
    v._redraw(0)
    v.close()
