"""The viewer's reference object pose on the scaled/grounded stages reads the method's
own grounded 'floor' stage (rt._obj_poses_raw, scaled + floor-grounded by its builder)
as the single source -- converted to MuJoCo order -- instead of re-deriving from disk and
re-adding the floor offset. Methods with no grounded object (GMR-SOCP) fall back to the
per-method placement."""
import types

import numpy as np

from HoloNew.examples.view_stages import _displayed_object_pose


def test_uses_grounded_ground_stage_as_single_source():
    # TEST-SOCP binds the scaled + floor-grounded object pose (pose7 [qw..,xyz]) to
    # rt._obj_poses_raw. The viewer shows THAT, converted to MuJoCo order [xyz, qw..],
    # with NO extra _obj_ground_shift (the offset is already baked into the ground stage).
    T = 4
    raw = np.arange(T * 7).reshape(T, 7).astype(float)        # pose7 [qw,qx,qy,qz,x,y,z]
    rt = types.SimpleNamespace(_obj_poses_raw=raw, _obj_ground_shift=0.9)
    out = _displayed_object_pose(rt, T, lambda: object())     # fallback must NOT be used
    np.testing.assert_array_equal(out, raw[:, [4, 5, 6, 0, 1, 2, 3]])


def test_slices_to_T():
    # The displayed window is frame-aligned with qpos: slice the grounded source to T.
    T = 3
    raw = np.arange(6 * 7).reshape(6, 7).astype(float)
    rt = types.SimpleNamespace(_obj_poses_raw=raw, _obj_ground_shift=0.0)
    out = _displayed_object_pose(rt, T, lambda: None)
    assert out.shape[0] == T
    np.testing.assert_array_equal(out, raw[:T][:, [4, 5, 6, 0, 1, 2, 3]])


def test_no_grounded_object_falls_back():
    # GMR-SOCP rt has no _obj_poses_raw attribute -> use the per-method placement fallback.
    sentinel = np.zeros((4, 7))
    out = _displayed_object_pose(types.SimpleNamespace(), 4, lambda: sentinel)
    assert out is sentinel


def test_none_grounded_object_falls_back():
    # Floor-only TEST run sets _obj_poses_raw = None -> fallback.
    sentinel = np.zeros((4, 7))
    rt = types.SimpleNamespace(_obj_poses_raw=None)
    out = _displayed_object_pose(rt, 4, lambda: sentinel)
    assert out is sentinel
