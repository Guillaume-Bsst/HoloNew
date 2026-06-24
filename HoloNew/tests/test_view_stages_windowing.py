"""--start-frame windowing must slice EVERY per-frame solve input, including the contact
probe's object poses. The probe (smplx_ground_probe) holds its OWN full-clip copy of the
object poses (obj_quat/obj_trans), separate from rt._obj_poses_raw. If _window_solve_frames
slices only rt._obj_poses_raw, the probe queries the windowed human (frame start+t) against
the un-windowed object (frame t) -> the object contact/direction overlays attach to the
wrong body part (the human's nearest point to the stale floor-object is the feet)."""
import types

import numpy as np

from HoloNew.examples.view_stages import _window_solve_frames


def _fake_rt(T):
    probe = types.SimpleNamespace(
        obj_quat=np.arange(T * 4).reshape(T, 4).astype(float),
        obj_trans=np.arange(T * 3).reshape(T, 3).astype(float))
    return types.SimpleNamespace(
        gmr_stages=None, gmr_ground=None,
        gmr_grounded=np.zeros((T, 52, 3)),
        _obj_poses_raw=np.zeros((T, 7)), _obj_poses_mj=None,
        _smplx_orientations=np.zeros((T, 22, 4)), human_quat=np.zeros((T, 52, 4)),
        foot_sticking_sequences=None, smplx_ground_probe=probe)


def test_window_slices_probe_object_poses():
    T, start = 100, 30
    rt = _fake_rt(T)
    _window_solve_frames(rt, start)
    # the probe's object poses are windowed in lockstep with rt._obj_poses_raw
    assert rt._obj_poses_raw.shape[0] == T - start
    assert rt.smplx_ground_probe.obj_quat.shape[0] == T - start
    assert rt.smplx_ground_probe.obj_trans.shape[0] == T - start
    # local frame 0 maps to global frame `start`
    np.testing.assert_array_equal(
        rt.smplx_ground_probe.obj_quat[0], np.arange(start * 4, start * 4 + 4))


def test_window_floor_only_probe_no_crash():
    # Floor-only run: the probe has no object (obj_quat is None). Must not crash.
    T, start = 50, 10
    rt = _fake_rt(T)
    rt.smplx_ground_probe.obj_quat = None
    rt.smplx_ground_probe.obj_trans = None
    _window_solve_frames(rt, start)   # no exception
    assert rt.smplx_ground_probe.obj_quat is None


def test_window_no_probe_attr_no_crash():
    # GMR-SOCP retargeter has no smplx_ground_probe at all.
    T, start = 40, 5
    rt = _fake_rt(T)
    del rt.smplx_ground_probe
    _window_solve_frames(rt, start)   # no exception
    assert rt._smplx_orientations.shape[0] == T - start


def test_window_slices_stage_object_pose():
    # The grounded object lives in gmr_stages["ground"]["object_pose"] and must be
    # windowed in lockstep with _obj_poses_raw (same data the builder binds them to).
    T, start = 60, 20
    op = np.arange(T * 7).reshape(T, 7).astype(float)
    ground = {"pos": np.zeros((T, 5, 3)), "quat": np.zeros((T, 5, 4)), "object_pose": op}
    rt = types.SimpleNamespace(
        gmr_stages={"ground": ground}, gmr_ground=ground,
        gmr_grounded=np.zeros((T, 52, 3)),
        _obj_poses_raw=op.copy(), _obj_poses_mj=None,
        _smplx_orientations=np.zeros((T, 22, 4)), human_quat=np.zeros((T, 52, 4)),
        foot_sticking_sequences=None,
        smplx_ground_probe=types.SimpleNamespace(obj_quat=None, obj_trans=None))
    _window_solve_frames(rt, start)
    assert rt.gmr_stages["ground"]["object_pose"].shape[0] == T - start
    # local frame 0 maps to global frame `start`
    np.testing.assert_array_equal(
        rt.gmr_stages["ground"]["object_pose"][0], np.arange(start * 7, start * 7 + 7))
