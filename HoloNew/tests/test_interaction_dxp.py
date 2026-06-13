"""Tests for Brick 1 interaction D/X/P: robot-side field query and per-frame references."""
import numpy as np
import pytest


def _rt():
    from HoloNew.examples.robot_retarget import RetargetingConfig
    from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
    return TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="object_interaction", task_name="sub3_largebox_003", data_format="smplh"))


def test_robot_control_points_and_query_shapes():
    from HoloNew.src.test_socp.interaction import robot_control_points, query_entities

    rt = _rt()
    if rt.correspondence is None or rt.object_sdf is None:
        pytest.skip("correspondence/object_sdf assets not present")

    q_pin = rt.pin.qpos_mj_to_q_pin(rt.q_init_full[:36])
    P = robot_control_points(rt, q_pin)                  # (M, 3) world
    assert P.shape == (rt.correspondence.link_idx.shape[0], 3)

    # Identity object pose [qw, qx, qy, qz, x, y, z] — tests shape, not correctness.
    obj_pose = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    fobj, fflr = query_entities(rt, P, obj_pose)
    assert fobj.distance.shape == (P.shape[0],)
    assert fflr.distance.shape == (P.shape[0],)


def test_reference_extraction_aligns_with_control_points():
    rt = _rt()
    if rt.correspondence is None or rt.object_sdf is None:
        pytest.skip("assets not present")
    from HoloNew.src.test_socp.interaction import frame_references
    d_obj, x_obj, d_flr, x_flr = frame_references(rt, t=0)
    M = rt.correspondence.link_idx.shape[0]
    assert d_obj.shape == (M,) and x_obj.shape == (M, 3)
    assert d_flr.shape == (M,) and x_flr.shape == (M, 3)
