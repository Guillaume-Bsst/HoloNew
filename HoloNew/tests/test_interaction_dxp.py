"""Tests for Task 1 of Brick 1: robot-side object/floor field query at control points."""
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
