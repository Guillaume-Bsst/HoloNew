"""Tests for Brick 1 interaction D/X/P: robot-side field query, references, batched Jacobians, and D/X assembly."""
import numpy as np
import pytest
import pinocchio as pin


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


def _pm():
    """Return a fully initialised PinModel for the robot_only clip."""
    from HoloNew.examples.robot_retarget import RetargetingConfig
    from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
    from HoloNew.src.test_socp.pin_model import PinModel
    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    pm = PinModel(rt.task_constants.ROBOT_URDF_FILE)
    pm.bind_mujoco_order(rt.robot_model)
    return rt, pm


def _point_world(pm, q_pin, body, offset):
    R = pm.body_rotation(q_pin, body)
    p = pm.body_position(q_pin, body)
    return p + R @ offset


def test_point_jacobians_batched_matches_individual():
    """point_jacobians returns the same result as repeated point_translational_jacobian calls."""
    rt, pm = _pm()
    q_pin = pm.qpos_mj_to_q_pin(rt.q_init_full[:36].copy())

    # Two points on different links to exercise the cache.
    bodies = ["left_ankle_roll_link", "right_elbow_link"]
    offsets = np.array([[0.02, -0.01, 0.03], [-0.01, 0.02, -0.02]])

    batched = pm.point_jacobians(q_pin, bodies, offsets)
    assert len(batched) == 2

    for i, (body, off) in enumerate(zip(bodies, offsets)):
        expected = pm.point_translational_jacobian(q_pin, body, off)
        np.testing.assert_allclose(batched[i], expected, atol=1e-10,
                                   err_msg=f"point_jacobians[{i}] mismatch for {body}")


def test_point_jacobians_fd_agreement():
    """Finite-difference check that batched point Jacobians are correct."""
    rt, pm = _pm()
    q_pin = pm.qpos_mj_to_q_pin(rt.q_init_full[:36].copy())

    bodies = ["left_ankle_roll_link", "right_elbow_link"]
    offsets = np.array([[0.02, -0.01, 0.03], [-0.01, 0.02, -0.02]])

    batched = pm.point_jacobians(q_pin, bodies, offsets)
    eps = 1e-6

    for i, (body, off) in enumerate(zip(bodies, offsets)):
        J = batched[i]
        p0 = _point_world(pm, q_pin, body, off)
        for k in range(pm.model.nv):
            v = np.zeros(pm.model.nv)
            v[k] = eps
            q1 = pin.integrate(pm.model, q_pin, v)
            fd = (_point_world(pm, q1, body, off) - p0) / eps
            np.testing.assert_allclose(J[:, k], fd, atol=1e-4,
                                       err_msg=f"point_jacobians[{i}] col {k}")


def test_dx_terms_assemble_and_solve():
    import cvxpy as cp
    rt = _rt()
    if rt.correspondence is None or rt.object_sdf is None:
        pytest.skip("assets not present")
    from HoloNew.src.test_socp.interaction import build_dx_terms
    q_pin = rt.pin.qpos_mj_to_q_pin(rt.q_init_full[:36])
    dqa = cp.Variable(rt.nv_a)
    # Use real frame-0 object pose if available, else identity.
    obj_pose = getattr(rt, "_obj_poses_raw", None)
    obj_pose = obj_pose[0] if obj_pose is not None else np.array([1., 0, 0, 0, 0, 0, 0])
    terms = build_dx_terms(rt, q_pin, dqa, 0, obj_pose, lambda_D=1.0, lambda_X=1.0)
    assert isinstance(terms, list)
    prob = cp.Problem(cp.Minimize(cp.sum(terms) + cp.sum_squares(dqa)), [cp.SOC(0.2, dqa)])
    prob.solve(solver=cp.CLARABEL)
    assert prob.status in ("optimal", "optimal_inaccurate")
