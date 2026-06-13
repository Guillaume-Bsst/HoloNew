"""Tests for PinModel point Jacobian, CoM, and CoM Jacobian (Task 6).

All Jacobians are validated against finite differences using pin.integrate
(pinocchio tangent order). CoM position is cross-validated against MuJoCo.
"""
import numpy as np
import mujoco
import pinocchio as pin
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.src.test_socp.pin_model import PinModel


def _setup():
    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    pm = PinModel(rt.task_constants.ROBOT_URDF_FILE)
    pm.bind_mujoco_order(rt.robot_model)
    return rt, pm


def _point_world(pm, q_pin, body, offset):
    R = pm.body_rotation(q_pin, body)
    p = pm.body_position(q_pin, body)
    return p + R @ offset


def test_point_jacobian_matches_finite_difference():
    rt, pm = _setup()
    body = "left_ankle_roll_link"
    offset = np.array([0.02, -0.01, 0.03])
    q_pin = pm.qpos_mj_to_q_pin(rt.q_init_full[:36].copy())
    J = pm.point_translational_jacobian(q_pin, body, offset)   # (3, nv) pinocchio v order
    p0 = _point_world(pm, q_pin, body, offset)
    eps = 1e-6
    for k in range(pm.model.nv):
        v = np.zeros(pm.model.nv)
        v[k] = eps
        q1 = pin.integrate(pm.model, q_pin, v)
        fd = (_point_world(pm, q1, body, offset) - p0) / eps
        np.testing.assert_allclose(J[:, k], fd, atol=1e-4, err_msg=f"col {k}")


def test_com_matches_mujoco():
    rt, pm = _setup()
    q_mj = rt.q_init_full[:36].copy()
    rt.robot_data.qpos[:] = q_mj
    mujoco.mj_forward(rt.robot_model, rt.robot_data)
    com_mj = rt.robot_data.subtree_com[0].copy()
    com_pin = pm.com(pm.qpos_mj_to_q_pin(q_mj))
    # NOTE: MuJoCo includes extra collision bodies (ankle spheres, pelvis contour,
    # rubber hands, etc.) absent from the URDF, giving a ~0.37 kg mass difference
    # (35.53 kg MJ vs 35.15 kg pin).  The resulting CoM offset is ~2 mm; atol=5e-3.
    np.testing.assert_allclose(com_pin, com_mj, atol=5e-3)


def test_com_jacobian_matches_finite_difference():
    rt, pm = _setup()
    q_pin = pm.qpos_mj_to_q_pin(rt.q_init_full[:36].copy())
    Jc = pm.com_jacobian(q_pin)
    c0 = pm.com(q_pin)
    eps = 1e-6
    for k in range(pm.model.nv):
        v = np.zeros(pm.model.nv)
        v[k] = eps
        fd = (pm.com(pin.integrate(pm.model, q_pin, v)) - c0) / eps
        np.testing.assert_allclose(Jc[:, k], fd, atol=1e-4, err_msg=f"col {k}")
