"""Finite-difference validation of frame translational Jacobian in pinocchio tangent space.

The Jacobian is validated self-consistently: FD uses pin.integrate (the same
integration operator the solver will use), so the check is convention-correct
by construction without any reference to MuJoCo velocity ordering.
"""
import numpy as np
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


def test_frame_translational_jacobian_matches_finite_difference():
    rt, pm = _setup()
    body = "left_ankle_roll_link"
    q_mj = rt.q_init_full[:36].copy()
    # Normalize so that computeJointJacobians and forwardKinematics operate on
    # the same configuration (computeJointJacobians normalizes unit quaternions
    # internally; without this, a tiny constant offset propagates to all columns).
    q_pin = pin.normalize(pm.model, pm.qpos_mj_to_q_pin(q_mj))
    J = pm.frame_translational_jacobian(q_pin, body)   # (3, nv) pinocchio v order
    p0 = pm.body_position(q_pin, body)
    eps = 1e-6
    nv = pm.model.nv
    for k in range(nv):
        v = np.zeros(nv)
        v[k] = eps
        q1 = pin.integrate(pm.model, q_pin, v)
        fd = (pm.body_position(q1, body) - p0) / eps
        np.testing.assert_allclose(J[:, k], fd, atol=1e-4,
                                   err_msg=f"column {k} mismatch")
