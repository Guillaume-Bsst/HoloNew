"""Verify pinocchio FK (body_position, body_rotation) matches MuJoCo at sampled configs."""
import numpy as np
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.src.test_socp.pin_model import PinModel


def _setup():
    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    pm = PinModel(rt.task_constants.ROBOT_URDF_FILE)
    pm.bind_mujoco_order(rt.robot_model)
    return rt, pm


def test_fk_position_rotation_match_mujoco():
    rt, pm = _setup()
    rng = np.random.default_rng(0)
    body = "left_ankle_roll_link"
    for _ in range(5):
        q_mj = rt.q_init_full[:36].copy()
        q_mj[7:] += 0.1 * rng.standard_normal(29)
        p_mj = rt.body_position(q_mj, body)
        R_mj = rt.body_rotation(q_mj, body)
        q_pin = pm.qpos_mj_to_q_pin(q_mj)
        p_pin = pm.body_position(q_pin, body)
        R_pin = pm.body_rotation(q_pin, body)
        np.testing.assert_allclose(p_pin, p_mj, atol=1e-6)
        np.testing.assert_allclose(R_pin, R_mj, atol=1e-6)
