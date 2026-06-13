# tests/test_pin_conversion.py
import numpy as np
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.src.test_socp.pin_model import PinModel


def _rt_pm():
    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    pm = PinModel(rt.task_constants.ROBOT_URDF_FILE)
    pm.bind_mujoco_order(rt.robot_model)
    return rt, pm


def test_qpos_roundtrip():
    rt, pm = _rt_pm()
    q_mj = rt.q_init_full[:36].copy()
    q_pin = pm.qpos_mj_to_q_pin(q_mj)
    q_mj2 = pm.q_pin_to_qpos_mj(q_pin)
    np.testing.assert_allclose(q_mj2, q_mj, atol=1e-12)


def test_quaternion_reordered():
    rt, pm = _rt_pm()
    q_mj = rt.q_init_full[:36].copy()
    q_pin = pm.qpos_mj_to_q_pin(q_mj)
    np.testing.assert_allclose(q_pin[3:7], q_mj[[4, 5, 6, 3]], atol=1e-12)
