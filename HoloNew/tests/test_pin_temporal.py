import numpy as np
import pinocchio as pin

from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter


def _pm():
    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    return rt, rt.pin


def test_difference_and_jacobian_consistent():
    rt, pm = _pm()
    rng = np.random.default_rng(0)
    q0 = pm.qpos_mj_to_q_pin(rt.q_init_full[:36])
    q1 = pin.integrate(pm.model, q0, 0.05 * rng.standard_normal(pm.model.nv))
    v, J = pm.difference_and_jac(q0, q1)
    np.testing.assert_allclose(pin.difference(pm.model, q0, q1), v, atol=1e-10)
    eps = 1e-6
    for k in range(pm.model.nv):
        d = np.zeros(pm.model.nv); d[k] = eps
        v2 = pin.difference(pm.model, q0, pin.integrate(pm.model, q1, d))
        np.testing.assert_allclose((v2 - v) / eps, J[:, k], atol=1e-4, err_msg=f"col {k}")
