"""Tests for PinModel.centroidal_map (A_G, the 6xnv centroidal momentum matrix).

h = A_G @ v  with  h[:3] = linear momentum, h[3:6] = angular momentum.
"""
import numpy as np
import pinocchio as pin

from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter


def _pm():
    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    return rt, rt.pin


def test_centroidal_map_matches_momentum():
    rt, pm = _pm()
    rng = np.random.default_rng(0)
    q = pm.qpos_mj_to_q_pin(rt.q_init_full[:36])
    v = 0.1 * rng.standard_normal(pm.model.nv)

    Ag = pm.centroidal_map(q)                                # (6, nv)
    assert Ag.shape == (6, pm.model.nv)

    data2 = pm.model.createData()
    h_ref = pin.computeCentroidalMomentum(pm.model, data2, q, v)   # Force: .linear / .angular
    h = Ag @ v
    np.testing.assert_allclose(h[:3], np.asarray(h_ref.linear), atol=1e-6)
    np.testing.assert_allclose(h[3:6], np.asarray(h_ref.angular), atol=1e-6)
