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


def test_centroidal_blocks_match_numpy():
    rt, pm = _pm()
    from HoloNew.src.test_socp.centroidal import build_centroidal_blocks
    q0 = pm.qpos_mj_to_q_pin(rt.q_init_full[:36])
    rng = np.random.default_rng(2)
    q1 = pin.integrate(pm.model, q0, 0.02*rng.standard_normal(pm.model.nv))   # t-1
    q2 = pin.integrate(pm.model, q1, 0.02*rng.standard_normal(pm.model.nv))   # current q_t0
    c_tm1 = pm.com(q1); c_tm2 = pm.com(q0)
    cddot_ref = np.array([0.0, 0.0, -9.81])
    c0_ref = pm.com(q2)
    c_ref = c0_ref + np.array([0.05, -0.03, 0.0])
    val = 0.01*rng.standard_normal(rt.nv_a)
    lam_c, lam_c_pos, lam_L, dt = 3.0, 2.0, 1.0, 1.0/30.0
    blocks = build_centroidal_blocks(rt, q2, q1, c_tm1, c_tm2, cddot_ref, c_ref,
                                     lam_c, lam_c_pos, lam_L, dt)
    assert len(blocks) > 0
    block_val = sum(float(np.sum((b.A @ val + b.c) ** 2)) for b in blocks)
    # independent numpy ground truth at val:
    c0 = pm.com(q2); Jc = pm.com_jacobian(q2)[:, rt.v_a_indices]
    cddot = (c0 + Jc@val - 2*c_tm1 + c_tm2)/dt**2
    Ag = pm.centroidal_map(q2); vrel, Jd = pm.difference_and_jac(q1, q2)
    v_full = np.zeros(pm.model.nv); v_full[rt.v_a_indices] = val
    L = (Ag @ (vrel + Jd@v_full))[3:6]
    gt = (lam_c * float(np.sum((cddot - cddot_ref)**2))
          + lam_c_pos * float(np.sum((c0 + Jc@val - c_ref)**2))
          + lam_L * float(np.sum(L**2)))
    np.testing.assert_allclose(block_val, gt, rtol=1e-6)


def test_centroidal_default_off_and_runs_on():
    import numpy as np
    from HoloNew.examples.robot_retarget import RetargetingConfig
    from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    assert rt.activate_centroidal is False  # default off
    rt.activate_centroidal = True
    rt.lambda_c = 1.0
    rt.lambda_c_pos = 2.0
    rt.lambda_l = 0.1
    res = rt.retarget(max_frames=8)
    assert np.all(np.isfinite(res.qpos))
