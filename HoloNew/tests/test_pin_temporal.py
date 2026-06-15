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


def test_wr_term_matches_numpy():
    import cvxpy as cp
    rt, pm = _pm()
    from HoloNew.src.test_socp.temporal import build_temporal_term
    q_tm2 = pm.qpos_mj_to_q_pin(rt.q_init_full[:36])
    rng = np.random.default_rng(1)
    q_tm1 = pin.integrate(pm.model, q_tm2, 0.03 * rng.standard_normal(pm.model.nv))
    q_t0  = pin.integrate(pm.model, q_tm1, 0.03 * rng.standard_normal(pm.model.nv))
    dqa = cp.Variable(rt.nv_a); val = 0.01 * rng.standard_normal(rt.nv_a); dqa.value = val
    lam, s_q, s_V, dt = 2.0, 0.5, 0.5, 1.0 / 30.0
    term = build_temporal_term(rt, q_t0, q_tm1, q_tm2, dqa, lam, s_q, s_V, dt)
    # ground truth: v_t(dqa) = difference(q_tm1, integrate(q_t0, v_full)); accel = (v_t - v_tm1)/dt^2
    v_full = np.zeros(pm.model.nv); v_full[rt.v_a_indices] = val
    v_t = pin.difference(pm.model, q_tm1, pin.integrate(pm.model, q_t0, v_full))
    v_tm1 = pin.difference(pm.model, q_tm2, q_tm1)
    w = np.ones(pm.model.nv) / s_q**2; w[:6] = 1.0 / s_V**2     # base rows use sigma_V
    gt = lam * float(np.sum(w * ((v_t - v_tm1) / dt**2) ** 2))
    # rtol=5e-6: the term uses the first-order linearisation of pin.difference while the
    # ground truth evaluates it exactly; the O(||dqa||^2) residual sets the floor.
    np.testing.assert_allclose(float(term.value), gt, rtol=5e-6)


def test_wr_when_enabled_uses_tuned_weight_and_solve_stays_finite():
    """W^r is opt-in (activate_wr); when on it uses the tuned lambda_r and stays finite."""
    import numpy as np
    from HoloNew.examples.robot_retarget import RetargetingConfig
    from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
    from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh",
        retargeter=TestSocpRetargeterConfig(activate_wr=True)))
    assert rt.lambda_r == TestSocpRetargeterConfig().lambda_r  # tuned weight applied
    assert rt.lambda_r > 0.0
    res = rt.retarget(max_frames=6)
    assert np.all(np.isfinite(res.qpos))
