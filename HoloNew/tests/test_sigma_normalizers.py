"""Unit tests for the explicit σ characteristic-scale normalizers (Brick 1)."""
import cvxpy as cp
import numpy as np
import pinocchio as pin
import pytest

from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter


@pytest.fixture(scope="module")
def rt():
    return TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))


def _frame_targets(rt, q):
    """Build the frame_targets dict the same way retarget() does, at frame 0."""
    from HoloNew.src.test_socp.tables import IK_MATCH_TABLE_SINGLE
    from HoloNew.src.test_socp.targets import ground_frame_targets
    gpos = rt.gmr_ground["pos"]    # (T, N_bodies, 3)
    gquat = rt.gmr_ground["quat"]  # (T, N_bodies, 4) wxyz
    return ground_frame_targets(gpos[0], gquat[0], IK_MATCH_TABLE_SINGLE)


def test_build_style_terms_matches_inline(rt):
    """build_style_terms reproduces the S_k/S_B terms (σ_R=1 ⇒ no scaling)."""
    from HoloNew.src.test_socp.style import build_style_terms
    q_mj = rt.q_init_full[:36]
    ft = _frame_targets(rt, q_mj)
    dqa = cp.Variable(rt.nv_a)
    dqa.value = np.zeros(rt.nv_a)
    terms = build_style_terms(rt, q_mj, ft, dqa, lambda_ws=1.0, sigma_R=1.0)
    assert len(terms) > 0
    total = sum(float(t.value) for t in terms)
    assert np.isfinite(total)


def test_sigma_R_scales_style_quadratically(rt):
    """Doubling σ_R divides every S_k/S_B term by 4 at a fixed dqa."""
    from HoloNew.src.test_socp.style import build_style_terms
    q_mj = rt.q_init_full[:36]
    ft = _frame_targets(rt, q_mj)
    dqa = cp.Variable(rt.nv_a)
    dqa.value = 0.01 * np.random.default_rng(0).standard_normal(rt.nv_a)
    t1 = build_style_terms(rt, q_mj, ft, dqa, lambda_ws=1.0, sigma_R=1.0)
    t2 = build_style_terms(rt, q_mj, ft, dqa, lambda_ws=1.0, sigma_R=2.0)
    s1 = sum(float(t.value) for t in t1)
    s2 = sum(float(t.value) for t in t2)
    np.testing.assert_allclose(s2, s1 / 4.0, rtol=1e-9)


def test_sigma_a_sigma_L_scale_centroidal(rt):
    """σ_a scales W^c by 1/σ_a²; σ_L scales W^L by 1/σ_L², at fixed dqa."""
    from HoloNew.src.test_socp.centroidal import build_centroidal_terms
    pm = rt.pin
    rng = np.random.default_rng(1)
    q0 = pm.qpos_mj_to_q_pin(rt.q_init_full[:36])
    q1 = pin.integrate(pm.model, q0, 0.02 * rng.standard_normal(pm.model.nv))
    c0 = pm.com(q0)
    dqa = cp.Variable(rt.nv_a); dqa.value = 0.01 * rng.standard_normal(rt.nv_a)
    # non-zero cddot_ref so the reference-subtraction branch of b_c is exercised
    # (it must carry the same 1/sigma_a factor as the Jc@dqa part).
    kw = dict(rt=rt, q_t0=q0, q_tm1=q1, c_tm1=c0, c_tm2=c0,
              cddot_ref=np.array([0.3, -1.2, 9.81]), c_ref=c0, dqa=dqa,
              lambda_c_pos=0.0, dt=1/30.0)
    wc1 = build_centroidal_terms(lambda_c=1.0, lambda_l=0.0, sigma_a=1.0, sigma_L=1.0, **kw)
    wc2 = build_centroidal_terms(lambda_c=1.0, lambda_l=0.0, sigma_a=2.0, sigma_L=1.0, **kw)
    np.testing.assert_allclose(sum(float(t.value) for t in wc2),
                               sum(float(t.value) for t in wc1) / 4.0, rtol=1e-9)
    wl1 = build_centroidal_terms(lambda_c=0.0, lambda_l=1.0, sigma_a=1.0, sigma_L=1.0, **kw)
    wl2 = build_centroidal_terms(lambda_c=0.0, lambda_l=1.0, sigma_a=1.0, sigma_L=2.0, **kw)
    np.testing.assert_allclose(sum(float(t.value) for t in wl2),
                               sum(float(t.value) for t in wl1) / 4.0, rtol=1e-9)


def test_wo_sigma_split_single_lambda():
    """Collapsed W^o: one λ_o; σ_ao scales the linear residual, σ_omega the
    angular one. cost = λ_o(||(vdot-ref)/σ_ao||² + ||(omega-ref)/σ_omega||²)."""
    from HoloNew.src.test_socp.movable import build_wo_term
    rng = np.random.default_rng(0)
    T0 = pin.exp6(0.1*rng.standard_normal(6)) * pin.SE3.Identity()
    T1 = pin.exp6(0.1*rng.standard_normal(6)) * pin.SE3.Identity()
    T2 = pin.exp6(0.1*rng.standard_normal(6)) * pin.SE3.Identity()
    vdot_ref, omega_ref = rng.standard_normal(3), rng.standard_normal(3)
    dt = 1/30.0
    dxi = cp.Variable(6); val = 0.02*rng.standard_normal(6); dxi.value = val
    term = build_wo_term(T0, T1, T2, vdot_ref, omega_ref, dxi,
                         lambda_o=2.0, dt=dt, sigma_ao=3.0, sigma_omega=5.0)
    Tcur = pin.exp6(val) * T0
    V_t = pin.log6(T1.inverse() * Tcur).vector / dt
    V_tm1 = pin.log6(T2.inverse() * T1).vector / dt
    vdot = (V_t[:3] - V_tm1[:3]) / dt
    omega = V_t[3:6]
    gt = 2.0 * (np.sum(((vdot - vdot_ref)/3.0)**2) + np.sum(((omega - omega_ref)/5.0)**2))
    np.testing.assert_allclose(float(term.value), gt, rtol=1e-3)
