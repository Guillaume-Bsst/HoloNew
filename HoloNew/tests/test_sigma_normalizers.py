"""Unit tests for the explicit σ characteristic-scale normalizers (Brick 1)."""
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
    gpos = rt.gmr_floor["pos"]    # (T, N_bodies, 3)
    gquat = rt.gmr_floor["quat"]  # (T, N_bodies, 4) wxyz
    return ground_frame_targets(gpos[0], gquat[0], IK_MATCH_TABLE_SINGLE)


def test_build_style_blocks_returns_finite(rt):
    """build_style_blocks returns non-empty finite blocks (σ_R=1 ⇒ no scaling)."""
    from HoloNew.src.test_socp.style import build_style_blocks
    q_mj = rt.q_init_full[:36]
    ft = _frame_targets(rt, q_mj)
    blocks = build_style_blocks(rt, q_mj, ft, lambda_ws=1.0, sigma_R=1.0)
    assert len(blocks) > 0
    for b in blocks:
        assert np.all(np.isfinite(b.A)) and np.all(np.isfinite(b.c))


def test_sigma_R_scales_style_blocks_quadratically(rt):
    """Doubling σ_R divides every S_k/S_B block cost by 4 at dqa=0."""
    from HoloNew.src.test_socp.style import build_style_blocks
    q_mj = rt.q_init_full[:36]
    ft = _frame_targets(rt, q_mj)
    b1 = build_style_blocks(rt, q_mj, ft, lambda_ws=1.0, sigma_R=1.0)
    b2 = build_style_blocks(rt, q_mj, ft, lambda_ws=1.0, sigma_R=2.0)
    s1 = sum(float(np.sum(b.c ** 2)) for b in b1)
    s2 = sum(float(np.sum(b.c ** 2)) for b in b2)
    np.testing.assert_allclose(s2, s1 / 4.0, rtol=1e-9)


def test_sigma_a_sigma_L_scale_centroidal_blocks(rt):
    """σ_a scales W^c block by 1/σ_a²; σ_L scales W^L block by 1/σ_L², at dqa=0."""
    from HoloNew.src.test_socp.centroidal import build_centroidal_blocks
    pm = rt.pin
    rng = np.random.default_rng(1)
    q0 = pm.qpos_mj_to_q_pin(rt.q_init_full[:36])
    q1 = pin.integrate(pm.model, q0, 0.02 * rng.standard_normal(pm.model.nv))
    c0 = pm.com(q0)
    kw = dict(rt=rt, q_t0=q0, q_tm1=q1, c_tm1=c0, c_tm2=c0,
              cddot_ref=np.array([0.3, -1.2, 9.81]), c_ref=c0,
              lambda_c_pos=0.0, dt=1/30.0)
    wc1 = build_centroidal_blocks(lambda_c=1.0, lambda_l=0.0, sigma_a=1.0, sigma_L=1.0, **kw)
    wc2 = build_centroidal_blocks(lambda_c=1.0, lambda_l=0.0, sigma_a=2.0, sigma_L=1.0, **kw)
    np.testing.assert_allclose(
        sum(float(np.sum(b.c ** 2)) for b in wc2),
        sum(float(np.sum(b.c ** 2)) for b in wc1) / 4.0, rtol=1e-9)
    wl1 = build_centroidal_blocks(lambda_c=0.0, lambda_l=1.0, sigma_a=1.0, sigma_L=1.0, **kw)
    wl2 = build_centroidal_blocks(lambda_c=0.0, lambda_l=1.0, sigma_a=1.0, sigma_L=2.0, **kw)
    np.testing.assert_allclose(
        sum(float(np.sum(b.c ** 2)) for b in wl2),
        sum(float(np.sum(b.c ** 2)) for b in wl1) / 4.0, rtol=1e-9)


def test_wo_sigma_split_single_lambda():
    """Collapsed W^o: one λ_o; σ_ao scales the linear block, σ_omega the angular one."""
    from HoloNew.src.test_socp.movable import build_wo_block
    rng = np.random.default_rng(0)
    T0 = pin.exp6(0.1*rng.standard_normal(6)) * pin.SE3.Identity()
    T1 = pin.exp6(0.1*rng.standard_normal(6)) * pin.SE3.Identity()
    T2 = pin.exp6(0.1*rng.standard_normal(6)) * pin.SE3.Identity()
    vdot_ref, omega_ref = rng.standard_normal(3), rng.standard_normal(3)
    dt = 1/30.0
    val = 0.02*rng.standard_normal(6)
    # nv_a=0: W^o blocks have A=zeros, so cost = ||c||^2 (robot DOF irrelevant here)
    blocks = build_wo_block(T0, T1, T2, vdot_ref, omega_ref, nv_a=0,
                            lambda_o=2.0, dt=dt, sigma_ao=3.0, sigma_omega=5.0)
    # Evaluate at dxi=val: cost = ||A_obj @ val + c||^2 summed over both blocks
    block_val = sum(float(np.sum((b.A_obj @ val + b.c) ** 2)) for b in blocks)
    Tcur = pin.exp6(val) * T0
    V_t = pin.log6(T1.inverse() * Tcur).vector / dt
    V_tm1 = pin.log6(T2.inverse() * T1).vector / dt
    vdot = (V_t[:3] - V_tm1[:3]) / dt
    omega = V_t[3:6]
    gt = 2.0 * (np.sum(((vdot - vdot_ref)/3.0)**2) + np.sum(((omega - omega_ref)/5.0)**2))
    np.testing.assert_allclose(block_val, gt, rtol=1e-3)


def test_p_scale_helper():
    from HoloNew.src.test_socp.interaction import _p_scale_sq
    np.testing.assert_allclose(_p_scale_sq(4.0, 0.05, 1/30.0),
                               4.0 / (0.05 * (1/30.0))**2, rtol=1e-12)


def test_config_sigma_defaults_present():
    from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
    c = TestSocpRetargeterConfig()
    assert c.sigma_R == 0.2 and c.sigma_a == 9.81 and c.sigma_L == 10.0
    assert c.sigma_ao == 9.81
    assert abs(c.sigma_omega - 2*np.pi) < 1e-9
    assert not hasattr(c, "lambda_omega")  # collapsed


def test_self_collision_margin_surfaced():
    """ε is a flat config field (Brick 4) defaulting to the SelfCollisionConfig value."""
    from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
    c = TestSocpRetargeterConfig()
    assert hasattr(c, "self_collision_margin")
    assert isinstance(c.self_collision_margin, float)
    assert c.self_collision_margin == 0.02


def test_self_collision_margin_feeds_tolerance(rt):
    """The flat ε reaches the solver's self-collision tolerance via the builder."""
    # The module rt was built from the default config (self_collision_margin=0.02);
    # the builder fed it into SelfCollisionConfig.tolerance -> _self_collision_tolerance.
    assert rt._self_collision_tolerance == 0.02


def test_L_floor_object_fields_and_attrs(rt):
    """Per-entity field ranges (Brick 3): config fields + resolved rt attrs."""
    from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
    c = TestSocpRetargeterConfig()
    assert hasattr(c, "L_floor") and hasattr(c, "L_object")
    # AUTO defaults (None) => both resolve to the shared probe margin on the rt.
    assert c.L_floor is None and c.L_object is None
    assert hasattr(rt, "L_floor") and hasattr(rt, "L_object")
    assert rt.L_floor > 0 and rt.L_object > 0
    # default (AUTO) reproduces the shared margin exactly
    m = rt.smplx_ground_probe.margin
    assert rt.L_floor == m and rt.L_object == m


def test_lambda_cv_sigma_cv_scale_wcvel_blocks(rt):
    """W^c_vel blocks scale by lambda_cv and 1/sigma_cv^2 at dqa=0."""
    from HoloNew.src.test_socp.centroidal import build_centroidal_blocks
    pm = rt.pin
    rng = np.random.default_rng(3)
    q0 = pm.qpos_mj_to_q_pin(rt.q_init_full[:36])
    q1 = pin.integrate(pm.model, q0, 0.02 * rng.standard_normal(pm.model.nv))
    c0 = pm.com(q0)
    kw = dict(rt=rt, q_t0=q0, q_tm1=q1, c_tm1=c0, c_tm2=c0, cddot_ref=np.zeros(3),
              c_ref=c0, lambda_c=0.0, lambda_c_pos=0.0, lambda_l=0.0, dt=1 / 30.0,
              cdot_ref=np.array([0.1, -0.2, 0.05]))
    b1 = build_centroidal_blocks(lambda_cv=1.0, sigma_cv=1.0, **kw)
    b2 = build_centroidal_blocks(lambda_cv=2.0, sigma_cv=1.0, **kw)
    b3 = build_centroidal_blocks(lambda_cv=1.0, sigma_cv=2.0, **kw)
    s1 = sum(float(np.sum(b.c ** 2)) for b in b1)
    assert s1 > 0
    np.testing.assert_allclose(sum(float(np.sum(b.c ** 2)) for b in b2), 2.0 * s1, rtol=1e-9)
    np.testing.assert_allclose(sum(float(np.sum(b.c ** 2)) for b in b3), s1 / 4.0, rtol=1e-9)
