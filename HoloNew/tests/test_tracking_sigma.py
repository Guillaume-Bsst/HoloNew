"""GMR world-frame pos/rot tracking: global lambda + sigma normalization.

Mirrors the σ convention of the other terms: each tracking residual becomes
lambda * w / sigma**2 * ||residual||^2, with the per-point w_p/w_r kept as the
intra-distribution. At the defaults (lambda=1, sigma=1) the effective weight is
exactly w_p/w_r, so behavior is unchanged.
"""
import cvxpy as cp
import numpy as np
import pytest

from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter


@pytest.fixture(scope="module")
def rt():
    return TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))


def _frame_targets(rt):
    from HoloNew.src.test_socp.tables import IK_MATCH_TABLE_SINGLE
    from HoloNew.src.test_socp.targets import ground_frame_targets
    gpos, gquat = rt.gmr_ground["pos"], rt.gmr_ground["quat"]
    return ground_frame_targets(gpos[0], gquat[0], IK_MATCH_TABLE_SINGLE)


def test_config_tracking_sigma_lambda_defaults():
    """Flat fields exist and default to behavior-preserving 1.0."""
    from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
    c = TestSocpRetargeterConfig()
    assert c.lambda_pos == 1.0 and c.sigma_p == 1.0
    assert c.lambda_rot == 1.0 and c.sigma_rot == 1.0


def test_tracking_pos_scaling(rt):
    """lambda_pos scales the pos terms by lambda_pos; sigma_p by 1/sigma_p^2."""
    from HoloNew.src.test_socp.tracking import build_tracking_terms
    ft = _frame_targets(rt)
    q = rt.q_init_full[:36]
    dqa = cp.Variable(rt.nv_a)
    dqa.value = 0.01 * np.random.default_rng(0).standard_normal(rt.nv_a)
    kw = dict(activate_pos=True, activate_rot=False)
    base = build_tracking_terms(rt, ft, dqa, q, 1.0, 1.0, 1.0, 1.0, **kw)
    lam2 = build_tracking_terms(rt, ft, dqa, q, 2.0, 1.0, 1.0, 1.0, **kw)
    sig2 = build_tracking_terms(rt, ft, dqa, q, 1.0, 2.0, 1.0, 1.0, **kw)
    s0 = sum(float(t.value) for t in base)
    assert len(base) > 0 and s0 > 0
    np.testing.assert_allclose(sum(float(t.value) for t in lam2), 2.0 * s0, rtol=1e-9)
    np.testing.assert_allclose(sum(float(t.value) for t in sig2), s0 / 4.0, rtol=1e-9)


def test_tracking_rot_scaling(rt):
    """lambda_rot scales the rot terms by lambda_rot; sigma_rot by 1/sigma_rot^2."""
    from HoloNew.src.test_socp.tracking import build_tracking_terms
    ft = _frame_targets(rt)
    q = rt.q_init_full[:36]
    dqa = cp.Variable(rt.nv_a)
    dqa.value = 0.01 * np.random.default_rng(1).standard_normal(rt.nv_a)
    kw = dict(activate_pos=False, activate_rot=True)
    base = build_tracking_terms(rt, ft, dqa, q, 1.0, 1.0, 1.0, 1.0, **kw)
    lam2 = build_tracking_terms(rt, ft, dqa, q, 1.0, 1.0, 2.0, 1.0, **kw)
    sig2 = build_tracking_terms(rt, ft, dqa, q, 1.0, 1.0, 1.0, 2.0, **kw)
    s0 = sum(float(t.value) for t in base)
    assert len(base) > 0 and s0 > 0
    np.testing.assert_allclose(sum(float(t.value) for t in lam2), 2.0 * s0, rtol=1e-9)
    np.testing.assert_allclose(sum(float(t.value) for t in sig2), s0 / 4.0, rtol=1e-9)


def test_tracking_defaults_equal_legacy_weight(rt):
    """At lambda=sigma=1 the pos term equals w_p*||residual||^2 exactly (legacy)."""
    from HoloNew.src.test_socp.tracking import build_tracking_terms
    ft = _frame_targets(rt)
    q = rt.q_init_full[:36]
    dqa = cp.Variable(rt.nv_a)
    dqa.value = 0.0 * np.zeros(rt.nv_a)  # at dqa=0 the term = w_p*||p_t - p_c||^2
    terms = build_tracking_terms(rt, ft, dqa, q, 1.0, 1.0, 1.0, 1.0,
                                 activate_pos=True, activate_rot=False)
    # Independent legacy ground truth.
    gt = 0.0
    for frame, (p_t, R_t, w_p, w_r) in ft.items():
        if w_p <= 0:
            continue
        body = rt.robot_link_names[frame]
        p_c = rt.body_position(q, body)
        gt += w_p * float(np.sum((p_t - p_c) ** 2))
    np.testing.assert_allclose(sum(float(t.value) for t in terms), gt, rtol=1e-9)
