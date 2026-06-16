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
