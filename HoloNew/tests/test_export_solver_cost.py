"""Integration: a short TEST-SOCP solve populates per_frame_cost, and it flows to a
``solver/cost`` export channel. Runs a real (small) robot_only solve, so it needs the
holonew env (cvxpy / pinocchio / mujoco).
"""
from __future__ import annotations

import numpy as np
import pytest

from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.evaluation.export.collect import RunSignals

MAX_FRAMES = 4


@pytest.fixture(scope="module")
def short_result():
    cfg = RetargetingConfig(task_type="robot_only", task_name="sub3_largebox_003",
                            data_format="smplh")
    rt = TestSocpRetargeter.from_config(cfg)
    return rt.retarget(max_frames=MAX_FRAMES)


def test_per_frame_cost_is_populated(short_result):
    c = short_result.per_frame_cost
    assert c is not None
    assert c.shape == (short_result.qpos.shape[0],)
    assert np.all(np.isfinite(c))


def test_solver_cost_reaches_export_channel(short_result):
    sig = RunSignals(short_result, fps=30.0)
    assert "solver/cost" in sig.channels
    np.testing.assert_allclose(sig.channels["solver/cost"],
                               short_result.per_frame_cost[:sig.T])
