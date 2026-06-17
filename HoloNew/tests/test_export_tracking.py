"""Integration: per-link tracking + orientation channels from a ReferenceContext.

Runs a short real robot_only solve, builds the reference context, and checks the
tracking channels are frame-aligned and finite. Needs the holonew env.
"""
from __future__ import annotations

import numpy as np
import pytest

from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.evaluation.reference_context import ReferenceContext
from HoloNew.evaluation.export.reference_signals import tracking_channels, roots_channels

MAX_FRAMES = 4


@pytest.fixture(scope="module")
def rt_and_res():
    cfg = RetargetingConfig(task_type="robot_only", task_name="sub3_largebox_003",
                            data_format="smplh")
    rt = TestSocpRetargeter.from_config(cfg)
    res = rt.retarget(max_frames=MAX_FRAMES)
    return rt, res


def test_tracking_channels_shapes_and_naming(rt_and_res):
    rt, res = rt_and_res
    ref_ctx = ReferenceContext.from_rt(rt)
    ch = tracking_channels(ref_ctx, res.qpos)

    T = min(res.qpos.shape[0], ref_ctx._gpos.shape[0])
    assert "tracking/base_track" in ch
    mpjpe = [k for k in ch if k.startswith("tracking/mpjpe/")]
    orient = [k for k in ch if k.startswith("tracking/orient/")]
    assert mpjpe and orient                       # per-link position + orientation
    for arr in ch.values():
        assert arr.shape == (T,)
        assert np.all(np.isfinite(arr))


def test_orient_only_for_tracked_links(rt_and_res):
    rt, res = rt_and_res
    ref_ctx = ReferenceContext.from_rt(rt)
    ch = tracking_channels(ref_ctx, res.qpos)
    n_tracked = int(np.sum(np.asarray(ref_ctx.tracked, dtype=bool)))
    assert len([k for k in ch if k.startswith("tracking/orient/")]) == n_tracked


def test_roots_channels_base_pose_error(rt_and_res):
    rt, res = rt_and_res
    ref_ctx = ReferenceContext.from_rt(rt)
    ch = roots_channels(ref_ctx, res.qpos)
    T = min(res.qpos.shape[0], ref_ctx._gpos.shape[0])
    assert set(ch) == {"roots/base_pos_err", "roots/base_rot_err"}
    for arr in ch.values():
        assert arr.shape == (T,)
        assert np.all(np.isfinite(arr)) and np.all(arr >= 0.0)
