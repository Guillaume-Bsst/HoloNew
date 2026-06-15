"""Integration checks for ReferenceContext (style + root-pose scoring) on real data."""
from pathlib import Path

import numpy as np
import pytest

_NPZ = Path("demo_results/g1/robot_only/omomo/sub3_largebox_003.npz")


def _ctx():
    from HoloNew.evaluation.reference_context import ReferenceContext
    return ReferenceContext.from_config("robot_only", "sub3_largebox_003", "smplh")


def test_root_pose_sanity_finite_on_real_npz():
    if not _NPZ.exists():
        pytest.skip(f"{_NPZ} not present")
    d = np.load(_NPZ, allow_pickle=True)
    m = _ctx().score_roots(d["qpos"])
    for k in ("root_pos_err", "root_rot_err"):
        assert k in m and np.isfinite(m[k]), (k, m)
        assert m[k] >= 0.0
    # robot-only npz has no trailing object pose -> no object keys
    assert "obj_root_pos_err" not in m


def test_style_score_finite_on_real_npz():
    if not _NPZ.exists():
        pytest.skip(f"{_NPZ} not present")
    d = np.load(_NPZ, allow_pickle=True)
    m = _ctx().score_style(d["qpos"])
    for k in ("style_orient_vs_smpl", "style_shape_vs_smpl"):
        assert k in m and np.isfinite(m[k]) and m[k] >= 0.0, (k, m)
