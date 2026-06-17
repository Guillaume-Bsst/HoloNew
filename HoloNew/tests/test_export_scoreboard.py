from types import SimpleNamespace
import pytest

import numpy as np

from HoloNew.evaluation.export.context import SignalContext
from HoloNew.evaluation.export.scoreboard import compute_scoreboard


def test_compute_scoreboard_assembles_canonical_families():
    T, dof = 6, 2
    qpos = np.zeros((T, 7 + dof))
    qpos[:, 3] = 1.0                       # valid base quaternion
    qpos[:, 7] = np.arange(T) ** 2         # joint 0 moves
    com = np.cumsum(np.ones((T, 3)), axis=0) * 0.1
    res = SimpleNamespace(qpos=qpos, com=com, com_ref=com.copy(),
                          angular_momentum=np.ones((T, 3)), angular_momentum_ref=np.zeros((T, 3)))
    ctx = SignalContext(dt=0.1, dof=dof,
                        joint_limit_cols=np.array([0]), joint_limit_lower=np.array([-1.0]),
                        joint_limit_upper=np.array([1.0]))
    channels = {
        "tracking/mpjpe/A": np.ones(T), "tracking/mpjpe/B": np.full(T, 3.0),  # mean -> 2.0
        "tracking/mpjpe_root_rel/A": np.zeros(T),
        "tracking/base_track": np.full(T, 0.5),
        "tracking/orient/A": np.full(T, 0.2),
        "roots/base_pos_err": np.full(T, 0.4), "roots/base_rot_err": np.full(T, 0.1),
    }
    sb = compute_scoreboard(channels, res, ctx, contact_arrays=None)

    assert {"smoothness", "effort", "dynamics", "tracking", "style", "roots"} <= set(sb)
    assert sb["tracking"]["mpjpe_global"] == pytest.approx(2.0)   # mean of links A,B over frames
    assert sb["tracking"]["base_track_err"] == pytest.approx(0.5)
    assert sb["style"]["orient_err"] == pytest.approx(0.2)
    assert sb["roots"]["pos_err"] == pytest.approx(0.4)
    assert sb["roots"]["rot_err"] == pytest.approx(0.1)
    # joint 0 is mid-range at t=0 (q=0, limits ±1) -> margin starts at 0.5
    assert sb["effort"]["joint_limit_margin_min"] <= 0.5


def test_scoreboard_skips_absent_families():
    res = SimpleNamespace(qpos=np.zeros((4, 9)), com=None, com_ref=None)
    sb = compute_scoreboard({}, res, SignalContext(), contact_arrays=None)
    assert "dynamics" not in sb and "tracking" not in sb
