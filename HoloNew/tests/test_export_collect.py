from types import SimpleNamespace

import numpy as np
import pytest

from HoloNew.evaluation.export.collect import RunSignals


def _result(T=5, **over):
    base = dict(qpos=np.zeros((T, 43)), com=None, com_ref=None,
                angular_momentum=None, angular_momentum_ref=None,
                foot_slip=None, human_flr_dist=None, human_obj_dist=None)
    base.update(over)
    return SimpleNamespace(**base)


def test_time_axis_from_fps():
    sig = RunSignals(_result(T=4), fps=10.0)
    np.testing.assert_allclose(sig.time, [0.0, 0.1, 0.2, 0.3])


def test_only_present_channels_appear():
    sig = RunSignals(_result(T=4, com=np.ones((4, 3))), fps=30.0)
    assert set(sig.channels) == {"dynamics/com/x", "dynamics/com/y", "dynamics/com/z"}


def test_no_diagnostics_gives_empty_channels():
    assert RunSignals(_result(), fps=30.0).channels == {}


def test_shorter_channel_raises():
    # foot_slip length 3 but qpos T=5 -> genuine misalignment
    with pytest.raises(ValueError):
        RunSignals(_result(T=5, foot_slip=np.zeros(3)), fps=30.0)


def test_longer_reference_channel_is_prefix_aligned():
    # com_ref carries the full source trajectory (longer than a truncated solve);
    # it is frame-aligned from frame 0, so it must be sliced to T, not rejected.
    full_ref = np.arange(8 * 3, dtype=float).reshape(8, 3)
    sig = RunSignals(_result(T=5, com_ref=full_ref), fps=30.0)
    assert sig.channels["dynamics/com_ref/x"].shape == (5,)
    np.testing.assert_allclose(sig.channels["dynamics/com_ref/x"], full_ref[:5, 0])
