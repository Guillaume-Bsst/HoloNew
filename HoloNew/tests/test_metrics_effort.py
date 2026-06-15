import numpy as np

from HoloNew.evaluation.metrics.effort import compute_effort


def test_at_limit_margin_zero_saturated():
    lo = np.array([-1.0, -1.0])
    hi = np.array([1.0, 1.0])
    joints = np.array([[1.0, 0.0], [0.0, 0.0]])  # joint0 at upper limit in frame 0
    m = compute_effort(joints, lo, hi, dt=0.1)
    assert m["joint_limit_margin_min"] <= 0.0 + 1e-9
    assert m["joint_limit_saturation_frac"] > 0.0


def test_midrange_margin_half():
    lo = np.array([-1.0])
    hi = np.array([1.0])
    joints = np.zeros((5, 1))  # always mid-range
    m = compute_effort(joints, lo, hi, dt=0.1)
    assert abs(m["joint_limit_margin_min"] - 0.5) < 1e-9
    assert m["joint_vel_rms"] < 1e-12
