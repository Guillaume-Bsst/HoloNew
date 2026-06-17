import numpy as np

from HoloNew.evaluation.metrics.effort import compute_effort, effort_series


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


def test_series_shapes_and_reduce_parity():
    lo = np.array([-1.0, -2.0])
    hi = np.array([1.0, 2.0])
    joints = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, -2.0], [0.2, 0.3]])
    dt = 0.1
    s = effort_series(joints, lo, hi, dt)
    assert s["margin"].shape == (4, 2)       # frame-aligned, no diff
    assert s["vel"].shape == (3, 2)          # first difference, T-1
    assert s["saturated"].shape == (4, 2)
    m = compute_effort(joints, lo, hi, dt)
    assert m["joint_limit_margin_min"] == float(np.min(s["margin"]))
    assert m["joint_limit_saturation_frac"] == float(np.mean(s["saturated"]))
    np.testing.assert_allclose(
        m["joint_vel_rms"], np.sqrt(np.mean(np.square(s["vel"]))))
    np.testing.assert_allclose(m["joint_vel_peak"], np.max(np.abs(s["vel"])))
