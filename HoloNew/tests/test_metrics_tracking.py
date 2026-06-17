import numpy as np

from HoloNew.evaluation.metrics.tracking import compute_tracking, tracking_series


def test_identical_zero_error():
    kp = np.random.RandomState(0).randn(8, 5, 3)
    m = compute_tracking(kp, kp.copy(), root_idx=0)
    assert m["mpjpe_global"] < 1e-12
    assert m["mpjpe_root_rel"] < 1e-12


def test_constant_offset():
    kp = np.random.RandomState(1).randn(8, 5, 3)
    d = np.array([0.0, 0.0, 0.3])
    m = compute_tracking(kp + d, kp, root_idx=0)
    assert abs(m["mpjpe_global"] - 0.3) < 1e-9
    assert m["mpjpe_root_rel"] < 1e-9  # offset cancels under root subtraction


def test_series_shapes_and_reduce_parity():
    rng = np.random.RandomState(2)
    robot = rng.randn(8, 5, 3)
    ref = rng.randn(8, 5, 3)
    base = rng.randn(8, 3)
    ref_root = rng.randn(8, 3)
    s = tracking_series(robot, ref, root_idx=0, base_xyz=base, ref_root_xyz=ref_root)
    assert s["mpjpe"].shape == (8, 5)          # per-frame, per-keypoint
    assert s["mpjpe_root_rel"].shape == (8, 5)
    assert s["base_track"].shape == (8,)
    m = compute_tracking(robot, ref, root_idx=0, base_xyz=base, ref_root_xyz=ref_root)
    np.testing.assert_allclose(m["mpjpe_global"], np.mean(s["mpjpe"]))
    np.testing.assert_allclose(m["mpjpe_root_rel"], np.mean(s["mpjpe_root_rel"]))
    np.testing.assert_allclose(m["base_track_err"], np.mean(s["base_track"]))
