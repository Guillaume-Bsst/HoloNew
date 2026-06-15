import numpy as np

from HoloNew.evaluation.metrics.tracking import compute_tracking


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
