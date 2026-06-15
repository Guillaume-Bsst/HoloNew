import numpy as np

from HoloNew.evaluation.metrics.dynamics import compute_dynamics


def test_matching_com_zero_accel_err():
    T = 12
    dt = 0.1
    t = np.arange(T) * dt
    com = np.stack([t, np.zeros(T), -0.5 * 9.81 * t ** 2], axis=1)  # free fall
    m = compute_dynamics(com, com.copy(), dt=dt)
    assert m["com_accel_err"] < 1e-9


def test_offset_accel_consistent_and_L_rms():
    T = 12
    dt = 0.1
    t = np.arange(T) * dt
    com = np.stack([t, 0 * t, 0 * t], axis=1)  # constant velocity -> zero accel
    ref = com.copy()
    L = np.zeros((T, 3))
    m = compute_dynamics(com, ref, dt=dt, L=L, L_ref=np.zeros((T, 3)))
    assert m["com_accel_err"] < 1e-9
    assert m["ang_momentum_rms"] < 1e-12
