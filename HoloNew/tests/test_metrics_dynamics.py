import numpy as np

from HoloNew.evaluation.metrics.dynamics import compute_dynamics, dynamics_series


def test_series_shapes_and_reduce_parity():
    rng = np.random.RandomState(0)
    T, dt = 10, 0.1
    com = rng.randn(T, 3)
    ref = rng.randn(T, 3)
    L = rng.randn(T, 3)
    s = dynamics_series(com, ref, dt, L=L, L_ref=np.zeros((T, 3)))
    assert s["com_accel_err"].shape == (T - 2,)   # 2nd finite difference
    assert s["ang_momentum_mag"].shape == (T,)
    m = compute_dynamics(com, ref, dt, L=L, L_ref=np.zeros((T, 3)))
    np.testing.assert_allclose(m["com_accel_err"], np.mean(s["com_accel_err"]))
    np.testing.assert_allclose(
        m["ang_momentum_rms"], np.sqrt(np.mean(s["ang_momentum_mag"] ** 2)))


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
