import numpy as np

from HoloNew.evaluation.metrics.smoothness import compute_smoothness, smoothness_series


def _qpos_const_vel(T=20, dof=5):
    t = np.arange(T)[:, None]
    base = np.hstack([0.01 * t * np.ones((T, 3)), np.tile([1, 0, 0, 0], (T, 1))])
    joints = 0.02 * t * np.ones((T, dof))
    return np.hstack([base, joints])


def test_constant_velocity_has_zero_accel_and_jerk():
    m = compute_smoothness(_qpos_const_vel(), dof=5, dt=1 / 30.0)
    assert m["joint_accel_rms"] < 1e-9
    assert m["joint_jerk_rms"] < 1e-9
    assert m["base_pos_accel_rms"] < 1e-9


def test_known_joint_accel():
    # joint angle = 0.5 * a * (t*dt)^2  => second derivative a
    dt = 0.1
    a = 3.0
    T = 10
    t = np.arange(T) * dt
    q = 0.5 * a * t ** 2
    base = np.tile([0, 0, 0, 1, 0, 0, 0], (T, 1)).astype(float)
    qpos = np.hstack([base, q[:, None]])
    m = compute_smoothness(qpos, dof=1, dt=dt)
    assert abs(m["joint_accel_rms"] - a) < 1e-6


def test_series_shapes_and_reduce_parity():
    qpos = _qpos_const_vel(T=20, dof=5)
    dt = 1 / 30.0
    s = smoothness_series(qpos, dof=5, dt=dt)
    # Finite-difference natural lengths: accel = T-2, jerk = T-3.
    assert s["base_acc"].shape == (18, 3)
    assert s["base_ang_acc"].shape == (18, 3)
    assert s["joint_accel"].shape == (18, 5)
    assert s["joint_jerk"].shape == (17, 5)
    # The scoreboard scalars are exactly the reductions of the series arrays.
    m = compute_smoothness(qpos, dof=5, dt=dt)
    np.testing.assert_allclose(
        np.sqrt(np.mean(s["joint_accel"] ** 2)), m["joint_accel_rms"])
    np.testing.assert_allclose(
        np.sqrt(np.mean(s["joint_jerk"] ** 2)), m["joint_jerk_rms"])
    np.testing.assert_allclose(
        np.mean(np.abs(s["joint_jerk_nodt"])), m["joint_jerk_meanabs"])
