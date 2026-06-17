from types import SimpleNamespace

import numpy as np
import pytest

from HoloNew.evaluation.export.producers import PRODUCERS, vec_channels, run_all, pad_to_T
from HoloNew.evaluation.export.context import SignalContext


def _fake_result(**over):
    base = dict(qpos=np.zeros((4, 43)), com=None, com_ref=None,
                angular_momentum=None, angular_momentum_ref=None,
                foot_slip=None, human_flr_dist=None, human_obj_dist=None,
                per_frame_cost=None)
    base.update(over)
    return SimpleNamespace(**base)


def test_vec_channels_expands_columns():
    arr = np.arange(6.0).reshape(3, 2)
    ch = vec_channels("dynamics/com", arr, ("x", "y"))
    assert set(ch) == {"dynamics/com/x", "dynamics/com/y"}
    np.testing.assert_allclose(ch["dynamics/com/y"], arr[:, 1])


def test_com_producer_emits_xyz_and_ref():
    res = _fake_result(com=np.ones((4, 3)), com_ref=np.zeros((4, 3)))
    ch = run_all(res)
    for leaf in ("x", "y", "z"):
        assert f"dynamics/com/{leaf}" in ch
        assert f"dynamics/com_ref/{leaf}" in ch
    assert ch["dynamics/com/x"].shape == (4,)


def test_absent_sources_emit_nothing():
    ch = run_all(_fake_result())
    assert ch == {}


def test_dist_channels_named_positionally():
    res = _fake_result(human_flr_dist=np.zeros((4, 2)))
    ch = run_all(res)
    assert "diag/human_flr_dist/probe_000" in ch
    assert "diag/human_flr_dist/probe_001" in ch


def test_foot_slip_scalar_channel():
    res = _fake_result(foot_slip=np.array([0.0, 1.0, 2.0, 3.0]))
    ch = run_all(res)
    np.testing.assert_allclose(ch["diag/foot_slip"], [0.0, 1.0, 2.0, 3.0])


def test_solver_cost_scalar_channel():
    res = _fake_result(per_frame_cost=np.array([1.0, 2.0, 3.0, 4.0]))
    ch = run_all(res)
    np.testing.assert_allclose(ch["solver/cost"], [1.0, 2.0, 3.0, 4.0])


def test_solver_cost_absent_emits_nothing():
    ch = run_all(_fake_result())
    assert not any(k.startswith("solver/") for k in ch)


def test_pad_to_T_right_aligns_with_edge_replicate():
    out = pad_to_T(np.array([5.0, 6.0, 7.0]), 5)
    # Right-aligned: real samples at the tail, leading edge replicated.
    np.testing.assert_allclose(out, [5.0, 5.0, 5.0, 6.0, 7.0])
    # Longer-than-T is truncated to the first T.
    np.testing.assert_allclose(pad_to_T(np.arange(5.0), 3), [0.0, 1.0, 2.0])


def test_smoothness_off_without_explicit_dof():
    # Default context (dof=None) must not emit smoothness (avoids object-DOF misread).
    res = _fake_result(qpos=np.zeros((6, 43)))
    assert not any(k.startswith("smoothness/") for k in run_all(res, SignalContext()))


def test_smoothness_emits_padded_per_joint_channels_with_dof():
    T, dof = 6, 2
    qpos = np.zeros((T, 7 + dof))
    qpos[:, 3] = 1.0  # identity base quaternion (wxyz), so omega is well-defined
    qpos[:, 7] = np.arange(T) ** 2  # joint 0 has non-trivial accel
    ch = run_all(_fake_result(qpos=qpos), SignalContext(dt=0.1, dof=dof))
    assert "smoothness/joint_accel/joint_000" in ch
    assert "smoothness/joint_jerk/joint_001" in ch
    assert "smoothness/base_pos_accel" in ch
    assert ch["smoothness/joint_accel/joint_000"].shape == (T,)


def test_effort_off_without_limits():
    qpos = np.zeros((6, 9))
    qpos[:, 3] = 1.0  # valid base quaternion (other producers may run)
    res = _fake_result(qpos=qpos)
    assert not any(k.startswith("effort/") for k in run_all(res, SignalContext(dt=0.1, dof=2)))


def test_effort_emits_margin_vel_saturated_with_limits():
    T, dof = 6, 2
    qpos = np.zeros((T, 7 + dof))
    qpos[:, 3] = 1.0
    qpos[:, 7] = np.linspace(-1.0, 1.0, T)  # limited joint 0 sweeps its range
    ctx = SignalContext(
        dt=0.1, dof=dof,
        joint_limit_cols=np.array([0]), joint_limit_lower=np.array([-1.0]),
        joint_limit_upper=np.array([1.0]), joint_limit_names=["left_hip"])
    ch = run_all(_fake_result(qpos=qpos), ctx)
    assert ch["effort/joint_margin/left_hip"].shape == (T,)
    assert "effort/joint_vel/left_hip" in ch
    assert "effort/saturated/left_hip" in ch
    # margin at the sweep endpoints (joint at limit) is ~0.
    np.testing.assert_allclose(ch["effort/joint_margin/left_hip"][-1], 0.0, atol=1e-9)


def test_extra_channels_emitted_verbatim():
    qpos = np.zeros((4, 9))
    qpos[:, 3] = 1.0
    ctx = SignalContext(extra_channels={"tracking/mpjpe/LeftFoot": np.arange(4.0)})
    ch = run_all(_fake_result(qpos=qpos), ctx)
    np.testing.assert_allclose(ch["tracking/mpjpe/LeftFoot"], [0, 1, 2, 3])


def test_registry_is_list_of_named_callables():
    assert PRODUCERS and all(callable(fn) for _, fn in PRODUCERS)
