from types import SimpleNamespace

import numpy as np
import pytest

from HoloNew.evaluation.export.producers import PRODUCERS, vec_channels, run_all


def _fake_result(**over):
    base = dict(qpos=np.zeros((4, 43)), com=None, com_ref=None,
                angular_momentum=None, angular_momentum_ref=None,
                foot_slip=None, human_flr_dist=None, human_obj_dist=None)
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


def test_registry_is_list_of_named_callables():
    assert PRODUCERS and all(callable(fn) for _, fn in PRODUCERS)
