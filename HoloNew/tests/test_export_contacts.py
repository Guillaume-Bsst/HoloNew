"""Integration: per-point floor (+ object) contact channels from the solved trajectory.

robot_only gives the floor channel; object_interaction additionally gives the object
channel. Short real solves; needs the holonew env.
"""
from __future__ import annotations

import numpy as np
import pytest

from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.evaluation.export.contact_signals import (
    contact_channels, contact_arrays, contact_scoreboard)

MAX_FRAMES = 4
_SIGNALS = ("robot_dist", "robot_active", "robot_active_phys",
            "ref_dist", "ref_active", "ref_active_phys", "slip")


def _solve(task_type):
    cfg = RetargetingConfig(task_type=task_type, task_name="sub3_largebox_003",
                            data_format="smplh")
    rt = TestSocpRetargeter.from_config(cfg)
    res = rt.retarget(max_frames=MAX_FRAMES)
    return rt, res


@pytest.fixture(scope="module")
def floor_run():
    return _solve("robot_only")


def test_floor_channel_complete_and_aligned(floor_run):
    rt, res = floor_run
    ch = contact_channels(rt, res)
    T = res.qpos.shape[0]
    floor = [k for k in ch if k.startswith("contacts/floor/")]
    assert floor
    for sig in _SIGNALS:
        assert any(k.startswith(f"contacts/floor/{sig}/") for k in floor)
    for arr in ch.values():
        assert arr.shape == (T,)
        assert np.all(np.isfinite(arr))


def test_active_channels_are_boolean(floor_run):
    rt, res = floor_run
    ch = contact_channels(rt, res)
    for k, arr in ch.items():
        if "/robot_active/" in k or "/ref_active/" in k:
            assert set(np.unique(arr)).issubset({0.0, 1.0})


def test_robot_only_has_no_object_channel(floor_run):
    rt, res = floor_run
    ch = contact_channels(rt, res)
    assert not any(k.startswith("contacts/object/") for k in ch)


def test_physical_active_independent_of_L(floor_run):
    rt, res = floor_run
    # A tighter physical threshold can only shrink (never grow) the active set.
    a_loose, _ = contact_arrays(rt, res, threshold=0.05)
    a_tight, _ = contact_arrays(rt, res, threshold=0.01)
    assert a_tight["floor"]["robot_active_phys"].sum() <= a_loose["floor"]["robot_active_phys"].sum()


def test_contact_scoreboard_canonical_keys(floor_run):
    rt, res = floor_run
    arrays, _ = contact_arrays(rt, res)
    sb = contact_scoreboard(arrays)
    assert "floor" in sb
    for k in ("contact_precision", "contact_recall", "contact_f1",
              "contact_place_err", "contact_slip_mean"):
        assert k in sb["floor"]


def test_object_interaction_has_object_channel():
    rt, res = _solve("object_interaction")
    ch = contact_channels(rt, res)
    assert any(k.startswith("contacts/object/robot_dist/") for k in ch)
    assert any(k.startswith("contacts/floor/robot_dist/") for k in ch)
