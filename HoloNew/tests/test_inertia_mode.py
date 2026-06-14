"""Inertia mode: parity preserved when off; floor channel + W^c active when on."""
import numpy as np
import pytest
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter


def _robot_only(**kw):
    return TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh",
        retargeter=TestSocpRetargeterConfig(**kw)))


def test_parity_mode_default_unchanged():
    rt = _robot_only()
    assert rt.activate_centroidal is False
    assert rt.pelvis_anchor_weight == 10.0
    assert getattr(rt, "floor_as_entity", False) is False


def test_inertia_mode_applies_bundle():
    rt = _robot_only(inertia_mode=True)
    assert rt.floor_as_entity is True
    assert rt.pelvis_anchor_weight == 0.0
    assert rt.lambda_c_pos == 0.0
    assert rt.activate_centroidal is True
    assert rt.lambda_c > 0 and rt.lambda_L > 0
    assert rt.correspondence is not None
    assert rt.smplx_ground_probe is not None


def test_inertia_mode_robot_only_runs_finite():
    rt = _robot_only(inertia_mode=True)
    if rt.correspondence is None:
        pytest.skip("assets not present")
    res = rt.retarget(max_frames=5)
    assert np.all(np.isfinite(res.qpos))
