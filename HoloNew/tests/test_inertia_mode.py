"""Paper-faithful placement (the explicit ex-inertia_mode config): GMR baseline by
default, floor channel + W^c active under paper_placement_config()."""
import numpy as np
import pytest
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.tests.paper_placement import PAPER_PLACEMENT, paper_placement_config


def _robot_only(**kw):
    return TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh",
        retargeter=TestSocpRetargeterConfig(**kw)))


def test_default_is_gmr_baseline():
    # The flat default is the bare GMR objective: no Style, no centroidal, no floor
    # channel, no contact terms — added back explicitly per test.
    rt = _robot_only()
    assert rt.activate_ws is False
    assert rt.activate_centroidal is False
    assert rt.lambda_d == 0.0 and rt.lambda_x == 0.0


def test_paper_placement_fields_pass_through():
    # No hidden bundle: every field of the explicit paper-placement config reaches the
    # retargeter unchanged (the builder only validates, it never rewrites).
    rt = _robot_only(**PAPER_PLACEMENT)
    assert rt.lambda_obj_floor > 0.0   # object<->floor contact enabled
    assert rt.pelvis_anchor_weight == 0.0
    assert rt.style_pelvis_relative is True
    assert rt.activate_centroidal is True
    assert rt.lambda_c > 0 and rt.lambda_l > 0
    assert rt.correspondence is not None
    assert rt.smplx_ground_probe is not None


def test_paper_placement_robot_only_runs_finite():
    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh",
        retargeter=paper_placement_config()))
    if rt.correspondence is None:
        pytest.skip("assets not present")
    res = rt.retarget(max_frames=5)
    assert np.all(np.isfinite(res.qpos))
