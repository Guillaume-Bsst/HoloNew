"""Regression: TEST-SOCP robot Z placement must match GMR-native (base) scaling.

Bug: the builder resolved scale_z_robot=None -> smpl_scale (ROBOT_HEIGHT/human_height),
bypassing scale()'s native None->base branch (base = HUMAN_SCALE_TABLE[pelvis]*ratio).
That scaled the robot's z by the wrong factor (~8% off, ~7 cm at the pelvis), so the
scale step no longer matched GMR on the z axis. The faithful gmr_socp builder passes
None straight through; test_socp must do the same.
"""
import numpy as np
import pytest

from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.src.test_socp.preprocess import compute_stages


@pytest.fixture(scope="module")
def rt():
    return TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))


def test_robot_z_matches_gmr_native_scaling(rt):
    """Default config (scale_z_robot=None) must use GMR-native base z scaling
    (compute_stages scale_z=None), not smpl_scale. Recompute the native ground
    from the same stored inputs (scale_xy=1.0 = the robot XACT default) and
    compare the z column of the ground stage."""
    native = compute_stages(rt.gmr_grounded, rt.human_quat, scale_xy=1.0, scale_z=None)
    z_rt = rt.gmr_ground["pos"][..., 2]
    z_native = native["ground"]["pos"][..., 2]
    np.testing.assert_allclose(z_rt, z_native, atol=1e-9)
