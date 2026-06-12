"""Integration tests for GmrSocpRetargeter (v1 and v2)."""
import numpy as np
import pytest


@pytest.fixture(scope="module")
def demo_cfg():
    from HoloNew.config_types.retargeting import RetargetingConfig
    return RetargetingConfig(task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh")


from HoloNew.src.gmr_socp.gmr_socp import GmrSocpRetargeter
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter


@pytest.mark.parametrize("cls", [GmrSocpRetargeter, TestSocpRetargeter])
def test_gmr_runs_and_tracks_pelvis(demo_cfg, cls):
    rt = cls.from_config(demo_cfg)
    result = rt.retarget()
    assert result.qpos.shape[1] == 7 + rt.task_constants.ROBOT_DOF
    assert np.isfinite(result.qpos[:, :3]).all()
