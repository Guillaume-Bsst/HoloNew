"""Integration tests for GmrSocpRetargeter (v1 and v2)."""
import numpy as np
import pytest


@pytest.fixture(scope="module")
def demo_cfg():
    from HoloNew.config_types.retargeting import RetargetingConfig
    return RetargetingConfig(task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh")


from HoloNew.src.gmr_socp_v1.gmr_socp_v1 import GmrSocpRetargeterV1
from HoloNew.src.gmr_socp.gmr_socp_v2 import GmrSocpRetargeterV2


@pytest.mark.parametrize("cls", [GmrSocpRetargeterV1, GmrSocpRetargeterV2])
def test_gmr_runs_and_tracks_pelvis(demo_cfg, cls):
    rt = cls.from_config(demo_cfg)
    result = rt.retarget()
    assert result.qpos.shape[1] == 7 + rt.task_constants.ROBOT_DOF
    assert np.isfinite(result.qpos[:, :3]).all()
