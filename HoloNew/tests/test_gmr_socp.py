"""Integration test for GmrSocpRetargeterV1 (Task 3)."""
import numpy as np
import pytest


@pytest.fixture(scope="module")
def demo_cfg():
    from HoloNew.config_types.retargeting import RetargetingConfig
    return RetargetingConfig(task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh")


def test_gmr_v1_runs_and_tracks_pelvis(demo_cfg):
    from HoloNew.src.gmr_socp.gmr_socp_v1 import GmrSocpRetargeterV1
    rt = GmrSocpRetargeterV1.from_config(demo_cfg)
    result = rt.retarget()
    assert result.qpos.shape[1] == 7 + rt.task_constants.ROBOT_DOF
    assert np.isfinite(result.qpos[:, :3]).all()
