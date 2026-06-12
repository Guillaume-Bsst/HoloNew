import pytest
from HoloNew.examples.robot_retarget import RetargetingConfig

@pytest.mark.parametrize("modpath,cls", [
    ("HoloNew.src.gmr_socp_v1.gmr_socp_v1", "GmrSocpRetargeterV1"),
    ("HoloNew.src.gmr_socp_v2.gmr_socp_v2", "GmrSocpRetargeterV2"),
])
def test_gmr_exposes_full_stage_dict(modpath, cls):
    import importlib
    Cls = getattr(importlib.import_module(modpath), cls)
    rt = Cls.from_config(RetargetingConfig(task_type="robot_only",
                                           task_name="sub3_largebox_003", data_format="smplh"))
    assert set(rt.gmr_stages) == {"mapped", "scaled", "offset", "ground"}
    for k in rt.gmr_stages:
        assert rt.gmr_stages[k]["pos"].shape[1] == 14   # 14 mapped bodies
    # the solve still uses the ground stage
    assert rt.gmr_ground is rt.gmr_stages["ground"]
