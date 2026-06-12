"""Test that GmrSocpRetargeterV2 loads the bundled correspondence."""


def test_v2_loads_bundled_correspondence():
    from HoloNew.examples.robot_retarget import RetargetingConfig
    from HoloNew.src.gmr_socp_v2.gmr_socp_v2 import GmrSocpRetargeterV2
    from HoloNew.src.gmr_socp_v2.correspondence.build_correspondence import CorrespondenceTable

    rt = GmrSocpRetargeterV2.from_config(
        RetargetingConfig(task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    assert isinstance(rt.correspondence, CorrespondenceTable)
    assert rt.correspondence.link_idx.shape[0] > 0
