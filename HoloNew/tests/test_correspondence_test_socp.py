"""Test that TestSocpRetargeter loads the bundled correspondence."""


def test_test_socp_loads_bundled_correspondence():
    from HoloNew.examples.robot_retarget import RetargetingConfig
    from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
    from HoloNew.src.test_socp.correspondence.build_correspondence import CorrespondenceTable

    rt = TestSocpRetargeter.from_config(
        RetargetingConfig(task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    assert isinstance(rt.correspondence, CorrespondenceTable)
    assert rt.correspondence.link_idx.shape[0] > 0
