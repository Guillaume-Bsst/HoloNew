from HoloNew.src.stages import (
    METHODS, ROBOT_STAGE, method_labels, robot_key_for_method, stages_for_method,
)

def test_method_labels():
    assert method_labels() == ["holosoma", "GMR-SOCP", "TEST-SOCP"]

def test_robot_keys():
    assert robot_key_for_method("holosoma") == "holosoma"
    assert robot_key_for_method("GMR-SOCP") == "gmr_socp_v1"
    assert robot_key_for_method("TEST-SOCP") == "gmr_socp_v2"

def test_stage_lists_end_with_robot():
    hs = stages_for_method("holosoma")
    assert hs == ["Original", "Grounded", "Scaled", "Mapped", ROBOT_STAGE]
    g1 = stages_for_method("GMR-SOCP")
    assert g1 == ["Original", "Grounded", "Mapped", "Scaled", "Offset", "Floor", ROBOT_STAGE]
    assert stages_for_method("TEST-SOCP") == g1
