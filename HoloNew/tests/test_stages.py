from HoloNew.src.stages import (
    STAGE_SPECS, stage_labels, spec_for_label, key_for_label, produces_qpos,
)

def test_registry_has_native_socp_stage():
    assert "SOCP" in stage_labels()
    assert stage_labels()[0] == "Original"

def test_socp_drives_robot():
    assert produces_qpos("SOCP") is True
    assert key_for_label("SOCP") == "socp"

def test_original_is_skeleton_only():
    assert produces_qpos("Original") is False
    assert key_for_label("Original") is None

def test_lookup_roundtrip():
    for s in STAGE_SPECS:
        assert spec_for_label(s.label).key == s.key

def test_gmr_stages_present_and_drive_robots():
    from HoloNew.src.stages import STAGE_SPECS, produces_qpos, key_for_label
    labels = [s.label for s in STAGE_SPECS]
    assert "GMR-SOCP v1" in labels and "GMR-SOCP v2" in labels
    assert produces_qpos("GMR-SOCP v1") and key_for_label("GMR-SOCP v1") == "gmr_socp_v1"
    assert produces_qpos("GMR-SOCP v2") and key_for_label("GMR-SOCP v2") == "gmr_socp_v2"
