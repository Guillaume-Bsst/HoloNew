import pytest

from HoloNew.examples.view_stages import resolve_captured_obj_scale


def test_missing_obj_scale_raises(tmp_path):
    with pytest.raises(ValueError, match="obj_scale"):
        resolve_captured_obj_scale(tmp_path, "sub3_unknownobj_000")
