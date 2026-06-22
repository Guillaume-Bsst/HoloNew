import pytest
from HoloNew.src.data_loaders.omomo import omomo_scale_factor
from HoloNew.src.paths import get_path

_SMPLH = get_path("smplh_models")
_OMOMO = get_path("omomo")
_HAVE = _SMPLH.is_dir() and (_OMOMO / "data").is_dir()


def test_scale_fallback_when_model_missing(tmp_path):
    # No SMPL-H model dir -> fall back to robot_height/default_human_height.
    s = omomo_scale_factor("sub3_largebox_003", robot_height=1.4,
                           omomo_dir=tmp_path, smplh_model_dir=tmp_path / "nope",
                           default_human_height=1.75)
    assert s == pytest.approx(1.4 / 1.75)


@pytest.mark.skipif(not _HAVE, reason="OMOMO + SMPL-H assets not present")
def test_scale_is_betas_fk_and_positive():
    s = omomo_scale_factor("sub3_largebox_003", robot_height=1.4,
                           omomo_dir=_OMOMO, smplh_model_dir=_SMPLH)
    assert 0.5 < s < 1.2   # G1 ~1.4 m vs adult human height
