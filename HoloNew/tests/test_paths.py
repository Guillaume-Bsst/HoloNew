from pathlib import Path

import pytest

from HoloNew.src import paths


def test_relative_resolved_against_repo_root(tmp_path, monkeypatch):
    y = tmp_path / "path.yaml"
    y.write_text("omomo: data/foo\nhodome: /abs/bar\n")
    monkeypatch.setattr(paths, "PATHS_YAML", y)

    assert paths.get_path("omomo") == paths.REPO_ROOT / "data/foo"  # relative -> repo root
    assert paths.get_path("hodome") == Path("/abs/bar")             # absolute -> as-is


def test_missing_key_raises(tmp_path, monkeypatch):
    y = tmp_path / "path.yaml"
    y.write_text("omomo: x\n")
    monkeypatch.setattr(paths, "PATHS_YAML", y)
    with pytest.raises(ValueError, match="hodome"):
        paths.get_path("hodome")


def test_committed_path_yaml_has_all_keys():
    cfg = paths.load_paths()  # the real committed path.yaml
    for key in ("omomo", "omomo_new", "hodome", "smplx_models", "smplh_models"):
        assert key in cfg and cfg[key]


def test_calculate_scale_factor_is_cwd_independent(tmp_path, monkeypatch):
    from HoloNew.src.holosoma.preprocess import calculate_scale_factor
    # height_dict.pkl lives under the package demo_data; calling from an unrelated cwd
    # must still resolve it (sub3 is a known OMOMO subject in the bundled dict).
    monkeypatch.chdir(tmp_path)
    scale = calculate_scale_factor("sub3_largebox_003", 1.0)
    assert scale > 0.0
