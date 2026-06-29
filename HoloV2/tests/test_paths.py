from pathlib import Path

import pytest

from src import paths


def _write(tmp_path: Path) -> Path:
    p = tmp_path / "paths.toml"
    p.write_text(
        "[models]\n"
        'smplx = "/models/smplx"\n'
        'smplh = "/models/smplh"\n'
        'smpl2smplx = "/models/mt/deftrafo.pkl"\n'
        "[datasets.hodome]\n"
        'motion = "/data/HODome"\n'
        "[datasets.omomo]\n"
        'motion = "/data/OMOMO_new"\n'
        'meta = "/data/OMOMO"\n'
    )
    return p


def test_load_paths_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        paths.load_paths(tmp_path / "nope.toml")


def test_load_paths_roundtrip(tmp_path):
    cfg = paths.load_paths(_write(tmp_path))
    assert cfg["models"]["smplx"] == "/models/smplx"
    assert cfg["datasets"]["omomo"]["meta"] == "/data/OMOMO"


def test_smplx_dir(tmp_path):
    cfg = paths.load_paths(_write(tmp_path))
    assert paths.smplx_dir(cfg) == Path("/models/smplx")


def test_smplx_dir_missing(tmp_path):
    p = tmp_path / "paths.toml"
    p.write_text('[datasets.hodome]\nmotion = "/data/HODome"\n')
    with pytest.raises(ValueError):
        paths.smplx_dir(path=p)


def test_smplh_dir_and_pkl(tmp_path):
    cfg = paths.load_paths(_write(tmp_path))
    assert paths.smplh_dir(cfg) == Path("/models/smplh")
    assert paths.smpl2smplx_pkl(cfg) == Path("/models/mt/deftrafo.pkl")


def test_smplh_dir_and_pkl_optional_none(tmp_path):
    p = tmp_path / "paths.toml"
    p.write_text('[models]\nsmplx = "/models/smplx"\n')
    cfg = paths.load_paths(p)
    assert paths.smplh_dir(cfg) is None
    assert paths.smpl2smplx_pkl(cfg) is None


def test_dataset_motion_root(tmp_path):
    cfg = paths.load_paths(_write(tmp_path))
    assert paths.dataset_motion_root("hodome", cfg) == Path("/data/HODome")


def test_dataset_motion_root_missing(tmp_path):
    cfg = paths.load_paths(_write(tmp_path))
    with pytest.raises(ValueError):
        paths.dataset_motion_root("sfu", cfg)


def test_dataset_meta_defaults_to_motion(tmp_path):
    cfg = paths.load_paths(_write(tmp_path))
    assert paths.dataset_meta_root("hodome", cfg) == Path("/data/HODome")   # meta omitted => motion


def test_dataset_meta_explicit(tmp_path):
    cfg = paths.load_paths(_write(tmp_path))
    assert paths.dataset_meta_root("omomo", cfg) == Path("/data/OMOMO")


def test_dataset_meta_absent_is_none(tmp_path):
    cfg = paths.load_paths(_write(tmp_path))
    assert paths.dataset_meta_root("sfu", cfg) is None


def test_resolve_motion_absolute_passthrough(tmp_path):
    cfg = paths.load_paths(_write(tmp_path))
    absolute = Path("/somewhere/seq.npz")
    assert paths.resolve_motion("hodome", absolute, cfg) == absolute


def test_resolve_motion_relative_joins_motion_root(tmp_path):
    cfg = paths.load_paths(_write(tmp_path))
    assert paths.resolve_motion("omomo", "sub10_x_000.pt", cfg) == Path("/data/OMOMO_new/sub10_x_000.pt")
