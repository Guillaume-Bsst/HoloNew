from pathlib import Path

import pytest

from src import paths


def _write(tmp_path: Path) -> Path:
    p = tmp_path / "paths.toml"
    p.write_text(
        'smplx = "/models/smplx"\n'
        "[roots]\n"
        'hodome = "/data/HODome"\n'
    )
    return p


def test_load_paths_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        paths.load_paths(tmp_path / "nope.toml")


def test_load_paths_roundtrip(tmp_path):
    cfg = paths.load_paths(_write(tmp_path))
    assert cfg["smplx"] == "/models/smplx"
    assert cfg["roots"]["hodome"] == "/data/HODome"


def test_smplx_dir(tmp_path):
    cfg = paths.load_paths(_write(tmp_path))
    assert paths.smplx_dir(cfg) == Path("/models/smplx")


def test_smplx_dir_missing_key(tmp_path):
    p = tmp_path / "paths.toml"
    p.write_text('[roots]\nhodome = "/data/HODome"\n')
    with pytest.raises(ValueError):
        paths.smplx_dir(path=p)


def test_dataset_root(tmp_path):
    cfg = paths.load_paths(_write(tmp_path))
    assert paths.dataset_root("hodome", cfg) == Path("/data/HODome")


def test_dataset_root_missing_key(tmp_path):
    cfg = paths.load_paths(_write(tmp_path))
    with pytest.raises(ValueError):
        paths.dataset_root("omomo", cfg)


def test_resolve_motion_absolute_passthrough(tmp_path):
    cfg = paths.load_paths(_write(tmp_path))
    absolute = Path("/somewhere/seq.npz")
    assert paths.resolve_motion("hodome", absolute, cfg) == absolute


def test_resolve_motion_relative_joins_root(tmp_path):
    cfg = paths.load_paths(_write(tmp_path))
    assert paths.resolve_motion("hodome", "smplx/s01.npz", cfg) == Path("/data/HODome/smplx/s01.npz")
