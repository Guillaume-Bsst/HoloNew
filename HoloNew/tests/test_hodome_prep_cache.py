"""The HODome processed-npz disk cache must be invalidated when the prep output
format changes (e.g. 22-joint -> 55-joint orientations for hand posing). A pre-change
cache (no/old version key) must be treated as stale and rebuilt, otherwise the contact
probe silently runs on the old format."""
from pathlib import Path

import numpy as np
import pytest

from HoloNew.src.data_loaders.facade import _hodome_cache_valid, ensure_hodome_processed
from HoloNew.src.data_loaders.hodome import PREP_FORMAT_VERSION


def test_cache_missing_is_invalid(tmp_path):
    assert _hodome_cache_valid(tmp_path / "nope.npz") is False


def test_cache_without_version_is_stale(tmp_path):
    # Simulate a pre-#5 cache: 22-joint orientations, no prep_version key.
    p = tmp_path / "seq.npz"
    np.savez(p, global_joint_positions=np.zeros((4, 22, 3), np.float32),
             global_joint_orientations=np.zeros((4, 22, 4), np.float32))
    assert _hodome_cache_valid(p) is False


def test_cache_with_wrong_version_is_stale(tmp_path):
    p = tmp_path / "seq.npz"
    np.savez(p, prep_version=PREP_FORMAT_VERSION - 1,
             global_joint_orientations=np.zeros((4, 55, 4), np.float32))
    assert _hodome_cache_valid(p) is False


def test_cache_with_current_version_is_valid(tmp_path):
    p = tmp_path / "seq.npz"
    np.savez(p, prep_version=PREP_FORMAT_VERSION,
             global_joint_orientations=np.zeros((4, 55, 4), np.float32))
    assert _hodome_cache_valid(p) is True


_REPO = Path(__file__).resolve().parents[5]
_HODOME_NPZ = _REPO / "data/00_raw_datasets/HODome/smplx/subject01_baseball.npz"
_SMPLX_DIR = _REPO / "data/00_raw_datasets/models/models_smplx_v1_1/models/smplx"


@pytest.mark.skipif(not (_HODOME_NPZ.exists() and _SMPLX_DIR.exists()),
                    reason="HODome data / SMPL-X model not present")
def test_stale_cache_is_rebuilt_to_55(tmp_path):
    # Plant a stale 22-joint cache at the path ensure_hodome_processed will use.
    stale = tmp_path / f"{_HODOME_NPZ.stem}.npz"
    np.savez(stale, global_joint_positions=np.zeros((4, 22, 3), np.float32),
             global_joint_orientations=np.zeros((4, 22, 4), np.float32))
    out = ensure_hodome_processed(_HODOME_NPZ, _SMPLX_DIR, cache_dir=tmp_path)
    assert out == stale
    with np.load(out) as d:
        assert int(d["prep_version"]) == PREP_FORMAT_VERSION
        assert d["global_joint_orientations"].shape[1] == 55   # hands now posed
