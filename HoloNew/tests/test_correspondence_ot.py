import numpy as np
from dataclasses import dataclass
from HoloNew.src.gmr_socp_v2.correspondence.ot_couple import couple

@dataclass
class _Src:
    points: np.ndarray
    seg: np.ndarray

@dataclass
class _Tgt:
    points_world: np.ndarray
    seg: np.ndarray

def test_couple_returns_valid_human_indices():
    rng = np.random.default_rng(0)
    src = _Src(points=rng.standard_normal((20, 3)).astype(np.float32), seg=np.zeros(20, np.int64))
    tgt = _Tgt(points_world=rng.standard_normal((12, 3)).astype(np.float32), seg=np.zeros(12, np.int64))
    human_idx = couple(src, tgt, reg=0.05)
    assert human_idx.shape == (12,)
    assert human_idx.min() >= 0 and human_idx.max() < 20

def test_couple_raises_on_missing_human_segment():
    import pytest
    src = _Src(points=np.zeros((5, 3), np.float32), seg=np.zeros(5, np.int64))
    tgt = _Tgt(points_world=np.zeros((3, 3), np.float32), seg=np.ones(3, np.int64))  # seg 1 absent in src
    with pytest.raises(ValueError):
        couple(src, tgt, reg=0.05)
