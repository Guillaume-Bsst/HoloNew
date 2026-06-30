"""SolveConfig : défaut valide + override inline + validation ValueError."""
import pytest

from src.solve.config import SolveConfig


def test_defaults_construct():
    c = SolveConfig()
    assert c.w_pos > 0.0 and c.w_reg > 0.0
    assert c.tr_joints > 0.0 and c.tr_base_pos > 0.0
    assert c.n_iter_first >= c.n_iter_per_frame >= 1
    assert c.backend == "cvxpy"


def test_inline_override():
    c = SolveConfig(w_cd=5.0, tr_joints=0.2, n_iter_first=12)
    assert c.w_cd == 5.0 and c.tr_joints == 0.2 and c.n_iter_first == 12
    assert c.w_pos == SolveConfig().w_pos          # untouched fields keep defaults


def test_negative_weight_raises():
    with pytest.raises(ValueError):
        SolveConfig(w_pos=-1.0)


def test_nonpositive_radius_raises():
    with pytest.raises(ValueError):
        SolveConfig(tr_joints=0.0)


def test_bad_backend_raises():
    with pytest.raises(ValueError):
        SolveConfig(backend="ipopt")


def test_bad_iter_raises():
    with pytest.raises(ValueError):
        SolveConfig(n_iter_per_frame=0)
