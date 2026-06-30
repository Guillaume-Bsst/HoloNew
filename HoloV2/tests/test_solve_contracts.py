"""Contrats solve : construction valide + rejet des formes incohérentes au __post_init__."""
import numpy as np
import pytest

from src.solve.contracts import (LinearConstraint, Problem, ResidualBlock, Step, TrustRegion)


def test_problem_valid_construction():
    nv, n_obj = 5, 1
    r = ResidualBlock(A=np.zeros((3, nv)), c=np.zeros(3),
                      A_obj=np.zeros((3, n_obj * 6)), name="C-D")
    lc = LinearConstraint(A=np.zeros((2, nv)), lb=np.zeros(2), ub=None, A_obj=None, name="jl")
    tr = TrustRegion(var="dv", radius=np.full(nv, 0.1), norm=-1)
    p = Problem(nv=nv, n_obj=n_obj, residuals=(r,), constraints=(lc,), trust_regions=(tr,))
    assert p.nv == nv and len(p.residuals) == 1


def test_residual_block_bad_A_cols_raises():
    with pytest.raises(ValueError):
        Problem(nv=5, n_obj=0,
                residuals=(ResidualBlock(A=np.zeros((3, 4)), c=np.zeros(3), A_obj=None, name="x"),),
                constraints=(), trust_regions=())


def test_residual_block_A_obj_without_n_obj_raises():
    with pytest.raises(ValueError):
        Problem(nv=5, n_obj=0,
                residuals=(ResidualBlock(A=np.zeros((3, 5)), c=np.zeros(3),
                                         A_obj=np.zeros((3, 6)), name="x"),),
                constraints=(), trust_regions=())


def test_residual_block_row_mismatch_raises():
    with pytest.raises(ValueError):                       # A 3 rows, c 2 rows
        Problem(nv=5, n_obj=0,
                residuals=(ResidualBlock(A=np.zeros((3, 5)), c=np.zeros(2), A_obj=None, name="x"),),
                constraints=(), trust_regions=())


def test_trust_region_bad_var_and_radius():
    with pytest.raises(ValueError):
        TrustRegion(var="dz", radius=np.ones(3), norm=-1)
    with pytest.raises(ValueError):
        TrustRegion(var="dv", radius=np.array([-1.0, 1.0]), norm=-1)


def test_step_shapes():
    s = Step(dv=np.zeros(5), dxi=np.zeros((1, 6)), value=0.0, status="optimal")
    assert s.dv.shape == (5,) and s.dxi.shape == (1, 6)
