import numpy as np
import pytest
from HoloNew.src.test_socp.solve.spec import (
    ResidualBlock, LinearConstraint, TrustRegion, ProblemSpec, SolveResult)


def test_residual_block_validates_columns():
    ProblemSpec(nv_a=3, n_obj=0,
                residuals=[ResidualBlock(A=np.zeros((2, 3)), c=np.zeros(2))],
                constraints=[], trust_regions=[])
    with pytest.raises(ValueError):
        ProblemSpec(nv_a=3, n_obj=0,
                    residuals=[ResidualBlock(A=np.zeros((2, 4)), c=np.zeros(2))],
                    constraints=[], trust_regions=[])


def test_obj_block_requires_n_obj():
    with pytest.raises(ValueError):
        ProblemSpec(nv_a=3, n_obj=0,
                    residuals=[ResidualBlock(A=np.zeros((2, 3)), c=np.zeros(2),
                                             A_obj=np.zeros((2, 6)))],
                    constraints=[], trust_regions=[])


def test_solve_result_fields():
    r = SolveResult(dqa=np.zeros(3), dxi=None, value=1.0, status="optimal")
    assert r.dqa.shape == (3,) and r.dxi is None and r.status == "optimal"
