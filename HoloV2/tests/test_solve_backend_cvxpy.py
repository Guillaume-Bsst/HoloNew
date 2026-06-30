"""CvxpyBackend : un Problem connu -> Step. Box trust-region (QP, route OSQP)."""
import numpy as np
import pytest

from src.solve.contracts import LinearConstraint, Problem, ResidualBlock, TrustRegion
from src.solve.backend.cvxpy import CvxpyBackend


def test_box_trust_region_clamps_to_optimum():
    # min ‖dv - [1,2,3]‖²  s.c. |dv| ≤ 0.5  ->  dv = [0.5, 0.5, 0.5] (chaque coord saturée).
    nv = 3
    r = ResidualBlock(A=np.eye(nv), c=-np.array([1.0, 2.0, 3.0]), A_obj=None, name="track")
    tr = TrustRegion(var="dv", radius=np.full(nv, 0.5), norm=-1)
    p = Problem(nv=nv, n_obj=0, residuals=(r,), constraints=(), trust_regions=(tr,))
    step = CvxpyBackend().solve(p)
    assert step.status in ("optimal", "optimal_inaccurate")
    assert np.allclose(step.dv, [0.5, 0.5, 0.5], atol=1e-4)
    assert step.dxi is None


def test_linear_constraint_active():
    # min ‖dv‖²  s.c. dv[0] ≥ 1 (one-sided lb) -> dv = [1, 0].
    nv = 2
    r = ResidualBlock(A=np.eye(nv), c=np.zeros(nv), A_obj=None, name="reg")
    lc = LinearConstraint(A=np.array([[1.0, 0.0]]), lb=np.array([1.0]), ub=None, A_obj=None, name="c")
    tr = TrustRegion(var="dv", radius=np.full(nv, 10.0), norm=-1)
    p = Problem(nv=nv, n_obj=0, residuals=(r,), constraints=(lc,), trust_regions=(tr,))
    step = CvxpyBackend().solve(p)
    assert np.allclose(step.dv, [1.0, 0.0], atol=1e-4)


def test_object_coupling_dxi():
    # min ‖dv - 1‖² + ‖dxi - 2‖²  s.c. box 0.5 -> dv=0.5, dxi=0.5 ; dxi shape (1,6).
    nv, n_obj = 1, 1
    r1 = ResidualBlock(A=np.ones((1, nv)), c=-np.ones(1), A_obj=None, name="dv")
    r2 = ResidualBlock(A=np.zeros((6, nv)), c=-2.0 * np.ones(6),
                       A_obj=np.eye(6), name="dxi")          # ‖dxi - 2‖²
    trv = TrustRegion(var="dv", radius=np.full(nv, 0.5), norm=-1)
    trx = TrustRegion(var="dxi", radius=np.full(n_obj * 6, 0.5), norm=-1)
    p = Problem(nv=nv, n_obj=n_obj, residuals=(r1, r2), constraints=(), trust_regions=(trv, trx))
    step = CvxpyBackend().solve(p)
    assert step.dxi.shape == (1, 6)
    assert np.allclose(step.dv, [0.5], atol=1e-4) and np.allclose(step.dxi, 0.5, atol=1e-4)


def test_l2_norm_not_implemented():
    nv = 2
    p = Problem(nv=nv, n_obj=0,
                residuals=(ResidualBlock(A=np.eye(nv), c=np.zeros(nv), A_obj=None, name="r"),),
                constraints=(), trust_regions=(TrustRegion(var="dv", radius=np.ones(nv), norm=2),))
    with pytest.raises(NotImplementedError):
        CvxpyBackend().solve(p)
