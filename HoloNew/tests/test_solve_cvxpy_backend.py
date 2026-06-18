import numpy as np
from HoloNew.src.test_socp.solve.spec import (
    ResidualBlock, LinearConstraint, TrustRegion, ProblemSpec)
from HoloNew.src.test_socp.solve.backend import make_backend


def test_unconstrained_least_squares():
    t = np.array([0.1, -0.05, 0.2])
    spec = ProblemSpec(nv_a=3, n_obj=0,
                       residuals=[ResidualBlock(A=np.eye(3), c=-t, name="track")],
                       constraints=[], trust_regions=[TrustRegion("dqa", 1.0)])
    r = make_backend("cvxpy").solve(spec)
    np.testing.assert_allclose(r.dqa, t, atol=1e-6)


def test_box_constraint_clips():
    t = np.array([1.0])
    spec = ProblemSpec(nv_a=1, n_obj=0,
                       residuals=[ResidualBlock(A=np.eye(1), c=-t)],
                       constraints=[LinearConstraint(A=np.eye(1),
                                                     lb=np.array([-0.3]), ub=np.array([0.3]))],
                       trust_regions=[TrustRegion("dqa", 10.0)])
    r = make_backend("cvxpy").solve(spec)
    np.testing.assert_allclose(r.dqa, [0.3], atol=1e-6)


def test_trust_region_bounds_step():
    t = np.array([5.0, 0.0])
    spec = ProblemSpec(nv_a=2, n_obj=0,
                       residuals=[ResidualBlock(A=np.eye(2), c=-t)],
                       constraints=[], trust_regions=[TrustRegion("dqa", 1.0)])
    r = make_backend("cvxpy").solve(spec)
    assert np.linalg.norm(r.dqa) <= 1.0 + 1e-6


def test_object_variable_block():
    spec = ProblemSpec(nv_a=1, n_obj=6,
                       residuals=[ResidualBlock(A=np.ones((1, 1)), c=np.array([-1.0]),
                                                A_obj=np.zeros((1, 6)))],
                       constraints=[], trust_regions=[TrustRegion("dqa", 5.0),
                                                      TrustRegion("dxi", 5.0)])
    r = make_backend("cvxpy").solve(spec)
    assert r.dxi is not None and r.dxi.shape == (6,)
    np.testing.assert_allclose(r.dqa, [1.0], atol=1e-6)
