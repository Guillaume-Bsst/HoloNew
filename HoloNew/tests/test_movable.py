"""Test for W^o object motion regularization term (Brick 5, Task 1)."""
import numpy as np
import cvxpy as cp
import pinocchio as pin
from HoloNew.src.test_socp.movable import build_wo_term


def _rand_se3(rng, scale=0.1):
    return pin.exp6(scale * rng.standard_normal(6)) * pin.SE3.Identity()


def test_wo_term_matches_numpy():
    rng = np.random.default_rng(0)
    T0 = _rand_se3(rng)
    T1 = _rand_se3(rng)
    T2 = _rand_se3(rng)
    vdot_ref = rng.standard_normal(3)
    omega_ref = rng.standard_normal(3)
    lam_o, lam_w, dt = 2.0, 3.0, 1.0 / 30.0
    dxi = cp.Variable(6)
    val = 0.02 * rng.standard_normal(6)
    dxi.value = val
    term = build_wo_term(T0, T1, T2, vdot_ref, omega_ref, dxi, lam_o, lam_w, dt)
    # Independent numpy ground truth at val: object pose = exp6(val)*T0
    Tcur = pin.exp6(val) * T0
    V_t = pin.log6(T1.inverse() * Tcur).vector / dt       # [v; omega] at t
    V_tm1 = pin.log6(T2.inverse() * T1).vector / dt
    vdot = (V_t[:3] - V_tm1[:3]) / dt
    omega = V_t[3:6]
    gt = lam_o * float(np.sum((vdot - vdot_ref) ** 2)) + lam_w * float(np.sum((omega - omega_ref) ** 2))
    np.testing.assert_allclose(float(term.value), gt, rtol=1e-3)
