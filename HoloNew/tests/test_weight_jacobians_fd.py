"""Finite-difference guards for the cost-term linearizations (the "weights").

Each weight builds a residual ``A @ dqa`` that linearizes some quantity f(q). These
tests check ``A @ dqa`` against the finite difference ``f(q (+) dqa) - f(q)`` for a
small active-tangent step, so a wrong Jacobian / sign / frame (the bug class that hit
the W^d interaction term) fails here at machine precision. Fast: no solve, one FD step.
"""
import numpy as np

from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter


def _rt():
    return TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))


def _perturb(rt):
    pin = rt.pin
    q = pin.qpos_mj_to_q_pin(rt.q_init_full[:36])
    dqa = np.random.RandomState(0).randn(rt.nv_a) * 1e-6
    v = np.zeros(pin.model.nv)
    v[rt.v_a_indices] = dqa
    return q, pin.integrate(q, v), dqa


def test_wc_com_jacobian_fd():
    """W^c / W^c_pos: com_jacobian must match the FD of the CoM."""
    rt = _rt()
    pin = rt.pin
    q, q2, dqa = _perturb(rt)
    Jc = pin.com_jacobian(q)[:, rt.v_a_indices]
    fd = pin.com(q2) - pin.com(q)
    assert np.max(np.abs(fd - Jc @ dqa)) < 1e-9


def test_wr_difference_jacobian_fd():
    """W^r / W^L: difference_and_jac's Jacobian must match the FD of difference()."""
    rt = _rt()
    pin = rt.pin
    q, q2, dqa = _perturb(rt)
    q1 = pin.qpos_mj_to_q_pin(rt.q_init_full[:36].copy())
    v0, Jd = pin.difference_and_jac(q1, q)
    fd = pin.difference_and_jac(q1, q2)[0] - v0
    assert np.max(np.abs(fd - Jd[:, rt.v_a_indices] @ dqa)) < 1e-9


def test_ws_pelvis_tilt_jacobian_fd():
    """W^s: pelvis_tilt_residual's A must match the FD of R_B^T z_hat."""
    rt = _rt()
    pin = rt.pin
    q, q2, dqa = _perturb(rt)
    from HoloNew.src.test_socp.style import pelvis_tilt_residual
    zhat = np.array([0.0, 0.0, 1.0])
    u = lambda qp: pin.body_rotation(qp, "pelvis").T @ zhat
    _, A = pelvis_tilt_residual(rt, rt.q_init_full.copy(), np.eye(3))
    assert np.max(np.abs((u(q2) - u(q)) - A @ dqa)) < 1e-8
