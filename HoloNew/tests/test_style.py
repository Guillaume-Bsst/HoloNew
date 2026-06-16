import numpy as np
import pinocchio as pin
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.src.test_socp.style import pelvis_tilt_residual


def _rt():
    return TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))


def test_tilt_residual_matches_fd_and_is_yaw_invariant():
    rt = _rt()
    q_mj = rt.q_init_full[:36].copy()
    R_ref = pin.exp3(np.array([0.1, -0.2, 0.3]))
    r0, A = pelvis_tilt_residual(rt, q_mj, R_ref)
    zhat = np.array([0.0, 0.0, 1.0])

    def u_of(qm):
        R_B = rt.body_rotation(qm, "pelvis")
        return R_B.T @ zhat

    u0 = u_of(q_mj)
    np.testing.assert_allclose(r0, R_ref.T @ zhat - u0, atol=1e-9)
    q_pin = rt.pin.qpos_mj_to_q_pin(q_mj)
    eps = 1e-6
    for k in range(rt.nv_a):
        v = np.zeros(rt.pin.model.nv)
        v[rt.v_a_indices[k]] = eps
        qm2 = q_mj.copy()
        qm2[:36] = rt.pin.q_pin_to_qpos_mj(pin.integrate(rt.pin.model, q_pin, v))
        fd = (u_of(qm2) - u0) / eps
        np.testing.assert_allclose(A[:, k], fd, atol=1e-4, err_msg=f"col {k}")


def test_style_off_by_default_and_additive_runs():
    from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
    # Style is OFF by default (GMR baseline); it is an additive weight.
    assert _rt().lambda_ws == 0.0
    # Turned on additively (on top of GMR tracking), the solve runs finite.
    rt_on = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh",
        retargeter=TestSocpRetargeterConfig(activate_ws=True, lambda_ws=1.0)))
    res = rt_on.retarget(max_frames=6)
    assert np.all(np.isfinite(res.qpos))
