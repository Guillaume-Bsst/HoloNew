"""Validation tests for the pinocchio solver swap (Task 7a).

Verifies that after migrating tracking kinematics to pinocchio:
- nv_a == 35 for the default full-body case (q_a_init_idx == -7)
- retarget() completes and returns a finite qpos trajectory with correct shape
"""
import numpy as np


def test_tangent_size_and_solve_finite():
    from HoloNew.examples.robot_retarget import RetargetingConfig
    from HoloNew.src.test_socp.test_socp import TestSocpRetargeter

    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    assert rt.nv_a == 35, f"expected nv_a=35, got {rt.nv_a}"
    res = rt.retarget()
    assert np.all(np.isfinite(res.qpos)), "NaN or Inf in qpos output"
    assert res.qpos.shape[1] >= 36, f"unexpected qpos column count: {res.qpos.shape[1]}"


def test_pin_tracking_quality():
    import numpy as np
    from HoloNew.examples.robot_retarget import RetargetingConfig
    from HoloNew.src.test_socp.test_socp import TestSocpRetargeter

    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    res = rt.retarget()

    # gmr_ground["pos"] is (T, 14, 3): 14 SMPL-H joints in ground frame.
    # Index 0 is the pelvis — the primary tracking target for the robot base.
    ref = rt.gmr_ground["pos"][:, 0, :]   # (T, 3) ground pelvis reference
    solved = res.qpos[:, 0:3]              # (T, 3) solved base position

    err = np.linalg.norm(solved - ref, axis=1)
    mean_err = float(np.mean(err))
    # Observed mean pelvis tracking error after pinocchio migration: ~0.045 m.
    # Threshold set to 0.08 m as a regression guard with reasonable headroom.
    assert mean_err < 0.08, (
        f"mean pelvis tracking error {mean_err:.4f} m too high "
        f"(expected < 0.08 m; observed baseline ~0.045 m)"
    )
