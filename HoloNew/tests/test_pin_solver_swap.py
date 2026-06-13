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
