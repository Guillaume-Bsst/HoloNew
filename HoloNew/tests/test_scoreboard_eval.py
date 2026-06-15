"""Integration check: _scoreboard_metrics on a real retargeted trajectory."""
from pathlib import Path

import numpy as np
import pytest

_NPZ = Path("demo_results/g1/robot_only/omomo/sub3_largebox_003.npz")


def _evaluator():
    from HoloNew.config_types.robot import RobotConfig
    from HoloNew.config_types.data_type import MotionDataConfig
    from HoloNew.evaluation.eval_retargeting import (
        create_task_constants, RetargetingEvaluator)

    rc = RobotConfig(robot_type="g1")
    mc = MotionDataConfig(data_format="smplh", robot_type="g1")
    constants = create_task_constants(rc, mc, object_name="ground")
    return RetargetingEvaluator(
        robot_model_path=constants.ROBOT_URDF_FILE,
        object_model_path=getattr(constants, "OBJECT_URDF_FILE", None),
        object_name=constants.OBJECT_NAME,
        demo_joints=constants.DEMO_JOINTS,
        joints_mapping=constants.JOINTS_MAPPING,
        visualize=False,
        constants=constants,
    )


def test_scoreboard_metrics_finite_on_real_npz():
    if not _NPZ.exists():
        pytest.skip(f"{_NPZ} not present")
    d = np.load(_NPZ, allow_pickle=True)
    m = _evaluator()._scoreboard_metrics(d["qpos"], d["human_joints"], d["fps"])

    expected = {
        "base_pos_accel_rms", "base_ang_accel_rms", "joint_accel_rms",
        "joint_jerk_rms", "joint_jerk_meanabs",
        "joint_limit_margin_min", "joint_limit_saturation_frac",
        "joint_vel_rms", "joint_vel_peak",
        "mpjpe_global", "mpjpe_root_rel", "base_track_err",
        "com_accel_err",
    }
    missing = expected - set(m)
    assert not missing, f"missing scoreboard keys: {missing}"
    assert all(np.isfinite(v) for v in m.values()), m
    # Sanity: MPJPE is positive and in a plausible meter range for a retarget.
    assert 0.0 < m["mpjpe_global"] < 1.0


def test_metric_family_gating():
    if not _NPZ.exists():
        pytest.skip(f"{_NPZ} not present")
    d = np.load(_NPZ, allow_pickle=True)
    ev = _evaluator()
    ev.metric_families = {"smoothness"}
    m = ev._scoreboard_metrics(d["qpos"], d["human_joints"], d["fps"])
    assert "joint_accel_rms" in m
    assert "mpjpe_global" not in m
    assert "com_accel_err" not in m
