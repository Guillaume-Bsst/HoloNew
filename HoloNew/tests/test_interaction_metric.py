"""Brick 1 acceptance metric: the D/X interaction costs reduce the robot-side
contact gap versus the reference, on a fast object-clip segment.

D/X are enabled by default (with the auto-coupled ground non-penetration that
keeps them stable). This test is the acceptance gate: turning them off must make
the mean floor-contact gap worse (i.e. on < off).
"""
import numpy as np
import pytest

from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
from HoloNew.src.test_socp.interaction import (
    robot_control_points, query_entities, frame_references,
)


def _mean_floor_gap(rt, res):
    """Mean |d_floor(robot) - d_floor_ref| over floor-active contact points."""
    gaps = []
    for t in range(res.qpos.shape[0]):
        q_pin = rt.pin.qpos_mj_to_q_pin(res.qpos[t, :36])
        P = robot_control_points(rt, q_pin)
        _, fflr = query_entities(rt, P, rt._obj_poses_raw[t])
        _, _, d_flr_ref, _, _ = frame_references(rt, t)
        m = (d_flr_ref < 0.05) & np.asarray(fflr.active, dtype=bool)
        if m.any():
            gaps.append(float(np.mean(np.abs(np.asarray(fflr.distance)[m] - d_flr_ref[m]))))
    return float(np.mean(gaps)) if gaps else None


def test_dx_reduces_contact_gap():
    # ON: default config (lambda_D=lambda_X=1.0, ground non-penetration coupled).
    rt_on = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="object_interaction", task_name="sub3_largebox_003", data_format="smplh"))
    if rt_on.correspondence is None or rt_on.object_sdf is None:
        pytest.skip("correspondence/object_sdf assets not present")
    # OFF: interaction weights zeroed.
    rt_off = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="object_interaction", task_name="sub3_largebox_003", data_format="smplh",
        retargeter=TestSocpRetargeterConfig(lambda_D=0.0, lambda_X=0.0, lambda_P=0.0)))

    K = 8
    gap_on = _mean_floor_gap(rt_on, rt_on.retarget(max_frames=K))
    gap_off = _mean_floor_gap(rt_off, rt_off.retarget(max_frames=K))
    # Observed (sub3_largebox_003, K=8): on ~0.012 vs off ~0.028.
    assert gap_on is not None and gap_off is not None
    assert gap_on < gap_off, f"D/X did not reduce the contact gap: on={gap_on:.4f} >= off={gap_off:.4f}"
