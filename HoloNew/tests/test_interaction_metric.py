"""Brick 1 acceptance metric: the D/X interaction costs reduce the robot-side
OBJECT contact gap versus the reference, in the full default pipeline.

D/X are enabled by default (lambda_D=lambda_X=5.0, with the auto-coupled ground
non-penetration that keeps them stable). This test is the acceptance gate:
turning them off must make the mean OBJECT contact gap worse (on < off).

Why the object channel (re-pointed 2026-06-14): D/X's original isolated
validation measured the FLOOR gap, but in the matured pipeline the floor
contact's no-slip is owned by the persistence hard constraint
(activate_persistence), which D/X cannot beat on the floor at any weight
(D/X off + persistence on gives floor gap ~0.0107 m). D/X's enduring,
non-redundant job is the OBJECT (manipulation) contact channel, where at the
re-tuned lambda=20.0 (aligned frame, root_xy_scale=1.0) it cuts the gap ~15%
(object ~0.0276 on vs ~0.0323 off on sub3_largebox_003, K=30). See the lambda
sweep recorded in config.py.
"""
import numpy as np
import pytest

from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
from HoloNew.src.test_socp.interaction import (
    robot_control_points, query_entities, frame_references, _activation,
)


def _mean_object_gap(rt, res):
    """Mean |d_obj(robot) - d_obj_ref| over object-active contact points."""
    L = rt.smplx_ground_probe.margin
    M = rt.correspondence.link_idx.shape[0]
    gaps = []
    for t in range(res.qpos.shape[0]):
        q_pin = rt.pin.qpos_mj_to_q_pin(res.qpos[t, :36])
        P = robot_control_points(rt, q_pin)
        fobj, _ = query_entities(rt, P, rt._obj_poses_raw[t], margin=L)
        d_obj_ref, _, _, _, _ = frame_references(rt, t)
        m = np.array([
            _activation(float(d_obj_ref[i]), L) > 0 and bool(fobj.active[i])
            for i in range(M)
        ])
        if m.any():
            gaps.append(float(np.mean(np.abs(np.asarray(fobj.distance)[m] - d_obj_ref[m]))))
    return float(np.mean(gaps)) if gaps else None


def test_dx_reduces_object_contact_gap():
    # ON: default config (lambda_D=lambda_X=5.0, ground non-penetration coupled).
    rt_on = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="object_interaction", task_name="sub3_largebox_003", data_format="smplh"))
    if rt_on.correspondence is None or rt_on.object_sdf is None:
        pytest.skip("correspondence/object_sdf assets not present")
    # OFF: interaction weights zeroed (persistence/movable defaults unchanged).
    rt_off = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="object_interaction", task_name="sub3_largebox_003", data_format="smplh",
        retargeter=TestSocpRetargeterConfig(lambda_D=0.0, lambda_X=0.0, lambda_P=0.0)))

    # Use a window that includes the manipulation phase: on sub3_largebox_003 the
    # hands engage the box only after ~frame 9, so a short window (e.g. K=8) can
    # contain no active object contacts at all (gap = None). K=30 covers the grasp.
    K = 30
    gap_on = _mean_object_gap(rt_on, rt_on.retarget(max_frames=K))
    gap_off = _mean_object_gap(rt_off, rt_off.retarget(max_frames=K))
    assert gap_on is not None and gap_off is not None, (
        "no active object contacts in the window — widen K")
    assert gap_on < gap_off, (
        f"D/X did not reduce the object contact gap: on={gap_on:.4f} >= off={gap_off:.4f}")
