"""Validation metric for W^o movable-entity bilateral coupling (Brick 5, Task 4).

Compares activate_movable=False (A) vs activate_movable=True (B) on
object_interaction sub3_largebox_003 over 30 frames and asserts:

  1. Finite + sane: all robot qpos finite; pelvis z in [0.35, 1.0] m.
  2. Object stays sensible: mean solved-object position error vs reference
     < 0.15 m for (B) — W^o + bilateral coupling keep the object near
     its reference trajectory.
  3. Contact gap not worse: mean |d_obj(robot) - d_obj_ref| for (B) is
     <= (A) + 0.005 m slack — making the object a variable must not
     worsen contact tracking.

Numbers re-recorded 2026-06-14 after the contact-persistence hard constraint
and the W^o position anchor (lambda_o_pos) landed. Persistence pins the robot's
contact points, so without an object position anchor the bilateral D/X coupling
offsets the (acceleration-only-regularized, position-blind) object and it drifts
~0.27 m. The lambda_o_pos=10 anchor (default) pins the absolute object position
to the reference path, which also improves contact tracking (the gap was being
measured against a drifted object pose):
  (A) activate_movable=False: mean obj pos err=0.0000 m (driven),
      mean contact gap~0.072 m
  (B) activate_movable=True, lambda_o=1.0, lambda_omega=1.0 (lambda_o_pos=10):
      mean obj pos err~0.0005 m, mean contact gap~0.054 m
  --> object stays within ~0.003 m of reference across 30 frames; contact gap
      is better with movable+anchor on. (With lambda_o_pos=0 the object drifts
      0.27 m and this test fails — the anchor is required.)
  Decision: ENABLED by default (activate_movable=True, lambda_o=1.0,
  lambda_omega=1.0, lambda_o_pos=10.0 in TestSocpRetargeterConfig).
"""
import numpy as np
import pytest

_MAX_FRAMES = 30
_CONTACT_GAP_SLACK = 0.005  # m: (B) gap <= (A) gap + slack
_OBJ_POS_ERR_LIMIT = 0.15   # m: mean solved-object error vs reference
_PELVIS_Z_LO = 0.35          # m
_PELVIS_Z_HI = 1.0           # m


def _build_rt(activate_movable, lambda_o=0.0, lambda_omega=0.0):
    from HoloNew.examples.robot_retarget import RetargetingConfig
    from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
    from HoloNew.src.test_socp.test_socp import TestSocpRetargeter

    cfg = RetargetingConfig(
        task_type="object_interaction",
        task_name="sub3_largebox_003",
        data_format="smplh",
        retargeter=TestSocpRetargeterConfig(
            activate_movable=activate_movable,
            lambda_o=lambda_o,
            lambda_omega=lambda_omega,
        ),
    )
    return TestSocpRetargeter.from_config(cfg)


def _contact_gap(rt, qpos_all, solved_obj_poses, n_frames):
    """Mean |d_obj(robot) - d_obj_ref| over active object-contact points.

    Uses the SOLVED object pose (so the metric is consistent with the actual
    decision the solve made for the object).
    """
    from HoloNew.src.test_socp.interaction import (
        _activation,
        frame_references,
        query_entities,
        robot_control_points,
    )

    L = rt.smplx_ground_probe.margin
    gaps = []
    for t in range(n_frames):
        obj_pose = solved_obj_poses[t]
        q_pin = rt.pin.qpos_mj_to_q_pin(qpos_all[t, :36])
        P = robot_control_points(rt, q_pin)
        fobj, _ = query_entities(rt, P, obj_pose, margin=L)
        d_obj_ref, _, _, _, _ = frame_references(rt, t)
        M = rt.correspondence.link_idx.shape[0]
        for i in range(M):
            if _activation(float(d_obj_ref[i]), L) > 0 and bool(fobj.active[i]):
                gaps.append(abs(float(fobj.distance[i]) - float(d_obj_ref[i])))
    return float(np.mean(gaps)) if gaps else float("nan")


def test_movable_metric():
    """Validate W^o movable-entity bilateral coupling on a 30-frame clip."""
    rt_a = _build_rt(activate_movable=False)
    if rt_a.correspondence is None or rt_a.object_sdf is None:
        pytest.skip("contact assets not present")

    # (A) Baseline: object driven at the reference.
    res_a = rt_a.retarget(max_frames=_MAX_FRAMES)
    assert np.all(np.isfinite(res_a.qpos)), "(A) qpos contains non-finite values"
    z_lo_a = float(res_a.qpos[:, 2].min())
    z_hi_a = float(res_a.qpos[:, 2].max())
    assert _PELVIS_Z_LO <= z_lo_a and z_hi_a <= _PELVIS_Z_HI, (
        f"(A) pelvis z out of sane range: [{z_lo_a:.3f}, {z_hi_a:.3f}] m"
    )

    ref_poses = rt_a._obj_poses_raw[:_MAX_FRAMES]
    solved_a = rt_a._obj_solved_poses[:_MAX_FRAMES]
    # (A) object pos err: driven => solved == reference => ~0.
    pos_err_a = float(np.mean([
        np.linalg.norm(s[4:7] - r[4:7]) for s, r in zip(solved_a, ref_poses)
    ]))

    gap_a = _contact_gap(rt_a, res_a.qpos, solved_a, _MAX_FRAMES)

    # (B) activate_movable=True with tuned lambda_o=1.0, lambda_omega=1.0.
    rt_b = _build_rt(activate_movable=True, lambda_o=1.0, lambda_omega=1.0)
    res_b = rt_b.retarget(max_frames=_MAX_FRAMES)
    assert np.all(np.isfinite(res_b.qpos)), "(B) qpos contains non-finite values"
    z_lo_b = float(res_b.qpos[:, 2].min())
    z_hi_b = float(res_b.qpos[:, 2].max())
    assert _PELVIS_Z_LO <= z_lo_b and z_hi_b <= _PELVIS_Z_HI, (
        f"(B) pelvis z out of sane range: [{z_lo_b:.3f}, {z_hi_b:.3f}] m"
    )

    solved_b = rt_b._obj_solved_poses[:_MAX_FRAMES]
    assert len(solved_b) == _MAX_FRAMES, (
        f"(B) expected {_MAX_FRAMES} solved object poses, got {len(solved_b)}"
    )
    for i, pose7 in enumerate(solved_b):
        assert np.all(np.isfinite(pose7)), f"(B) frame {i}: solved object pose is non-finite"

    pos_err_b = float(np.mean([
        np.linalg.norm(s[4:7] - r[4:7]) for s, r in zip(solved_b, ref_poses)
    ]))
    # Object must stay sensible: mean position error < 0.15 m.
    # Numbers: 0.0004 m (B) vs 0.0000 m (A).
    assert pos_err_b < _OBJ_POS_ERR_LIMIT, (
        f"(B) object drifted from reference: mean pos err={pos_err_b:.4f} m "
        f"(limit={_OBJ_POS_ERR_LIMIT} m)"
    )

    gap_b = _contact_gap(rt_b, res_b.qpos, solved_b, _MAX_FRAMES)
    # Contact gap must not worsen: (B) <= (A) + slack.
    # Numbers: gap_a=0.07202 m, gap_b=0.07183 m (B is slightly better).
    assert gap_b <= gap_a + _CONTACT_GAP_SLACK, (
        f"Contact gap regressed: (A)={gap_a:.5f} m, (B)={gap_b:.5f} m "
        f"(slack={_CONTACT_GAP_SLACK} m)"
    )
