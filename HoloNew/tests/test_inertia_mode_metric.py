"""Inertia mode produces a PHYSICALLY COHERENT body placed by contacts.

Design decision (2026-06-14): inertia mode places the body where the robot's
morphology + contacts dictate, NOT at the human pelvis. The contacts pin the feet
(and hands/object) and a weak W^c fills the residual; the pelvis ends up at a
roughly CONSTANT offset from the human reference (the robot's pelvis-over-feet
differs from the human's), which is expected and fine. So this metric does NOT
assert "distance to the human pelvis" — that is human tracking, which inertia mode
deliberately does not do. It asserts the placement is coherent:

  1. finite, pelvis z sane;
  2. feet PLANTED (low tangential foot slip — the body is not skating);
  3. NO runaway: the pelvis wander AFTER removing the constant frame offset is
     bounded (the offset is a fixed registration shift, not a divergence).

Observed (sub3_largebox_003, object_interaction, 30 frames, lambda_c=1e-5):
  foot slip ~4 mm; residual wander (after removing the mean offset) ~0.11 m.
"""
import numpy as np
import pytest
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.tests.paper_placement import paper_placement_config
from HoloNew.src.test_socp.tables import IK_MATCH_TABLE1, ROBOT_ROOT_NAME
from HoloNew.src.test_socp.targets import ground_frame_targets
from HoloNew.src.test_socp.interaction import (
    robot_control_points, query_entities, frame_references, _activation)

N = 30


def _ref_pelvis_xy(rt, n):
    gpos, gquat = rt.gmr_floor["pos"], rt.gmr_floor["quat"]
    out = []
    for t in range(n):
        tg = ground_frame_targets(gpos[t], gquat[t], IK_MATCH_TABLE1)
        for frame, (p_t, _, _, _) in tg.items():
            if rt.robot_link_names[frame] == ROBOT_ROOT_NAME:
                out.append(p_t[:2]); break
    return np.array(out)


def _foot_slip_mm(rt, qpos, n):
    """Mean tangential motion of active floor foot points minus the reference
    foot motion (skating). Low = feet planted."""
    L = rt.smplx_ground_probe.margin
    corr = rt.correspondence; M = corr.link_idx.shape[0]
    foot = [i for i in range(M) if any(k in corr.link_names[corr.link_idx[i]].lower()
            for k in ("ankle", "foot", "toe"))]
    I3 = np.eye(3); slips = []
    for t in range(1, n):
        qp = rt.pin.qpos_mj_to_q_pin(qpos[t, :36]); qpm = rt.pin.qpos_mj_to_q_pin(qpos[t-1, :36])
        Pt = robot_control_points(rt, qp); Ptm = robot_control_points(rt, qpm)
        _, fflr = query_entities(rt, Pt, rt._obj_poses_raw[t], margin=L)
        _, _, d_flr_t, _, pr_t = frame_references(rt, t)
        _, _, _, _, pr_tm = frame_references(rt, t-1)
        for i in foot:
            if _activation(float(d_flr_t[i]), L) > 0 and bool(fflr.active[i]):
                n0 = np.asarray(fflr.direction[i], float); Pi = I3 - np.outer(n0, n0)
                slips.append(np.linalg.norm(Pi @ ((Pt[i]-Ptm[i]) - (pr_t[i]-pr_tm[i]))))
    return float(np.mean(slips)) * 1000 if slips else float("nan")


def test_inertia_mode_largebox_coherent_placement():
    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="object_interaction", task_name="sub3_largebox_003",
        data_format="smplh", retargeter=paper_placement_config()))
    if rt.correspondence is None:
        pytest.skip("assets not present")
    res = rt.retarget(max_frames=N)

    # 1. finite + sane pelvis z.
    assert np.all(np.isfinite(res.qpos)), "non-finite qpos in inertia mode"
    z = res.qpos[:N, 2]
    assert z.min() >= 0.3 and z.max() <= 1.0, f"pelvis z out of range [{z.min():.3f},{z.max():.3f}]"

    # 2. feet planted (no skating). Parity gives ~3.6 mm; allow a generous ceiling.
    slip = _foot_slip_mm(rt, res.qpos, N)
    assert slip < 12.0, f"feet skating in inertia mode: slip={slip:.1f} mm (contacts not holding)"

    # 3. no runaway: pelvis wander after removing the constant frame offset is bounded.
    sol = res.qpos[:N, :2]; ref = _ref_pelvis_xy(rt, N)
    off = sol - ref
    residual = np.linalg.norm(off - off.mean(axis=0), axis=1)  # deviation from a constant offset
    assert residual.mean() < 0.15, (
        f"pelvis wanders (not a constant offset): residual mean={residual.mean():.3f} m")
