"""Validation metrics for the pelvis-relative Style objective (Brick 3).

Runs a 30-frame retarget on robot_only sub3_largebox_003 with activate_style=True
and checks:
  1. Finite + no pose collapse: all qpos finite; pelvis z in [0.4, 1.0] m.
  2. Joint-orientation fidelity: mean pelvis-relative orientation error over
     tracked non-pelvis bodies is comparable to (or better than) the same
     metric from world-frame tracking (activate_style=False).
  3. Base-xy drift: on object_interaction sub3_largebox_003, 30 frames, the
     default config (pelvis_anchor_weight=10.0) keeps mean xy-drift below
     0.12 m.  Guards against the position scaffold silently weakening again.

Pelvis-relative orientation error for body k at frame t:
    err_k_t = || log(R_tilde_k^{-1} @ R_tilde_k_ref) ||
where
    R_tilde_k     = R_B^{-1} @ R_k     (solved pelvis-relative joint orientation)
    R_tilde_k_ref = R_B_ref^{-1} @ R_k_ref  (reference pelvis-relative joint orientation)

Observed numbers (recorded 2026-06-13, sub3_largebox_003, robot_only, smplh, 30 frames):
    Style pelvis-relative fidelity : 0.6003 rad
    World pelvis-relative fidelity : 0.8180 rad  (Style is ~27 % better)
    Pelvis z range (30 frames)     : [0.562, 0.800] m  (well within [0.4, 1.0])

Base-xy drift (recorded 2026-06-14, sub3_largebox_003, object_interaction, smplh,
    30 frames, pelvis_anchor_weight=10.0):
    max  drift : 0.1261 m
    mean drift : 0.0491 m  (well within the 0.12 m limit)
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.tables import IK_MATCH_TABLE1, ROBOT_ROOT_NAME
from HoloNew.src.test_socp.targets import ground_frame_targets
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter

MAX_FRAMES = 30
TASK_TYPE = "robot_only"
TASK_NAME = "sub3_largebox_003"
DATA_FORMAT = "smplh"


def _make_rt(activate_style: bool) -> TestSocpRetargeter:
    cfg = RetargetingConfig(
        task_type=TASK_TYPE,
        task_name=TASK_NAME,
        data_format=DATA_FORMAT,
    )
    rt = TestSocpRetargeter.from_config(cfg)
    rt.activate_style = activate_style
    return rt


def _pelvis_relative_fidelity(rt: TestSocpRetargeter, qpos: np.ndarray) -> float:
    """Compute mean pelvis-relative orientation error over all tracked non-pelvis
    bodies and over all solved frames.

    For each frame t and each non-pelvis body k:
        R_tilde_k     = R_B^{-1}     @ R_k       (solved, pelvis-relative)
        R_tilde_k_ref = R_B_ref^{-1} @ R_k_ref   (reference, pelvis-relative)
        err_k_t       = || log(R_tilde_k^{-1} @ R_tilde_k_ref) ||  [rad]

    Returns the mean over all (t, k) pairs.
    """
    gpos = rt.gmr_ground["pos"]     # (T_full, B, 3)
    gquat = rt.gmr_ground["quat"]   # (T_full, B, 4) wxyz

    T = min(qpos.shape[0], MAX_FRAMES)
    errors = []

    for t in range(T):
        # Reconstruct the full config at frame t from the solved qpos row.
        q = rt.q_init_full.copy()
        q[:qpos.shape[1]] = qpos[t]

        # Reference targets for this frame.
        tg = ground_frame_targets(gpos[t], gquat[t], IK_MATCH_TABLE1)

        # Pelvis reference and solved rotations.
        R_B_ref = None
        R_B_solved = rt.body_rotation(q, ROBOT_ROOT_NAME)
        for frame, (_, R_t, _, _) in tg.items():
            if rt.robot_link_names[frame] == ROBOT_ROOT_NAME:
                R_B_ref = R_t
                break

        if R_B_ref is None:
            continue  # pelvis not in table (should not happen)

        for frame, (_, R_t, _, w_r) in tg.items():
            body = rt.robot_link_names[frame]
            if body == ROBOT_ROOT_NAME or w_r == 0:
                continue  # skip pelvis and untracked bodies

            R_k_solved = rt.body_rotation(q, body)
            # Pelvis-relative orientations.
            R_tilde_solved = R_B_solved.T @ R_k_solved
            R_tilde_ref = R_B_ref.T @ R_t
            # log-map distance (rotvec norm).
            delta = R_tilde_solved.T @ R_tilde_ref
            err = np.linalg.norm(Rotation.from_matrix(delta).as_rotvec())
            errors.append(err)

    return float(np.mean(errors)) if errors else float("nan")


def test_style_finite_no_collapse_and_fidelity():
    """Validate activate_style=True over 30 frames: finite, sane pelvis z,
    comparable pelvis-relative joint fidelity to world-frame tracking.

    Observed on sub3_largebox_003 (2026-06-13):
        style  pelvis-relative err : 0.6003 rad
        world  pelvis-relative err : 0.8180 rad  (Style ~27 % better)
        pelvis z range             : [0.562, 0.800] m
    """
    # --- Style solve ---
    rt_style = _make_rt(activate_style=True)
    res_style = rt_style.retarget(max_frames=MAX_FRAMES)

    # 1. Finite + no pose collapse.
    assert np.all(np.isfinite(res_style.qpos)), "Style solve produced non-finite qpos"
    pelvis_z = res_style.qpos[:, 2]
    assert np.all(pelvis_z >= 0.4), (
        f"Pelvis z collapsed below 0.4 m; min={pelvis_z.min():.4f}"
    )
    assert np.all(pelvis_z <= 1.0), (
        f"Pelvis z exceeded 1.0 m; max={pelvis_z.max():.4f}"
    )

    # --- World-frame solve (reference) ---
    rt_world = _make_rt(activate_style=False)
    res_world = rt_world.retarget(max_frames=MAX_FRAMES)

    # 2. Pelvis-relative fidelity comparison.
    style_err = _pelvis_relative_fidelity(rt_style, res_style.qpos)
    world_err = _pelvis_relative_fidelity(rt_world, res_world.qpos)

    # Record observed values in the assertion message for traceability.
    assert not np.isnan(style_err), "Style fidelity metric is NaN"
    assert not np.isnan(world_err), "World fidelity metric is NaN"

    # Style tracks pelvis-relative orientations explicitly; it should be at least
    # as good as world tracking (which does not optimise pelvis-relative errors
    # directly). Allow a 0.05 rad slack for numerical noise.
    assert style_err <= world_err + 0.05, (
        f"Style pelvis-relative fidelity ({style_err:.4f} rad) is worse than "
        f"world tracking ({world_err:.4f} rad) by more than 0.05 rad slack"
    )

    # Log the numbers so they appear in verbose test output.
    print(
        f"\n[style_metric] style_err={style_err:.4f} rad  "
        f"world_err={world_err:.4f} rad  "
        f"pelvis_z=[{pelvis_z.min():.3f}, {pelvis_z.max():.3f}] m"
    )


def test_style_base_drift_bounded():
    """Assert that the default Style config (pelvis_anchor_weight=10.0) keeps
    the solved base-xy close to the reference pelvis trajectory on
    object_interaction sub3_largebox_003, 30 frames.

    Style frees the pelvis ORIENTATION (yaw), not its position; the scaffold
    must be strong enough to prevent xy drift.

    Observed (2026-06-14, paw=10.0): max=0.126 m, mean=0.049 m.
    Regression limit: mean < 0.12 m.
    """
    cfg = RetargetingConfig(
        task_type="object_interaction",
        task_name=TASK_NAME,
        data_format=DATA_FORMAT,
    )
    rt = TestSocpRetargeter.from_config(cfg)
    res = rt.retarget(max_frames=MAX_FRAMES)

    gpos = rt.gmr_ground["pos"]   # (T_full, B, 3)
    gquat = rt.gmr_ground["quat"]  # (T_full, B, 4) wxyz

    T = min(res.qpos.shape[0], MAX_FRAMES)
    ref_xy = []
    for t in range(T):
        tg = ground_frame_targets(gpos[t], gquat[t], IK_MATCH_TABLE1)
        for frame, (p_t, _, _, _) in tg.items():
            if rt.robot_link_names[frame] == ROBOT_ROOT_NAME:
                ref_xy.append(p_t[:2])
                break

    ref_xy = np.array(ref_xy)
    solved_xy = res.qpos[:len(ref_xy), :2]
    drifts = np.linalg.norm(solved_xy - ref_xy, axis=1)
    mean_drift = float(drifts.mean())
    max_drift = float(drifts.max())

    print(
        f"\n[style_drift] paw={rt.pelvis_anchor_weight}  "
        f"mean_drift={mean_drift:.4f} m  max_drift={max_drift:.4f} m"
    )

    assert mean_drift < 0.12, (
        f"Base-xy mean drift {mean_drift:.4f} m exceeds 0.12 m limit "
        f"(pelvis_anchor_weight={rt.pelvis_anchor_weight}); "
        f"the Style position scaffold may have been weakened."
    )
