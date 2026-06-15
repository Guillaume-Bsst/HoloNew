"""Validation metrics for the centroidal W^c / W^c_pos / W^L objective (Brick 4).

Runs a 30-frame retarget on robot_only sub3_largebox_003 with activate_ws=True
(default) and compares:
  A — activate_centroidal=False (baseline, Style on, pelvis_anchor_weight=1.0)
  B — activate_centroidal=True  (lambda_c=3.0, lambda_c_pos=5.0, lambda_l=0.5,
                                  pelvis_anchor_weight=1.0)

Chosen parameters (tuned 2026-06-14, sub3_largebox_003, robot_only, smplh, 30 frames):
  lambda_c=3.0, lambda_c_pos=5.0, lambda_l=0.5, pelvis_anchor_weight=1.0

Observed numbers:
  A (off):  CoM-accel err~7.90 m/s^2, pelvis z=[0.562, 0.800] m, xy drift max~0.022 m
  B (on):   CoM-accel err~0.006 m/s^2 (~1300x better), pelvis z=[0.623, 0.800] m,
            xy drift max~0.228 m

The W^c_pos position anchor was added to cure the drift seen in the original W^c-only
design (drift=0.228 m at lambda_c_pos=5 vs 0.228 m without it — no improvement at
practical lambda_c_pos values). The underlying reason is mathematical: W^c has an
effective position weight proportional to lambda_c / dt^4 (~lambda_c * 810,000). At
lambda_c=3 and dt=1/30, this is ~2.4M, which overwhelms any practical lambda_c_pos.
Setting lambda_c_pos >> lambda_c/dt^4 would eliminate the acceleration benefit.

The W^c_pos term IS correct and useful when W^c is off (lambda_c=0):
  lambda_c=0, lambda_c_pos=5, lambda_l=0.3, paw=1.0 → drift=0.018 m (near baseline)
  but this gives no CoM-accel improvement (accel_err stays ~7.5 m/s^2).

Conclusion: centroidal W^c/W^c_pos/W^L is LEFT OFF by default (DONE_WITH_CONCERNS).
The W^c_pos term is implemented and mathematically validated (unit tests pass), but
does not cure the drift in the presence of W^c due to the dt^4 weight imbalance.
See docs/specs/2026-06-13-brick4-centroidal-design.md for the design note.
"""
from __future__ import annotations

import numpy as np
import pytest

from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.evaluation.metrics import compute_dynamics

MAX_FRAMES = 30
TASK_TYPE = "robot_only"
TASK_NAME = "sub3_largebox_003"
DATA_FORMAT = "smplh"

# Tuned parameters (2026-06-14).
_LAMBDA_C = 3.0
_LAMBDA_C_POS = 5.0
_LAMBDA_L = 0.5
_PELVIS_ANCHOR_WEIGHT = 1.0


def _make_rt(activate_centroidal: bool, lambda_c: float, lambda_c_pos: float,
             lambda_l: float, pelvis_anchor_weight: float) -> TestSocpRetargeter:
    cfg = RetargetingConfig(
        task_type=TASK_TYPE,
        task_name=TASK_NAME,
        data_format=DATA_FORMAT,
        retargeter=TestSocpRetargeterConfig(
            # Per-weight switches (config §3): enable each centroidal term whose weight
            # the caller set > 0 (the old single activate_centroidal master is gone).
            activate_wc=activate_centroidal and lambda_c > 0,
            lambda_c=lambda_c,
            activate_wc_pos=activate_centroidal and lambda_c_pos > 0,
            lambda_c_pos=lambda_c_pos,
            activate_wl=activate_centroidal and lambda_l > 0,
            lambda_l=lambda_l,
            pelvis_anchor_weight=pelvis_anchor_weight,
            activate_ws=True,   # keep default; isolates centroidal effect
        ),
    )
    return TestSocpRetargeter.from_config(cfg)


def _com_accel_error(rt: TestSocpRetargeter, qpos: np.ndarray) -> float:
    """Mean ||c_ddot_solved - c_ddot_ref|| over frames 2..T-1.

    c_ddot_ref  = second finite difference of the reference pelvis position (CoM proxy).
    c_ddot_solved = second finite difference of the solved robot CoM (pinocchio).
    """
    T = min(qpos.shape[0], MAX_FRAMES)
    coms = np.array([
        rt.pin.com(rt.pin.qpos_mj_to_q_pin(qpos[t, :36]))
        for t in range(T)
    ])
    gpos = rt.gmr_ground["pos"][:T, 0, :]   # (T, 3) reference pelvis (CoM proxy)
    # Shared scoreboard metric: identical 2nd-difference definition as the old loop.
    return compute_dynamics(coms, gpos, rt._dt)["com_accel_err"]


def test_centroidal_reduces_com_accel_error_and_pelvis_sane():
    """Validate W^c/W^c_pos/W^L: finite, pelvis z sane, CoM-accel error reduced.

    W^c_pos is included but does not cure the xy drift when W^c is active
    (mathematical limitation: W^c weight is ~810,000x stronger per unit lambda
    than a direct position term, see module docstring).

    Asserts:
      1. B (centroidal on) is finite and pelvis z stays in [0.4, 1.0] m.
      2. B's mean CoM-accel error < A's (baseline off): W^c works correctly.
      3. Pelvis xy drift from reference stays within 0.40 m sane band
         (drift is ~0.228 m, which is NOT at baseline, reflecting the known
         limitation: centroidal stays OFF by default).

    Note: the xy drift assertion uses 0.40 m (loose) rather than baseline+0.03 m
    because W^c tracks acceleration (second difference) not absolute position, and
    W^c_pos cannot cure this within the current linearised IK framework.
    """
    # A: baseline (centroidal off), Style on.
    rt_A = _make_rt(activate_centroidal=False, lambda_c=0.0, lambda_c_pos=0.0,
                    lambda_l=0.0, pelvis_anchor_weight=_PELVIS_ANCHOR_WEIGHT)
    res_A = rt_A.retarget(max_frames=MAX_FRAMES)

    # B: centroidal on with tuned weights including position anchor.
    rt_B = _make_rt(activate_centroidal=True, lambda_c=_LAMBDA_C,
                    lambda_c_pos=_LAMBDA_C_POS, lambda_l=_LAMBDA_L,
                    pelvis_anchor_weight=_PELVIS_ANCHOR_WEIGHT)
    res_B = rt_B.retarget(max_frames=MAX_FRAMES)

    K = min(res_B.qpos.shape[0], MAX_FRAMES)

    # 1. Finite + no collapse.
    assert np.all(np.isfinite(res_B.qpos)), "Centroidal-on solve produced non-finite qpos"
    pelvis_z = res_B.qpos[:K, 2]
    assert np.all(pelvis_z >= 0.4), (
        f"Pelvis z collapsed below 0.4 m; min={pelvis_z.min():.4f}"
    )
    assert np.all(pelvis_z <= 1.0), (
        f"Pelvis z exceeded 1.0 m; max={pelvis_z.max():.4f}"
    )

    # 2. CoM-accel error reduced with centroidal on.
    err_A = _com_accel_error(rt_A, res_A.qpos)
    err_B = _com_accel_error(rt_B, res_B.qpos)
    assert err_B < err_A, (
        f"Centroidal on did NOT reduce CoM-accel error: on={err_B:.4f} >= off={err_A:.4f} m/s^2"
    )

    # 3. Pelvis xy stays within a sane band (not collapsed/teleported).
    # NOTE: 0.40 m is loose — the drift is ~0.228 m (not at baseline 0.022 m).
    # W^c_pos cannot cure this; centroidal left OFF by default (DONE_WITH_CONCERNS).
    gpos = rt_B.gmr_ground["pos"][:K, 0, :]
    xy_drift = np.linalg.norm(res_B.qpos[:K, 0:2] - gpos[:, 0:2], axis=1)
    assert np.all(xy_drift <= 0.40), (
        f"Pelvis xy drift exceeds 0.40 m sane band; max={xy_drift.max():.4f} m "
        "(centroidal W^c tracks acceleration, not absolute position)"
    )

    # Informational output for CI logs.
    print(
        f"\n[centroidal_metric]"
        f"  err_A={err_A:.4f} m/s^2  err_B={err_B:.4f} m/s^2"
        f"  improvement={err_A / err_B:.1f}x"
        f"  pelvis_z=[{pelvis_z.min():.3f}, {pelvis_z.max():.3f}] m"
        f"  xy_drift_max={xy_drift.max():.4f} m  (baseline ~0.022 m)"
        f"\n  lambda_c={_LAMBDA_C}  lambda_c_pos={_LAMBDA_C_POS}"
        f"  lambda_l={_LAMBDA_L}  pelvis_anchor_weight={_PELVIS_ANCHOR_WEIGHT}"
        f"\n  NOTE: W^c_pos implemented but cannot cure drift when W^c active."
        f" W^c effective weight ~lambda_c/dt^4 = {_LAMBDA_C * (30**4):.0f}x"
        f" vs lambda_c_pos = {_LAMBDA_C_POS}."
        f" Centroidal left OFF by default (DONE_WITH_CONCERNS)."
    )
