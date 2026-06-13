"""Validation metrics for the centroidal W^c / W^L objective (Brick 4).

Runs a 30-frame retarget on robot_only sub3_largebox_003 with activate_style=True
(default) and compares:
  A — activate_centroidal=False (baseline, Style on, pelvis_anchor_weight=1.0)
  B — activate_centroidal=True  (lambda_c=3.0, lambda_L=0.5, pelvis_anchor_weight=1.0)

Chosen parameters (tuned 2026-06-14, sub3_largebox_003, robot_only, smplh, 30 frames):
  lambda_c=3.0, lambda_L=0.5, pelvis_anchor_weight=1.0

Observed numbers:
  A (off):  CoM-accel err=7.9028 m/s^2, pelvis z=[0.562, 0.800] m, xy drift max=0.022 m
  B (on):   CoM-accel err=0.0060 m/s^2, pelvis z=[0.622, 0.800] m, xy drift max=0.228 m

The centroidal W^c term reduces CoM-acceleration tracking error by ~1300x (7.9028 ->
0.0060 m/s^2) and pelvis z is well within [0.4, 1.0] m. However, the pelvis xy drifts
up to 0.228 m from the reference (baseline: 0.022 m max). This is a characteristic
limitation of tracking acceleration without an absolute position constraint: once the
pelvis scaffold only covers position weakly (via w_p from the Style objective), and
W^c only constrains the second difference of CoM (not its absolute position), the xy
can drift despite the scaffold remaining at weight=1.0. Increasing the scaffold
(pelvis_anchor_weight > 1.0) reduces the xy drift but degrades CoM-accel tracking
and causes the pelvis z to drop below safe levels at paw >= 5.0 (z_min < 0.5 m).
There is no scaffold level that simultaneously keeps xy drift < 0.05 m and maintains
the CoM-accel improvement + sane z.

Conclusion: centroidal W^c/W^L is LEFT OFF by default (DONE_WITH_CONCERNS). The
CoM-accel reduction is validated and the z-stability is acceptable, but the xy drift
of 0.228 m (10x worse than baseline despite full scaffold) is an inherent limitation
that a scaffold adjustment cannot fix without sacrificing the centroidal benefit.

See docs/specs/2026-06-13-brick4-centroidal-design.md for the design note.
"""
from __future__ import annotations

import numpy as np
import pytest

from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter

MAX_FRAMES = 30
TASK_TYPE = "robot_only"
TASK_NAME = "sub3_largebox_003"
DATA_FORMAT = "smplh"

# Tuned parameters (2026-06-14).
_LAMBDA_C = 3.0
_LAMBDA_L = 0.5
_PELVIS_ANCHOR_WEIGHT = 1.0   # scaffold kept at default (removal causes excessive xy drift)


def _make_rt(activate_centroidal: bool, lambda_c: float, lambda_L: float,
             pelvis_anchor_weight: float) -> TestSocpRetargeter:
    cfg = RetargetingConfig(
        task_type=TASK_TYPE,
        task_name=TASK_NAME,
        data_format=DATA_FORMAT,
        retargeter=TestSocpRetargeterConfig(
            activate_centroidal=activate_centroidal,
            lambda_c=lambda_c,
            lambda_L=lambda_L,
            pelvis_anchor_weight=pelvis_anchor_weight,
            activate_style=True,   # keep default; isolates centroidal effect
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
    gpos = rt.gmr_ground["pos"][:T, 0, :]   # (T, 3) reference pelvis
    dt = rt._dt
    errors = []
    for t in range(2, T):
        cddot_ref = (gpos[t] - 2.0 * gpos[t - 1] + gpos[t - 2]) / dt ** 2
        cddot_solved = (coms[t] - 2.0 * coms[t - 1] + coms[t - 2]) / dt ** 2
        errors.append(float(np.linalg.norm(cddot_solved - cddot_ref)))
    return float(np.mean(errors))


def test_centroidal_reduces_com_accel_error_and_pelvis_sane():
    """Validate W^c/W^L: finite, pelvis z sane, CoM-accel error reduced vs baseline.

    Asserts:
      1. B (centroidal on) is finite and pelvis z stays in [0.4, 1.0] m.
      2. B's mean CoM-accel error < A's (baseline off).
      3. Pelvis xy drift from reference is within 0.40 m (sane band, not collapsed;
         0.228 m observed, which exceeds the baseline 0.022 m but is not a collapse).

    Note: the xy drift assertion uses 0.40 m (loose) rather than the baseline 0.022 m
    because W^c tracks acceleration (second difference) not absolute position; a weak
    scaffold cannot simultaneously prevent xy drift and preserve the CoM-accel gain.
    This is documented as a known limitation; centroidal is left OFF by default.
    """
    # A: baseline (centroidal off), Style on.
    rt_A = _make_rt(activate_centroidal=False, lambda_c=0.0, lambda_L=0.0,
                    pelvis_anchor_weight=_PELVIS_ANCHOR_WEIGHT)
    res_A = rt_A.retarget(max_frames=MAX_FRAMES)

    # B: centroidal on with tuned weights.
    rt_B = _make_rt(activate_centroidal=True, lambda_c=_LAMBDA_C, lambda_L=_LAMBDA_L,
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
        f"\n  lambda_c={_LAMBDA_C}  lambda_L={_LAMBDA_L}"
        f"  pelvis_anchor_weight={_PELVIS_ANCHOR_WEIGHT}"
        f"\n  NOTE: xy_drift={xy_drift.max():.4f} m > baseline 0.022 m"
        f" -- W^c tracks acceleration only; scaffold cannot fully fix xy drift."
        f" Centroidal left OFF by default (DONE_WITH_CONCERNS)."
    )
