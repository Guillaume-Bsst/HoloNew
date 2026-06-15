# Retargeting Scoreboard — Design Spec

Date: 2026-06-15
Status: approved (design), implementation pending

## Goal

A "giga scoreboard" of retargeting-quality metrics to drive term-by-term ablation
("use the battery of weights intelligently to beat SOTA"). Target is **internal-
relative first**: beat our own GMR baseline; metric definitions are ours, not yet
matched to a specific SOTA paper.

Usage mode: **single-config report** — run one config over a sweep of clips, print
mean/std per metric (the existing `evaluation/eval_retargeting.py` behavior). Ablation
= rerun with a different config. No multi-config orchestration in v1.

## Principle: pure metric functions on extracted arrays

Each metric is a **pure function taking already-extracted numpy arrays**, not models or
retargeter objects. Extraction (FK, CoM, joint limits) stays at the call site:
- Offline evaluator (`RetargetingEvaluator`): MuJoCo FK / `subtree_com` / model ranges.
- Live tests (`test_*_metric.py`): pinocchio `rt.pin`, `rt.gmr_ground`, or the
  diagnostics already stored on `RetargetResult` (`com`, `angular_momentum`, refs).

This gives a **single source of truth** for each formula, shared by the offline sweep
and the in-process A/B tests, regardless of backend.

## Architecture

```
evaluation/metrics/
  __init__.py
  smoothness.py   compute_smoothness(qpos, dof, dt)                  -> dict[str, float]
  effort.py       compute_effort(joints, q_lower, q_upper, dt)       -> dict[str, float]
  tracking.py     compute_tracking(robot_kpts, ref_kpts, root_idx)   -> dict[str, float]
  dynamics.py     compute_dynamics(com, ref_com, dt, *, L=None, L_ref=None) -> dict[str, float]
```

Data flow:
`retarget -> qpos (+ RetargetResult diagnostics)` ->
`RetargetingEvaluator` extracts arrays (FK, subtree_com, model ranges, SMPL ref joints
via `preprocess_motion_data` + `joints_mapping`/`demo_joints`) ->
calls the four `compute_*` -> merges their dicts into the existing per-task result dict
(alongside penetration / contact precision / foot sliding) ->
`main()` aggregates mean/std per key; new keys flow through automatically.

## Metric definitions

Conventions: qpos columns `[0:3]` base xyz, `[3:7]` base quat (wxyz), `[7:7+dof]`
joints, optional trailing `[-7:]` object pose. `dt = 1/fps`. Acceleration = 2nd finite
difference `/ dt**2`; jerk = 3rd finite difference `/ dt**3`. Physical units throughout.

### smoothness.py (pure qpos)
- `base_pos_accel_rms` — RMS of 2nd diff of `qpos[:,0:3]` / dt² (m/s²).
- `base_ang_accel_rms` — RMS of 2nd diff of base angular velocity (rad/s²); ω from
  `2 * log(q_{t-1}^{-1} q_t) / dt` via scipy Rotation.
- `joint_accel_rms` — RMS of 2nd diff of joints / dt² (rad/s²).
- `joint_jerk_rms` — RMS of 3rd diff of joints / dt³ (rad/s³).
- `joint_jerk_meanabs` — `mean(|diff(joints, n=3)|)` **per-frame, no dt** — kept
  identical to the existing W^r metric in `test_temporal_metric.py` for continuity.

### effort.py (joints + model limits)
- `joint_limit_margin_min` — min over (frame, joint) of normalized distance to nearest
  limit: `min((q - q_lower), (q_upper - q)) / (q_upper - q_lower)`. Negative = violation.
- `joint_limit_saturation_frac` — fraction of (frame, joint) with normalized margin < ε
  (ε = 0.02).
- `joint_vel_rms` — RMS of 1st diff of joints / dt (rad/s).
- `joint_vel_max` — max |joint velocity| (rad/s).
- Torque proxy: out of scope v1 (needs inverse dynamics). YAGNI.

### tracking.py (robot keypoints via FK vs SMPL joints via joints_mapping)
- `mpjpe_global` — mean over frames & mapped joints of `||p_robot - p_smpl||` (m).
- `mpjpe_root_rel` — same after subtracting the root (pelvis index `root_idx`) from both.
- `base_track_err` — mean `||robot base xyz - ref root xyz||` (m) — matches the existing
  `track` metric in `test_temporal_metric.py`.
- Reference = the same SMPL demo joints the retargeter consumes (`demo_joints`,
  `joints_mapping`). No Procrustes alignment (global placement is part of the task).

### dynamics.py (CoM + reference; reuses RetargetResult diagnostics when present)
- `com_accel_err` — `mean ||c_ddot_robot - c_ddot_ref||` over frames 2..T-1, c_ddot via
  2nd diff (m/s²). **Identical** to `_com_accel_error` in `test_centroidal_metric.py`.
  Offline: `c_robot` from MuJoCo `subtree_com[0]`, `c_ref` from SMPL CoM proxy (pelvis).
  Live: from `RetargetResult.com` / `com_ref`.
- `ang_momentum_rms` — RMS magnitude of centroidal angular momentum L (kg·m²/s). Reuses
  `RetargetResult.angular_momentum` when present; offline fallback lumps L from MuJoCo
  body masses/inertias + finite-difference body velocities.
- ZMP / support-polygon margin — **stretch, out of v1** (contact-detection dependent,
  low ROI for the first cut). Documented here as the next dynamic metric.

## Integration into eval_retargeting.py
- `Args` gains `metrics: str = "smoothness,effort,tracking,dynamics"` (comma list; each
  family toggleable) and an optional `dump_json: str | None` for per-task + aggregate.
- `RetargetingEvaluator` extracts the arrays each family needs and calls `compute_*`,
  merging the returned dicts into the existing per-task scalar dict.
- Existing print of mean/std per key is unchanged; new keys appear automatically.

## Testing
- `tests/test_metrics_smoothness.py`, `_effort.py`, `_tracking.py`, `_dynamics.py`:
  unit tests with synthetic inputs and closed-form expected values:
  - smoothness: constant-velocity trajectory -> accel/jerk ≈ 0; pure sinusoid -> known
    accel amplitude.
  - effort: qpos exactly at a limit -> margin 0, saturation 1; mid-range -> margin 0.5.
  - tracking: robot == ref -> all errors 0; constant offset d -> mpjpe_global = |d|,
    mpjpe_root_rel = 0.
  - dynamics: free-fall CoM (c = -0.5 g t²) -> c_ddot ≈ -g (small residual vs ref);
    static pose -> ang_momentum_rms ≈ 0.
- Refactor `test_temporal_metric.py` and `test_centroidal_metric.py` to import the shared
  `compute_smoothness` / `compute_dynamics` instead of inline formulas; keep their A/B
  assertions (off vs on) intact.

## Out of scope (v1)
- Multi-config ablation orchestration (user chose single-config report).
- Matching a specific SOTA paper's exact metric definitions.
- ZMP / support-polygon margin.
- Torque / inverse-dynamics effort proxy.
