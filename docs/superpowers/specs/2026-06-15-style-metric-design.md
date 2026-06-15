# Style Metric — Design Spec

Date: 2026-06-15
Status: approved (design + decomposition), implementation pending
Parent: retargeting scoreboard (sub-project 1 of 3: style, then contacts, then root/sanity)

## Goal

An objective **style-replication** metric — the criterion on which we aim to beat
Holosoma. "Style" = how faithfully the result reproduces the source's body posture
and limb configuration, **independent of global placement (heading + translation)**.

Two complementary, heading-invariant sub-metrics, each in the **pelvis frame**:
- **orientation fidelity** — do the limbs adopt the same orientation (twist/bend)?
- **shape fidelity** — are the limbs in the same configuration (where the keypoints sit)?

Scored against two references (two columns):
- **vs SMPL** = vs `rt.gmr_ground` (the source motion grounded onto the robot skeleton
  by GMR). This is the shared source target — the head-to-head reference for Holosoma.
- **vs GMR** = vs a GMR-baseline solved trajectory (internal non-regression column).

## Key architectural fact

Reference orientations come from `rt.gmr_ground["quat"]` (per-robot-link world
quaternions, wxyz) and positions from `rt.gmr_ground["pos"]`. These are NOT in the
result npz, so style needs a **reference context** rebuilt via `from_config(source)`
— heavier than the pure-npz sweep, but the only way to get reference orientations,
and it works for ANY method's output trajectory (ours or Holosoma's).

## Architecture

```
evaluation/
  reference_context.py   ReferenceContext: wraps an rt, exposes gmr reference arrays
                         + fk_links(qpos) for the tracked bodies; build via from_rt or from_config
  metrics/style.py       compute_style(R, p, R_ref, p_ref, pelvis_idx, tracked) -> dict (pure)
```

Data flow:
`from_config(source)` -> `ReferenceContext` (gmr_pos/gmr_quat, tracked-body order,
pelvis index) -> `fk_links(method_qpos)` returns method link rotations/positions in the
same body order -> `compute_style(...)` vs the gmr reference (vs SMPL) and, when a GMR
baseline qpos is supplied, vs its FK (vs GMR).

The pure function stays backend-agnostic; the reference context does all extraction.
The in-process A/B test (`test_style_metric`) is a real consumer (uses a live rt).

## Metric definitions (pure, in evaluation/metrics/style.py)

Inputs per frame t, link k: solved world rotation `R[t,k]` (3,3) and position
`p[t,k]` (3,); reference `R_ref`, `p_ref`; `pelvis_idx`; boolean `tracked` (K,) mask
of non-pelvis tracked links. Let B = pelvis index.

**Orientation fidelity** (rad) — identical to the existing `_pelvis_relative_fidelity`:
```
R~[t,k]     = R[t,B]^T  @ R[t,k]            # pelvis-relative, solved
R~_ref[t,k] = R_ref[t,B]^T @ R_ref[t,k]     # pelvis-relative, reference
err[t,k]    = || logSO3( R~[t,k]^T @ R~_ref[t,k] ) ||      (rotvec norm)
style_orient_err = mean over t and tracked k
```

**Shape fidelity** (m) — heading-invariant pelvis-frame keypoint error:
```
q[t,k]     = R[t,B]^T  @ (p[t,k]   - p[t,B])      # solved, in pelvis frame
q_ref[t,k] = R_ref[t,B]^T @ (p_ref[t,k] - p_ref[t,B])
style_shape_err = mean over t and tracked k of || q[t,k] - q_ref[t,k] ||
```
Distinct from the scoreboard's `mpjpe_root_rel` (translation-only, heading-dependent):
shape additionally rotates into the pelvis frame, so a pure yaw of the whole body
yields zero shape error — that is what makes it a *style* metric.

Returns `{"style_orient_err": float, "style_shape_err": float}`.

## reference_context.py

`ReferenceContext`:
- `gmr_pos` (Tf, K, 3), `gmr_quat` (Tf, K, 4 wxyz) — reference per tracked link, in a
  fixed body order derived from `IK_MATCH_TABLE1` / `rt.robot_link_names` (mirrors the
  mapping in the current `_pelvis_relative_fidelity`).
- `body_order: list[str]`, `pelvis_idx: int`, `tracked: np.ndarray` (bool, K).
- `fk_links(qpos) -> (R, p)` with `R` (T, K, 3, 3), `p` (T, K, 3): reconstruct the full
  config per frame (`q_init_full` overwritten by the qpos row) and call
  `rt.body_rotation` / `rt.body_position` for each body in `body_order`.
- `reference_RP() -> (R_ref, p_ref)`: gmr_quat -> rotation matrices + gmr_pos, truncated
  to T and the body order.
- Constructors: `from_rt(rt)` and `from_config(task_type, task_name, data_format)`.
- `score_style(method_qpos, gmr_baseline_qpos=None) -> dict`: FK the method qpos, call
  `compute_style` vs the gmr reference -> `style_orient_vs_smpl`, `style_shape_vs_smpl`;
  when `gmr_baseline_qpos` given, FK it and call again -> `*_vs_gmr`.

## Testing

- `tests/test_metrics_style.py` (pure, synthetic):
  - identical solved == reference -> both errors 0.
  - apply a fixed extra rotation `Rd` to every non-pelvis link's solved orientation ->
    `style_orient_err == angle(Rd)`, shape unchanged.
  - apply a global yaw to the WHOLE body (pelvis included, rotate R and p about z) ->
    both errors ~0 (heading invariance — the defining property).
- Refactor `tests/test_style_metric.py`: replace the inline `_pelvis_relative_fidelity`
  with `ReferenceContext.from_rt(rt)` + `compute_style(...)["style_orient_err"]`. Keep
  the A/B assertion (Style activate_ws on <= world tracking + 0.05 slack). Single source
  of truth for the orientation formula.

## Out of scope (this sub-project)
- Contacts metric (timing/placement/no-slip) — sub-project 2.
- Root/object pose sanity metrics — sub-project 3.
- A CLI to score an external (Holosoma) trajectory — thin follow-up once the reference
  context exists; `score_style` already accepts any qpos.
