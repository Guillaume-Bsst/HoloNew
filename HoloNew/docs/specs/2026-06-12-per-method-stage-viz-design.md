# Per-method preprocessing-stage visualization (sub-project 2) — design

**Date:** 2026-06-12
**Status:** Design — pending implementation plan
**Scope:** `modules/01_retargeting/HoloNew` (the HoloNew package)
**Builds on:** sub-project 1 (three autonomous solver folders) and the multi-robot Viewer.

## Goal

In the annex `view_stages` viser app, let the user pick a **method** (holosoma /
GMR-SOCP v1 / GMR-SOCP v2) and a **stage** of that method's pipeline — the raw
source through every preprocessing step to the solved robot — and scrub it over
time. For each method you visualize its source and all its resolution steps.

## Decisions (from brainstorming)

1. **Two dropdowns: Method + Stage.** The Method dropdown selects holosoma / GMR v1 /
   GMR v2. The Stage dropdown lists that method's ordered stages plus a final `Robot`
   entry. Changing the Method refreshes the Stage options.
2. **Show only the current selection.** When a (method, stage) is selected, only that
   one thing is drawn; the other methods' robots and skeletons are hidden.
3. **Per-method data bundle** drives the viewer: each method provides its `qpos`
   (robot trajectory) and its `stages` (skeleton point arrays per stage name).
4. **holosoma stages** come from a new `compute_holosoma_stages(...)` in
   `src/holosoma/preprocess.py` that reproduces holosoma's preprocessing steps
   (scale + ground + map) as skeleton arrays. The native retargeter's qpos is
   **untouched** (parity preserved).
5. **GMR stages** come from the full `compute_stages` dict (`{mapped, scaled, offset,
   ground}`), which each GMR `from_config` now stores on the retargeter
   (`rt.gmr_stages`) in addition to the `ground` it already uses for the solve.
6. **Points only** for skeletons (no bones) — colored point clouds. Bones are out of
   scope. Object/climbing-task stages are out of scope (robot_only demo).

## Stage lists per method

- **holosoma:** `Original` (raw 52-joint human) → `Scaled` (smpl_scale) → `Ground`
  (grounded) → `Mapped` (the matched-joint subset) → `Robot` (native SOCP qpos).
- **GMR-SOCP v1 / v2:** `Original` → `Mapped` → `Scaled` (morphological) → `Offset`
  (table1 offsets) → `Ground` → `Robot` (the GMR qpos).

(The stage names overlap but mean different things per method — each method owns its
own ordered list; that is the point of the per-method registry.)

## Architecture

### `MethodViz` bundle (data model)
A small dataclass carrying everything the viewer needs for one method:
```
MethodViz:
    label: str                    # "holosoma" | "GMR-SOCP v1" | "GMR-SOCP v2"
    robot_key: str                # "holosoma" | "gmr_socp_v1" | "gmr_socp_v2"
    qpos: np.ndarray              # (T, 7+dof) robot trajectory
    stages: dict[str, np.ndarray] # ordered {stage_label: (T, B, 3)} skeleton points
```

### Per-method registry (`src/stages.py`, restructured)
Replace the flat `STAGE_SPECS` with a per-method structure declaring, for each method,
its `label`, `robot_key`, and its ordered list of stage labels (skeletons) followed by
the implicit `Robot` stage. Helpers expose: the method labels, the stage labels for a
given method, and whether a (method, stage) is the robot or a skeleton.

### Viewer (`src/viewer.py`)
- A new `bind_methods(methods: list[MethodViz])` builds: a Frame slider (max = min T
  across methods), a **Method** dropdown, and a **Stage** dropdown (options from the
  selected method's stage list + `Robot`).
- `_redraw(frame)`: read the selected method + stage. If stage is `Robot`, draw that
  method's `qpos[frame]` on its robot instance (`/world/robot_<robot_key>`) and hide
  all other robots + clear the skeleton layer. If stage is a skeleton, draw
  `method.stages[stage][frame]` as a colored point cloud and hide all robots.
- On Method change, repopulate the Stage dropdown (reset to the method's first stage or
  `Robot`).
- Robots are created one per method (`stage_keys = [m.robot_key for m in methods]`),
  reusing the existing multi-robot support.
- The existing `bind(result, extra_qpos=...)` may be kept or replaced; `view_stages` is
  the only caller and moves to `bind_methods`.

### Stage producers
- `src/holosoma/preprocess.py`: add `compute_holosoma_stages(raw_joints, scale_args,
  mapped_indices) -> {"Original": (T,52,3), "Scaled": (T,52,3), "Ground": (T,52,3),
  "Mapped": (T,B,3)}`, reproducing the scale (`calculate_scale_factor`/`smpl_scale`) and
  ground steps holosoma applies, then the mapped-joint selection. No change to the
  retargeter's solve.
- `src/gmr_socp_v1/gmr_socp_v1.py` and `v2`: store `rt.gmr_stages = compute_stages(...)`
  (the full dict) in `from_config`; keep `rt.gmr_ground = rt.gmr_stages["ground"]` for
  the solve. Expose the mapped-body skeletons per stage (each `(T, 14, 3)`).

### `examples/view_stages.py`
Build three `MethodViz`:
- holosoma: `qpos = run_headless(cfg).qpos`; `stages = compute_holosoma_stages(...)`.
- gmr v1 / v2: run each retargeter; `qpos = result.qpos`; `stages` from its
  `gmr_stages` (pos arrays per stage). Then `viewer.bind_methods([...])`.

## Data flow
1. Native retarget → holosoma qpos; `compute_holosoma_stages` → holosoma skeletons.
2. Each GMR retarget → its qpos + `gmr_stages` skeletons.
3. `view_stages` assembles 3 `MethodViz`, calls `viewer.bind_methods`.
4. The viewer's Method+Stage+Frame selection draws exactly one thing.

## Testing
- `compute_holosoma_stages` returns the four stages with shapes `(T,52,3)` /
  `(T,52,3)` / `(T,52,3)` / `(T,B,3)` on the demo sequence (gated on what it needs).
- GMR `from_config` exposes `rt.gmr_stages` with all four stage keys, each `(T,14,3)`.
- The per-method registry: method labels + per-method stage labels are correct.
- Viewer: `bind_methods` builds the dropdowns and `_redraw` draws a skeleton stage and
  a Robot stage without error (construct + draw test).
- Bounded headless smoke of `view_stages` (opens, scrubs, no Traceback).
- Existing parity/golden/full suite unchanged (native qpos untouched).

## Out of scope
- Bones for skeletons (points only).
- Object/climbing-task stages (only robot_only demo).
- Per-subject scaling beyond the demo defaults.

## Open items for the plan
- The exact `compute_holosoma_stages` reproduction of holosoma's scale/ground (reuse
  `calculate_scale_factor` + the grounding holosoma uses in `preprocess_motion_data`;
  verify the produced `Ground`/`Mapped` match what the native retarget actually sees).
- Whether to keep the old flat `bind`/`STAGE_SPECS` for backward compatibility or remove
  them (only `view_stages` uses them — likely remove and update its callers/tests).
- Coloring scheme for skeleton points (single color per stage, or per-segment).
- The viser dropdown-refresh API for repopulating Stage options on Method change.
