# Three autonomous solver folders (restructure) — design

**Date:** 2026-06-12
**Status:** Design — pending implementation plan
**Scope:** `modules/01_retargeting/HoloNew` (the HoloNew package)
**This is sub-project 1 of 2.** Sub-project 2 (per-method preprocessing-stage
visualization in viser) builds on this and is specified separately.

## Goal

Restructure the three retargeting methods into three self-contained, autonomous
solver folders — `src/holosoma/`, `src/gmr_socp_v1/`, `src/gmr_socp_v2/` — each
owning its solver, its preprocessing, and (for the GMR methods) its own copy of
the GMR tables/targets. This is a **behaviour-preserving refactor**: every method's
output (qpos) is unchanged, guarded by the existing parity and golden tests.

## Decisions (from brainstorming)

1. **Three folders**, one per solver: `src/holosoma/`, `src/gmr_socp_v1/`,
   `src/gmr_socp_v2/`.
2. **Full autonomy** for each solver folder: it owns its solver + its `preprocess.py`
   + (GMR) its own copy of `tables.py` and `targets.py`. No sharing of solver-specific
   code between the GMR versions — v2 can later diverge without touching v1.
3. **Shared plumbing stays in `src/`** (not solver-specific): the viewer
   (`viewer.py`), the stage registry (`stages.py`), `retarget_result.py`, the data
   loaders / config (`create_task_constants`, `load_motion_data`, `config_types/`),
   and the v2 data layers (`correspondence/`, `contact/`).
4. **Native holosoma moves into `src/holosoma/`**, content byte-identical (only its
   location and import paths change). The native qpos output must stay identical
   (parity test against upstream holosoma + the golden test guard this).
5. **Three preprocessing files** are created here as part of the structure
   (`holosoma/preprocess.py`, `gmr_socp_v1/preprocess.py`, `gmr_socp_v2/preprocess.py`).
   They *own* each method's preprocessing logic; exposing their stages to the viewer
   is sub-project 2.
6. **Decouple GMR from holosoma's preprocessing.** After the GMR preprocessing fix,
   GMR no longer needs `preprocess_motion_data` / `initialize_robot_pose`: each GMR
   `from_config` uses its own `compute_stages` plus a minimal init (base at the
   frame-0 pelvis target, joints zero). The generic `.pt` loading and
   `create_task_constants` remain shared (not solver-specific). The GMR-vs-mink parity
   test guards that the qpos is unchanged.

## Target structure

```
src/
  holosoma/
    interaction_mesh_retargeter.py     # moved, content unchanged
    preprocess.py                      # NEW: holosoma preprocessing (smpl_scale + ground)
    <holosoma-private support>         # mujoco_utils, viser_utils, interaction-mesh helpers
  gmr_socp_v1/
    gmr_socp_v1.py                     # moved from src/gmr_socp/
    preprocess.py                      # compute_stages (copy)
    tables.py  targets.py              # copies (autonomous)
  gmr_socp_v2/
    gmr_socp_v2.py
    preprocess.py  tables.py  targets.py   # copies (autonomous; may diverge later)
  # shared plumbing (unchanged location):
  viewer.py  stages.py  retarget_result.py
  correspondence/  contact/
  <shared loaders/config>              # create_task_constants, load_motion_data, config_types
examples/  tests/  assets/
```

The old `src/gmr_socp/` package is removed once its contents are split into the two
GMR folders.

## What moves where

- **`src/holosoma/`** receives: `interaction_mesh_retargeter.py`; a new
  `preprocess.py` owning holosoma's preprocessing (the scale/ground logic currently in
  `utils.py:preprocess_motion_data` + `calculate_scale_factor`); and the holosoma-only
  support the retargeter imports (`mujoco_utils.py`, `viser_utils.py`, and the
  interaction-mesh helpers from `utils.py` — `create_interaction_mesh`,
  `calculate_laplacian_*`, `get_adjacency_list`, `transform_points_*`). The native
  retargeter's internal `sys.path`/relative imports are updated to its new location.
- **`src/gmr_socp_v1/` and `src/gmr_socp_v2/`** each receive a full copy of the GMR
  module set: the retargeter, `preprocess.py` (compute_stages), `tables.py`, `targets.py`.
  Their cross-package imports become folder-local (`from .tables import ...`, etc.).
- **Stays shared in `src/`**: `viewer.py`, `stages.py`, `retarget_result.py`,
  `correspondence/`, `contact/`, and the generic loading/config used by every method.

The exact membership of individual `utils.py` functions (holosoma-specific vs generic
loader/config) is enumerated in the implementation plan; the rule is: anything only the
holosoma retargeter uses → `holosoma/`; anything generic (data loading, task constants,
config types) → shared.

## Import updates

Updating import paths is mechanical but wide-reaching:
- `HoloNew.src.interaction_mesh_retargeter` → `HoloNew.src.holosoma.interaction_mesh_retargeter`
- `HoloNew.src.gmr_socp.gmr_socp_v1` → `HoloNew.src.gmr_socp_v1.gmr_socp_v1` (and v2)
- `HoloNew.src.gmr_socp.{tables,targets,preprocess}` → the per-folder copies.
Affected: `examples/robot_retarget.py`, `examples/view_stages.py`,
`examples/view_correspondence.py` (no), `src/contact/*` and `src/correspondence/*`
(only where they reference gmr_socp), and the `tests/`.

## Testing / guarantees

This is a behaviour-preserving refactor. The safety net is the existing suite:
- **Native parity** (`test_parity_native_vs_holosoma.py`): native qpos byte-identical
  to upstream holosoma — must still pass after the move.
- **GMR-vs-mink parity** (`test_parity_gmr_socp_vs_mink.py`): base-pos RMSE < 0.1 — must
  still pass after the decoupling/minimal-init.
- **Golden** (`test_retarget_golden.py`), plus the full suite (40 tests) — all green.
- Test imports are updated to the new module paths.

No new tests are required for the move itself beyond keeping the suite green; the
preprocess modules' stage outputs are tested in sub-project 2 when wired to the viewer.

## Out of scope (sub-project 2)

- Exposing each method's preprocessing stages to the viewer.
- The Method + Stage dropdowns and per-method stage skeletons.

## Open items for the plan

- Exact per-function split of `utils.py` (holosoma-specific vs shared loader/config).
- Whether `gmr_socp_v1.py`/`gmr_socp_v2.py` keep their names or become `retargeter.py`
  inside their folders (decision: keep `gmr_socp_v1.py` / `gmr_socp_v2.py`).
- The minimal GMR `from_config` init once `preprocess_motion_data` /
  `initialize_robot_pose` are dropped (base from ground pelvis[0], joints zero) — verify
  the GMR-vs-mink parity still holds.
- Whether `examples/robot_retarget.py` (which holds shared loaders + the native CLI)
  is split so the shared loaders are importable without pulling the native example.
