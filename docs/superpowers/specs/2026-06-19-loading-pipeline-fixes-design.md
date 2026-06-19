# Loading-pipeline fixes (#2–#5) — design

Date: 2026-06-19
Status: approved (design), pending implementation plan
Scope: data loading / preprocessing for TEST-SOCP. Findings #2–#5 from the loading
audit. Finding #1 (only `largebox` bundled) is subsumed by #2 and not a separate item.
Finding #6 (MJCF object inertia) is **explicitly out of scope** (dropped by request).

## Background

The loading audit surfaced six issues. Four are in scope here (#2–#5); #6 was dropped
by request. Two distinct loading paths exist:

- **`MotionLoader`** (`src/data_loaders/`) → unified contract for
  `examples/robot_retarget.main()`; also provides `object_source()`.
- **`build_from_config`** (`src/test_socp/builder.py`) → TEST-SOCP; reads motion
  directly (`targets.py`), uses `object_source()` only for the object.

Key facts established during the audit (must stay true):

- OMOMO object `.pt` poses place the object **centroid**. The bundled
  `models/<obj>/<obj>.obj` is already **recentred on its centroid + pre-scaled**
  (= captured unit mesh recentred × `obj_scale`). The captured
  `captured_objects/<obj>_cleaned_simplified.obj` is the **unit, off-origin** mesh.
- `captured_objects/` and HODome `scaned_object/` are present locally under the
  `path.yaml` roots, so both fallbacks are testable.
- Test env: `/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python`,
  run from `HoloNew/HoloNew/` (cwd). Commits carry **no** Claude trailer.

## Fix #2 + #3 — Shared, package-anchored OMOMO object-mesh resolver

**Problem.** The solver's `object_source` (`omomo.py:73`) only knows the bundled
`models/<obj>/<obj>.obj`, so only `largebox` gets an object SDF; every other OMOMO
object degrades to floor-only. The viewer (`view_stages.py:222-259`) already has the
broader logic (captured mesh + recenter + `obj_scale`) but duplicated. The bundled
path is also `Path("models")/...` — **CWD-relative** (#3).

**Design.** One function used by both solver and viewer:

```
resolve_omomo_object_mesh(seq_name, omomo_dir, cache_dir=None) -> Path | None
```

in `src/data_loaders/omomo.py`. Resolution order:

1. **Bundled** `<PKG>/models/<obj>/<obj>.obj`, where `<PKG>` is derived from
   `Path(__file__)` (package-anchored — fixes #3, no CWD dependence). Already
   centred + pre-scaled → returned as-is.
2. **Captured fallback** `omomo_dir/data/captured_objects/<obj>_cleaned_simplified.obj`:
   load, **recenter on centroid**, **× `obj_scale`** (`load_object_scale`), export a
   derived `.obj` to `cache_dir` (default `/tmp/holonew_omomo_meshes/<seq>.obj`,
   mirroring the HODome mesh cache), return the cache path. Covers all OMOMO objects.
3. Captured mesh present but `obj_scale` missing → **raise** a clear error (wrong size
   is worse than a crash); `object_source` catches it and degrades to floor-only with a
   warning (consistent with the solver's existing "object mesh not found" behaviour).
4. Nothing found → `None`.

`<obj>` is the 2nd `_`-token of `seq_name` (e.g. `sub3_largebox_003` → `largebox`).
The derived mesh is centroid-centred so it stays consistent with the `.pt` centroid
poses; obj poses are used as-is.

**Consumers.**
- `OmomoMixedLoader.object_source` replaces its bundled-only lookup with the resolver.
- `view_stages.py` drops the duplicated block (lines ~222-259) and calls the resolver,
  keeping its own staging/coloring around it.
- `calculate_scale_factor` (`preprocess.py:16`) path to `demo_data/height_dict.pkl`
  re-anchored to the package dir (#3); but it leaves the OMOMO canonical scale path
  per #4.

## Fix #4 — Canonical OMOMO scale = betas-FK (single source)

**Problem.** Two divergent scale computations: the builder uses
`calculate_scale_factor` (curated `height_dict.pkl` table, fallback 1.78), the loader
uses `omomo_height_from_betas` (SMPL-H FK on the subject betas). They disagree.

**Design.** Single canonical function in the OMOMO loader module:

```
omomo_scale_factor(seq, robot_height, omomo_dir, smplh_model_dir) -> float
```

= `robot_height / omomo_height_from_betas(load_human_metadata(seq).betas, gender, smplh_model_dir)`.

- Called by **both** `loader.load()` and `builder.build_from_config` (OMOMO branch).
- Builder obtains `smplh_model_dir` from `cfg.smpl_model_dir`, else
  `get_path("smplh_models")`, guarded by `.exists()`.
- Fallback chain when betas/model unavailable → `default_human_height` (1.78).
- `calculate_scale_factor` (height_dict) is **deprecated for OMOMO** (kept for viewer /
  legacy callers / tests only).

**Consequence (accepted).** OMOMO solve numerics shift → **golden tests must be
regenerated** (`demo_results/g1/**` and any golden fixtures). The implementation step
lists the impacted tests and regenerates under the conda env.

Scope note: only OMOMO has the dual-computation problem; AMASS/HODome already take a
single height from their processed npz, so they are untouched by #4.

## Fix #5 — Hand posing in the contact probe (OMOMO + HODome + AMASS)

**Corrected premise.** **Neither** posing path poses hands today. Both
`HumanBody.placed_verts` (OMOMO) and `placed_verts_smpl` (HODome/AMASS) compute
relative rotations for all joints but only pass `global_orient` + `body_pose` (the 22
body joints) to the SMPL-X forward (`human_body.py:100-107` and `:172-176`). Hands stay
at the model default. So the audit's "OMOMO already poses 52" was wrong; all three
sources need the fix.

**Verified joint layouts** (checked against the installed models):

- SMPL-X (55): body `0-21`, face `22-24`, left hand `25-39`, right hand `40-54`.
- SMPL-H (52, OMOMO `.pt`): body `0-21`, left hand `22-36`, right hand `37-51`.
- The two hand sub-trees are **identical MANO trees** shifted by +3 (SMPL-X inserts the
  3 face joints). The per-hand pose vectors are therefore **interchangeable** between
  SMPL-H and SMPL-X; only the parent tree used for the global→local conversion differs.

Verified parent arrays (to embed in the plan):

```
SMPL-X parents[:55] = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,
                       15,15,15, 20,25,26,20,28,29,20,31,32,20,34,35,20,37,38,
                       21,40,41,21,43,44,21,46,47,21,49,50,21,52,53]
SMPL-H parents[:52] = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,
                       20,22,23,20,25,26,20,28,29,20,31,32,20,34,35,
                       21,37,38,21,40,41,21,43,44,21,46,47,21,49,50]
```

**Design.**

1. **Shared forward in `HumanBody`.** Both `placed_verts` and `placed_verts_smpl`
   call one helper that, given per-joint global rotations, a parent tree, and the
   left/right hand slot ranges, computes relative rotations and calls the SMPL-X forward
   with `global_orient` + `body_pose` + `left_hand_pose` + `right_hand_pose`. The model
   is already built `use_pca=False`, so each hand pose is 15 joints × 3 = 45.
   - OMOMO path: parents = `SMPLH_PARENTS` (the verified 52-array, added as a module
     constant), hands at slots `22:37` / `37:52`.
   - SMPL-X path: parents = the model's native parents (55), hands at `25:40` / `40:55`.

2. **Prep emits SMPL-X-native orientations.** `prep_amass_smplx_for_rt` and
   `prep_hodome_processed` output `global_joint_orientations` as **(T, 55, 4)** in
   native SMPL-X order (generalize `compute_global_joint_orientations` to N joints with
   the full 55-parent tree). AMASS extracts left/right hand from the raw SMPL-X pose
   (joints `25-39` / `40-54`) instead of the current `[:52]` truncation (which mixed in
   face joints); HODome assembles the 55 axis-angle from
   `global_orient + body_pose + jaw/eyes + left_hand_pose + right_hand_pose` and applies
   the same Y→Z conjugation as the body. `global_joint_positions` stays (T, 22, 3) (GMR
   tables use body joints only); `load_smplx_to_smplh_layout` indexes only 0-21, so it
   is unaffected by the wider orientation array.

3. **Backward compatibility.** `placed_verts_smpl` accepts `(J, 4)` with `J ∈ {22, 55}`:
   `J == 22` → no hand poses passed (current behaviour, hands at default); `J >= 55` →
   hands posed. Old (T,22,4) npz/caches keep working; regenerate to enable hands.
   OMOMO needs **no** data change — its 52 SMPL-H quats already carry the hands.
   The hand **surface points already exist** in the point cloud (sampled on the full
   rest mesh); only their posing changes — no density change.

## Fix #6 — out of scope

Dropped by request. `hodome_scene.py`'s hardcoded mass/inertia is left untouched.

## Testing (TDD; conda env; cwd `HoloNew/HoloNew`)

- **#2/#3**: `object_source` returns the recentred+scaled captured mesh for a
  non-bundled OMOMO seq (`skipif` dataset absent); bundled stays preferred; resolver
  works from an arbitrary CWD; viewer and solver call the same function.
- **#4**: `omomo_scale_factor` betas-FK; builder == loader; golden regeneration.
- **#5**: `placed_verts` (OMOMO) and `placed_verts_smpl` (SMPL-X) move hand-region
  vertices vs the no-hand baseline; prep emits (T,55,4) orientations; a curled MANO
  pose visibly differs from the default. Cross-check the verified SMPL-H↔SMPL-X +3 hand
  reindex in a unit test.

**Golden impact.** Only `tests/golden/inertia_mode_qpos.npz` (TEST-SOCP builder path,
`object_interaction`, `sub3_largebox_003`, atol 1e-6) is affected — by both #4 (scale)
and #5 (probe surface near hands). It is regenerated, with a sanity check that the
change matches the intended cause. `tests/golden/baseline_qpos.npz`
(`test_retarget_golden`, GMR/holosoma `run_headless`, height_dict legacy path, no
probe) is **not** touched: `robot_retarget.load_motion_data` stays on
`calculate_scale_factor` as a documented legacy caller.

## Modules touched

`src/data_loaders/omomo.py`, `examples/view_stages.py`, `src/test_socp/builder.py`,
`src/holosoma/preprocess.py`, `data_utils/prep_amass_smplx_for_rt.py`,
`src/data_loaders/hodome.py`, `src/test_socp/correspondence/human_body.py`,
plus tests under `tests/`.
