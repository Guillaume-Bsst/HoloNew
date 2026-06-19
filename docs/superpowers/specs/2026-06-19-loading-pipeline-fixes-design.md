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

## Fix #5 — Hand posing for the `smpl_order` probe path (HODome + AMASS)

**Problem.** `HumanBody.placed_verts_smpl` (`human_body.py:145-183`) hardcodes the
22 body joints and leaves joints 22-51 (hands) at identity. The OMOMO path
(`placed_verts`) already poses all 52, but the `smpl_order` path (HODome + AMASS) does
not — so the contact-probe hands are collapsed to rest, which matters most for HODome
(human-object interaction / grasping).

**Design.**
1. Generalize `compute_global_joint_orientations` to **N joints** using the full SMPL-X
   parent tree (52), not just the 22-body subtree. The full parents come from the
   SMPL-X model / kintree.
2. `prep_amass_smplx_for_rt` and `prep_hodome_processed` emit
   `global_joint_orientations` as **(T, 52, 4)** (body + hands). AMASS already carries
   hands in `aa_rot_52[:, 22:]`; HODome carries them in `left/right_hand_pose`.
   `global_joint_positions` stays (T, 22, 3) (GMR tables use body joints only).
3. `HumanBody.placed_verts_smpl` accepts **up to 52** quats: fill the provided slots,
   identity for the rest (instead of fixing `[:22]`).
4. **Backward compatibility.** Existing npz/caches with 22 orientations keep working
   (hands fall back to identity); regenerate to enable hands. The hand **surface points
   already exist** in the cache (sampled on the full rest mesh); only their posing
   changes — no density change.

## Fix #6 — out of scope

Dropped by request. `hodome_scene.py`'s hardcoded mass/inertia is left untouched.

## Testing (TDD; conda env; cwd `HoloNew/HoloNew`)

- **#2/#3**: `object_source` returns the recentred+scaled captured mesh for a
  non-bundled OMOMO seq (`skipif` dataset absent); bundled stays preferred; resolver
  works from an arbitrary CWD; viewer and solver call the same function.
- **#4**: `omomo_scale_factor` betas-FK; builder == loader; golden regeneration.
- **#5**: `placed_verts_smpl` moves hand vertices when 52 quats are passed (vs
  identity); prep emits 52 orientations.

## Modules touched

`src/data_loaders/omomo.py`, `examples/view_stages.py`, `src/test_socp/builder.py`,
`src/holosoma/preprocess.py`, `data_utils/prep_amass_smplx_for_rt.py`,
`src/data_loaders/hodome.py`, `src/test_socp/correspondence/human_body.py`,
plus tests under `tests/`.
