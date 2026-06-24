# Object grounding in the `ground` stage — single source for interaction

Date: 2026-06-24
Status: approved (brainstorm), pending plan

## Problem

The object's floor-grounding z-shift (HODome) is computed ad-hoc inside the SDF
build block (`builder.py:462-469`) and applied to only **one** of the four object-pose
copies the pipeline maintains. The result is that the same object lives at different
z depending on which consumer reads it.

The four current object-pose preparations, all derived from `_obj_poses_all`
(raw loader / `.pt` poses, `[qw,qx,qy,qz,x,y,z]`):

| Copy | Built | XY/Z scale | HODome ground shift | Consumers |
|---|---|---|---|---|
| `rt._obj_poses_mj` | `builder.py:367-376` | yes | **no** | MuJoCo free-joint drive (`q[-7:]`), non-penetration |
| `rt._obj_poses_raw` | `builder.py:447-470` | yes | **yes** (`_obj_ground_shift`) | `smplx_ground_probe` (SDF), object surface point cloud, per-frame `obj_pose_ref`, movable reference motion, object↔floor inertia |
| `view_stages.object_pose_raw` / `object_pose_scaled` | `view_stages.py:380-417` | scaled variant only | **no** | viewer stage display (independent of the solve) |
| `solved_obj_pose` | in the solve | — | inherited via warm-start from `_obj_poses_raw` | movable output (`_obj_solved_poses`) |

Only `_obj_poses_raw` gets the shift. In HODome this means the **driven/displayed**
object (MuJoCo + viewer) floats relative to the human and the SDF/interaction object
by `_obj_ground_shift`.

The grounding shift is **not** in the solved object pose today — it is on the
*reference* `_obj_poses_raw`, which then warm-starts the movable solve. What is
missing is that the object grounding is not part of the `ground` *stage*
(`gmr_stages["ground"]`, currently human-only), so there is no single grounded
scene that all interaction consumers share.

## Goal

Make the **`ground` stage** carry the grounded object pose, and make that the single
source of truth for every interaction consumer (SDF probe, object surface point cloud,
per-frame reference, movable reference, MuJoCo drive, viewer). Align the currently
inconsistent copies onto it.

Scope decisions (from brainstorming):
- **Single source + consistency**: unify all consumers AND align `_obj_poses_mj`
  (the MuJoCo drive) onto the grounded scene. This intentionally **changes HODome
  behaviour** (the driven/displayed object z moves so it no longer floats).
- **Structure**: extend `gmr_stages` (store the grounded object in the stage), not a
  separate side field.

## Design

### 1. Data structure — the single source

`gmr_stages["ground"]` gains a key `"object_pose"` = `(T, 7)` `[qw,qx,qy,qz,x,y,z]`,
the **scaled + grounded** object pose. Since `rt.gmr_ground` is the same dict object
as `rt.gmr_stages["ground"]`, it is reachable as `rt.gmr_ground["object_pose"]`.

This is additive and non-breaking: existing `["pos"]` / `["quat"]` access on stage
dicts is untouched. The key is present only when the task has an object
(`object_sdf is not None and _obj_poses_all is not None`); otherwise absent / `None`.

Only the `ground` stage gets `object_pose`. The viewer's other stages (raw/scaled)
keep their existing independent object-pose computation.

### 2. Computation & ordering (builder)

- `compute_stages` stays **human-only** (pure function in `preprocess.py`).
- New helper, e.g. `ground_object_pose(obj_poses_scaled, object_surface_local, dataset)
  -> (poses_grounded (T,7), shift)`, encapsulating exactly the current
  `builder.py:462-469` logic: sample up to ~60 frames, transform `object_surface_local`
  by each frame's pose, take the min z over the clip, `shift = -min_z`, **gated to
  `dataset == "hodome"`** (else `shift = 0.0`). XY/Z object scaling (`_obj_xy`/`_obj_z`)
  is applied once when producing the scaled poses fed to this helper.
- Ordering: the shift needs `object_surface_local` (sampled at `builder.py:432`, after
  `compute_stages` at `:331`). Move the surface sampling + the `ground_object_pose`
  call to **right after `compute_stages`**, and write the result into
  `gmr_stages["ground"]["object_pose"]`. This must happen **before** `_obj_poses_mj`
  is built (`:367`) so the MuJoCo drive can derive from the grounded stage.
  - `sample_object_surface(_mesh_file)` depends only on `_mesh_file` (resolved at
    `:329`, before `compute_stages`), **not** on `rt.object_sdf`. The current surface
    sampling is gated behind `rt.object_sdf is not None` (`:429`); decouple that guard
    so the surface (hence the grounded object pose) is available early. The SDF build
    itself can stay where it is.

### 3. Consumers rewired onto the single source

| Consumer | Before | After |
|---|---|---|
| `rt._obj_poses_raw` (SDF probe, `obj_pose_ref`, movable ref, object↔floor inertia) | ad-hoc, with shift | **alias of** `gmr_stages["ground"]["object_pose"]` |
| `rt._obj_poses_mj` (MuJoCo drive `q[-7:]`) | scaled, **no** shift | `convert_object_poses_to_mujoco_order(gmr_stages["ground"]["object_pose"])` → **gains the shift** |
| `smplx_ground_probe` (object-local SDF transform) | `_obj_poses_arg` ad-hoc | same single source |
| `view_stages` Grounded/Floor stage object | independent copy, no shift | reads `gmr_stages["ground"]["object_pose"]` |

### 4. Gating & behaviour change

- Object ground shift stays **HODome only** (OMOMO objects are floor-consistent and
  golden-locked).
- **OMOMO byte-identical**: `_obj_poses_raw` value unchanged (shift was already 0);
  aligning `_obj_poses_mj` is a no-op (shift 0).
- **HODome changes (accepted)**: `_obj_poses_mj` and the viewer display gain the shift
  → driven/displayed object no longer floats vs. human/floor.

### 5. Viewer + tests

- `view_stages.py`: the Grounded/Floor stage object reads the grounded object from the
  stage instead of its local copy. The raw/scaled stages keep their current object
  computation.
- Tests:
  - HODome: the stage's grounded object rests its lowest surface point at z ≈ 0
    (object-side mirror of `test_floor_offset`).
  - `_obj_poses_mj` equals the MuJoCo-order conversion of the grounded stage object.
  - OMOMO unchanged (regression guard on `_obj_poses_raw` / `_obj_poses_mj`).
  - Adapt `test_view_stages_grounding` if needed so the displayed object matches the
    grounded human.

## Minor decisions (defaults taken)

- **Keep the name `_obj_poses_raw`** (now an alias of the grounded stage object) to
  limit churn, even though "raw" is misleading. Alternative considered: rename to
  `_obj_poses_ground`. Chosen: keep, because `view_stages.py` slices it by name.
- **Only the `ground` stage gets `object_pose`** (per scope). Scaled/offset stages are
  not extended.

## Non-goals

- No change to the solve formulation (movable SQP variable, D/X/P terms) beyond the
  reference pose now coming from the grounded stage (numerically identical to today
  for the reference path).
- No generalisation of object grounding beyond HODome.
- No rename/refactor unrelated to routing object poses through the grounded stage.
