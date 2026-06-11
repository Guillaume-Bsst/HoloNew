# Contact fields in v2 (increment 4b) — design

**Date:** 2026-06-11
**Status:** Design — pending implementation plan
**Scope:** `modules/01_retargeting/HoloNew` (the HoloNew package)
**Builds on:** the OT correspondence increment (4a) — reuses its SMPL-X `HumanBody`,
`PointCloudCache`, and the bundled correspondence — and the GMR-SOCP v2 retargeter.

## Goal

Bring test_pipe's **per-frame contact fields** (SDF method) into HoloNew, attached
to the **v2** GMR-SOCP retargeter, and visualize them in a dedicated **annex viser
app** with a frame slider. Contact fields are **computed/loaded and displayed only —
never used in the solve** yet. This is the second of the two contact/correspondence
increments (4a was OT correspondence).

## Decisions (from brainstorming)

1. **SDF method only.** Port test_pipe's SDF contact path: the object's signed-distance
   field is built **once offline (with `coal`)**; at runtime the per-frame contact is a
   numpy query of that SDF (`sdf_surface_field`) plus an analytic floor field. The exact
   `coal` per-frame mode (`backends/coal.py`) is NOT ported.
2. **Dependency:** add `coal` (3.0.3, manylinux pip wheels — no compilation) to the
   `holonew` env and the installer. It is needed **only** to build the object SDF
   offline; the runtime SDF query is numpy-only.
3. **Bundle two precomputed artifacts** (built here with coal + SMPL-X):
   - `assets/contact/largebox_sdf.npz` — the static object SDF for `largebox`.
   - `assets/contact/contact_sub3_largebox_003.npz` — the precomputed per-frame
     human-side contact fields for the demo sequence, so the demo viz is turnkey
     without SMPL-X. SMPL-X is needed only to compute contact for other sequences.
4. **Dedicated annex viser app** `examples/view_contact.py` with a frame slider.
5. **v2 only.** v1 stays a pure GMR retargeter. `retarget()` is unchanged — the contact
   fields are attached as data, not consumed by the solve.
6. **Contact is keyed 1:1 to the correspondence.** The human point-cloud cache used to
   sample contact is rebuilt from v2's loaded `CorrespondenceTable` (`tri_idx`/`bary`),
   so each human contact sample matches the same human point the OT map drives — the
   seam a future solve term plugs into.

## Background

test_pipe's `fields/` computes, per frame, contact channels on probe point sets.
`compute_contact_fields(T, quats, pelvises, human_faces, human_body_params,
human_pc_cache, object_mesh, object_grid_local, obj_poses, floor_grid, margin,
object_sdf=...)` poses the SMPL-X human each frame (via `HumanBody`) and, in **SDF
mode** (an `object_sdf` is supplied), returns the human-side channels
`{"human_floor", "human_object"}` stacked over T frames (each a `ContactField`).
`backends/sdf.py` provides `ObjectSDF`, `build_object_field(mesh, margin, resolution)`
(builds it — uses coal), `sdf_surface_field(probes, sdf, margin)` (numpy query →
`ContactField`), `save_object_sdf`/`load_object_sdf`, and a differentiable
`sdf_surface_distance_torch` (ported for future solve use, not wired now).

## Architecture

New package `src/contact/` (additive; native + v1 untouched), a faithful port of the
SDF path of test_pipe `fields/`, imports adapted to HoloNew:

- `contact_field.py` — `ContactField` dataclass + `_contains`, `_probe_distance`.
- `probes.py` — `make_object_grid`, `make_floor_grid`.
- `object_input.py` — `load_mesh`, `parse_obj_name`.
- `backends/__init__.py`, `backends/sdf.py` — `ObjectSDF`, `build_object_field`
  (coal, offline), `sdf_surface_field` (numpy), `band_points`, `save_object_sdf`,
  `load_object_sdf`, `sdf_surface_distance_torch`.
- `backends/floor.py` — analytic floor field.
- `combined.py` — `compute_contact_fields` (SDF mode) + `_init_frame_ctx`, `_frame_fields`.
- `constants.py` — `CONTACT_MARGIN_M`, `OBJECT_FIELD_RESOLUTION`, grid densities.
- Reuses `src/correspondence/human_body.py` (`HumanBody`, `PointCloudCache`).

`backends/coal.py` (exact per-frame mode) is intentionally NOT ported.

### Bundled assets
- `assets/contact/largebox_sdf.npz` — static object SDF (built offline with coal).
- `assets/contact/contact_sub3_largebox_003.npz` — per-frame demo contact fields.
- Not bundled: SMPL-X (external, configurable path from 4a's `constants.py`).

### v2 integration
- `GmrSocpRetargeterV2.from_config(cfg)` gains contact loading (after correspondence):
  - load `self.object_sdf = load_object_sdf(assets/contact/largebox_sdf.npz)` if present;
  - load `self.contact_fields` from the bundled per-sequence `.npz` if it matches the
    sequence; else, if SMPL-X is available, compute live via `compute_contact_fields`
    (rebuilding `human_pc_cache` from `self.correspondence.tri_idx/bary`); else leave
    `self.contact_fields = None`.
  - Store `self.object_sdf`, `self.contact_fields`. `retarget()` unchanged.

### Annex viser app
- `examples/view_contact.py` — standalone viser app with a frame slider. For the
  selected frame it draws the human-side contact probe points colored by the contact
  signal (e.g. `human_object` distance/active), loaded from the bundled demo `.npz`
  (no SMPL-X required). Opens its own viser server.

## Dependencies & installer
- Add `coal` to `pyproject.toml`/`setup.py` and install into the `holonew` env. Update
  the installer note if needed (editable reinstall picks up the dep). `trimesh`,
  `smplx`, `scipy` already present.

## Data flow
1. (Offline, once, here) build `largebox_sdf.npz` (`build_object_field`, coal) and the
   demo `contact_sub3_largebox_003.npz` (`compute_contact_fields`, SMPL-X) → bundled.
2. `GmrSocpRetargeterV2.from_config` loads the object SDF + the demo contact fields.
3. `examples/view_contact.py` draws the per-frame contact (from the bundled demo `.npz`).

## Testing
- **SDF backend:** `build_object_field` on the `largebox` mesh (coal) → an `ObjectSDF`;
  `sdf_surface_field` on a few probe points → a `ContactField` with finite distances;
  `save_object_sdf`/`load_object_sdf` round-trip.
- **compute_contact_fields (SDF mode):** on a tiny synthetic / the demo, asserting the
  returned dict has `human_floor`/`human_object` of shape `(T, N)`. Gated with `skipif`
  on SMPL-X presence for the live build; the bundled artifact path needs no SMPL-X.
- **v2:** `from_config` exposes `self.object_sdf` and `self.contact_fields` from the
  bundled artifacts; v1 has neither / stays unchanged; the native golden test still passes.
- **Annex app:** bounded headless smoke that it builds the scene and opens without a
  Traceback using the bundled demo contact field (no SMPL-X).

## Out of scope (later)
- The exact Coal per-frame mode (`backends/coal.py`).
- Using contact fields (or the differentiable SDF) in v2's solve.
- Contact for object/climbing tasks beyond the demo largebox sequence.

## Open items for the plan
- `ContactField`'s exact fields and which channel/value drives the viz coloring (pick
  one — likely the signed distance, clamped to the margin band).
- Bundled artifact sizes must stay < 100 MB (object SDF resolution and the per-frame
  contact arrays); lower the SDF resolution / probe density if needed.
- Confirm `coal` imports cleanly in the env after install (manylinux wheels) and that
  `build_object_field` runs on the `largebox` mesh.
- How `from_config` decides the per-sequence contact cache path / matches the sequence
  name; and the object mesh + `obj_poses` source for live computation.
