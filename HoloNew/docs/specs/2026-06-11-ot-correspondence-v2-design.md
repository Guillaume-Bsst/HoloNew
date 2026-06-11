# OT human→G1 correspondence in v2 (increment 4a) — design

**Date:** 2026-06-11
**Status:** Design — pending implementation plan
**Scope:** `modules/01_retargeting/HoloNew` (the HoloNew package)
**Builds on:** the GMR-SOCP retargeter (`gmr_socp_v2.py`) and the modular viewer.

## Goal

Bring test_pipe's **optimal-transport human→G1 surface correspondence** into HoloNew,
attached to the **v2** GMR-SOCP retargeter, and visualize it in a **dedicated
("annex") viser app** like test_pipe's original pipe. The correspondence is
**computed/loaded and displayed only — never used in the solve** yet.

This is the first of two contact/correspondence increments. **Contact fields
(`coal`) are a separate later increment (4b)** and are out of scope here.

## Decisions (from brainstorming)

1. **Faithful port** of test_pipe's `transport/` modules + the SMPL-X human body
   builder (`human/body.py`), imports adapted `test_pipe_retargeting` → `HoloNew`.
2. **Dependency:** add `ot` (POT, Python Optimal Transport) to the `holonew`
   conda env and the HoloNew installer. (`coal` is NOT needed for OT; it belongs
   to increment 4b.)
3. **SMPL-X model is NOT bundled** (104 MB > GitHub's 100 MB limit; license
   forbids redistribution). The SMPL-X model directory is a **configurable path**
   defaulting to the one already on this machine
   (`data/00_raw_datasets/models/models_smplx_v1_1/models/smplx`).
4. **Bundle the precomputed correspondence `.npz`** (small) in HoloNew so loading
   and using the OT correspondence is turnkey for other users; only the
   human-surface *visualization* requires SMPL-X.
5. **Dedicated annex viser app** for OT visualization (not an overlay bolted into
   `view_stages`). It degrades gracefully: G1 surface points colored by
   correspondence are always available (from the bundled `.npz`); the human
   surface mesh is drawn only when a SMPL-X model path is available.
6. **v2 only.** v1 stays a pure GMR retargeter. `retarget()` is unchanged — the
   correspondence is attached as data, not consumed by the cvxpy solve.

## Background

test_pipe's correspondence is a **static** human↔G1 surface map (depends on body
shape + the G1 URDF, not on motion). Pipeline (`transport/build_correspondence.build_table`):
build the SMPL-X `HumanBody`, sample a human surface point cloud (`human_source`),
sample the G1 surface per link (`g1_surface.sample_g1_surface`), couple them with
optimal transport (`ot_couple.couple`, POT), and produce a `CorrespondenceTable`
(per G1 surface point: its link, local offset, the human point it maps to, segment
labels). It is saved/loaded as a `.npz` (`save_correspondence`/`load_correspondence`).

## Architecture

New package `src/correspondence/` (additive; native + v1 untouched):

- Ported, imports adapted, kept close to the originals:
  - `ot_couple.py` — OT coupling (`couple(src, tgt, reg)`), needs `ot`.
  - `g1_surface.py` — `sample_g1_surface(urdf, density)`, `build_rest_cfg(urdf)`.
  - `human_source.py` — `build_human_source(body, density)`.
  - `segments.py`, `quality.py` — segment labels + coupling quality metrics.
  - `build_correspondence.py` — `CorrespondenceTable`, `build_table`,
    `save_correspondence`, `load_correspondence`, and a `main()` CLI to build the
    cache offline.
  - `human_body.py` — port of test_pipe `human/body.py` (`HumanBody`: SMPL-X mesh,
    rest verts, point-cloud cache). Needs `smplx` (present) + the SMPL-X model dir.
  - `viz.py` — port of `transport/viz.py` drawing helpers, adapted to the annex app.
- `src/config_values/` (or the existing config): a default SMPL-X model dir and the
  default G1 URDF, both overridable.

### Assets bundled in HoloNew
- `HoloNew/assets/correspondence/corr_neutral.npz` — the precomputed neutral
  correspondence (built here with SMPL-X available). Small; turnkey.
- **Not** bundled: the SMPL-X model files (external, configurable path).

### v2 integration
- `GmrSocpRetargeterV2.from_config(cfg)` gains correspondence loading:
  - if the bundled/cached `.npz` exists → `load_correspondence`;
  - else, if a SMPL-X model dir is configured → `build_table(...)` live and cache it;
  - else → leave `self.correspondence = None` (the retarget still runs).
  Store `self.correspondence: CorrespondenceTable | None`. `retarget()` unchanged.

### Annex viser app
- `examples/view_correspondence.py` — a standalone viser app (like test_pipe's pipe):
  loads/builds the correspondence, shows the G1 (at rest or a chosen frame from a
  v2 retarget) with its surface points colored by their human correspondence
  (segment- or index-based coloring), and — only if a SMPL-X path is available —
  the human surface mesh colored consistently. Opens its own viser server.

## Dependencies & installer

- Add `ot` to HoloNew's `pyproject.toml`/`setup.py` dependencies and install it into
  the `holonew` env. Update `scripts/setup_retargeting.sh` only if needed (the
  editable install picks up the new dep on reinstall).
- `smplx`, `trimesh`, `yourdfpy`, `scipy`, `igl` are already present.

## Data flow

1. (Offline, once, here) `build_correspondence.main` builds `corr_neutral.npz` from
   the SMPL-X neutral model + G1 URDF → bundled in `assets/correspondence/`.
2. `GmrSocpRetargeterV2.from_config` loads it → `self.correspondence`.
3. `examples/view_correspondence.py` draws the correspondence (G1 points always;
   human surface if SMPL-X available).

## Testing

- **Unit (no heavy deps where possible):**
  - `ot_couple.couple` on a tiny synthetic source/target → returns a valid index map.
  - `save_correspondence`/`load_correspondence` round-trip a `CorrespondenceTable`.
  - `g1_surface.sample_g1_surface` on the G1 URDF → points with valid `link_idx`
    into `link_names`.
- **Integration (needs SMPL-X locally):**
  - `build_table` on neutral SMPL-X + G1 URDF → a `CorrespondenceTable` with
    `human_idx` in range and `link_idx` valid. Marked to skip if the SMPL-X path is
    absent (so CI without the model still passes).
  - `GmrSocpRetargeterV2.from_config` loads the bundled `.npz` and exposes a valid
    `self.correspondence`; v1 has no such attribute / stays unchanged.
- **Annex app:** a bounded headless smoke test that it builds the scene and opens
  without a Traceback (human surface skipped when SMPL-X absent).

## Out of scope (later)

- Contact fields (`coal`) — increment 4b.
- Using the correspondence in v2's solve (objective/constraints) — a later increment.
- Per-subject (non-neutral) correspondences / betas from OMOMO `.p` files.

## Open items for the plan

- Confirm `smplx.create` loads the neutral `.npz` from the configured dir (it does
  in test_pipe); pin the exact model-dir layout the loader expects.
- The G1 URDF used for surface sampling (`models/g1/g1_29dof.urdf`) vs the mujoco
  body names used by v2 — the correspondence uses URDF link frames; ensure the
  annex viz poses G1 points consistently with the viewer's robot.
- Coloring scheme for the annex viz (segment-based vs human-index-based) — pick one
  in the plan; segment-based is most legible.
- Exact bundled cache path + how `from_config` resolves it (package-relative).
