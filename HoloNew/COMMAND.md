# HoloNew — Command Cheat Sheet

Quick reference for everything added on top of upstream holosoma: the three
retargeting solvers, the annex viewers, the OT-correspondence / contact tooling,
and the parity tests.

For the upstream pipeline (single/batch retargeting, data prep, evaluation, data
conversion) see [README.md](README.md). This file only covers the local additions.

## 0. Setup

```bash
# Activate the isolated env (defaults to the "holonew" conda env)
source ../scripts/source_retargeting_setup.sh

# Run everything from the package directory so the relative paths
# (examples/, demo_data/, models/, assets/) resolve as written below.
cd modules/01_retargeting/HoloNew/HoloNew
```

The package is installed editable (`holonew.egg-info` present), so `python -m
HoloNew...` module commands work from here as well.

---

## 1. Retargeters (three solver folders)

The solvers live under `src/`:

| Solver | Folder | Description |
|--------|--------|-------------|
| Holosoma native SOCP | `src/holosoma/` | Original interaction-mesh retargeter |
| GMR-SOCP v1 | `src/gmr_socp_v1/` | Autonomous GMR-SOCP solver, mink-aligned preprocessing |
| GMR-SOCP v2 | `src/gmr_socp_v2/` | Second GMR-SOCP variant |

### Holosoma native — standalone CLI

```bash
# Robot-only (OMOMO)
python examples/robot_retarget.py \
  --data_path demo_data/OMOMO_new --task-type robot_only \
  --task-name sub3_largebox_003 --data_format smplh \
  --retargeter.debug --retargeter.visualize
```

(Full task-type / dataset matrix is in [README.md](README.md).)

### GMR-SOCP v1 / v2

The GMR solvers are driven through the **stage viewer**, which runs all three
methods on the same sequence and lets you compare them (see §2). To exercise them
in isolation, use the Python API or the targeted tests:

```bash
# Targeted GMR tests (solve, tables, targets, orientation, exposed stages)
pytest tests/test_gmr_socp.py tests/test_gmr_tables.py \
       tests/test_gmr_targets.py tests/test_gmr_orientation.py \
       tests/test_gmr_stages_exposed.py -q
```

```python
# Minimal API usage
from HoloNew.src.gmr_socp_v1.gmr_socp_v1 import GmrSocpRetargeterV1
from HoloNew.src.gmr_socp_v2.gmr_socp_v2 import GmrSocpRetargeterV2
```

---

## 2. Stage visualization — `view_stages.py`

Runs holosoma native + GMR-SOCP v1 + GMR-SOCP v2 headless on one sequence, then
opens a viser viewer with **Method** and **Stage** dropdowns (per-method
preprocessing stages: Original → Mapped → Scaled → Offset → Ground).

The task defaults to `object_interaction` (so the manipulated object shows); pass
`--task-type robot_only` / `climbing` to override.

```bash
# All three optimizers, default task (object_interaction), default sequence
python examples/view_stages.py
# Viewer at http://localhost:8080 — Enter in the terminal to exit.
```

Use `--methods` to solve only a subset instead of all three. Choices:
`holosoma`, `gmr_socp_v1`, `gmr_socp_v2` (space-separated, order preserved).

```bash
# Only GMR-SOCP v1
python examples/view_stages.py --task-name sub3_largebox_003 --methods gmr_socp_v1

# holosoma + GMR-SOCP v2 only
python examples/view_stages.py --task-name sub3_largebox_003 --methods holosoma gmr_socp_v2
```

Takes the same `RetargetingConfig` flags as `robot_retarget.py`, plus `--methods`
and `--omomo_dir` (see SMPL-X mesh below).

The viewer also exposes (GUI folders, right panel):

- **Playback** — Frame slider, **Play / Pause**, and an **FPS** control to play
  the motion (scrubbing the slider pauses it).
- **Display** — Method / Stage dropdowns and a **Show G1 URDF** toggle (hide the
  robot mesh to see its skeleton underneath).
- **Skeleton** — toggle body/finger **bones** and **joints**. Every stage renders
  as a skeleton: the 52-joint source on `Original`, the mapped bodies on the
  preprocessing stages, and the solved G1 (link FK) on the `Robot` stage.
- **Meshes** — overlay the posed **SMPL-X mesh** and the **object mesh**, on any
  stage.
- **Ghost** — pick a second `(Method, Stage)` to overlay faded for comparison
  (`Off` to disable). The ghost covers skeleton stages, not the robot.

### SMPL-X mesh (Meshes → SMPL-X mesh)

The mesh is posed from the `.pt` per-joint quaternions on the same raw joints the
solvers receive (no extra re-grounding — the `Original` stage is the literal
preprocess input, so the mesh stays aligned with it). Two data sources, both used
automatically:

- **SMPL-X model dir** — `SMPLX_MODEL_DIR_DEFAULT`:
  `/home/gbesset/Documents/wbt_rl/data/00_raw_datasets/models/models_smplx_v1_1/models`
- **Subject shape (betas + gender)** — loaded by default from `OMOMO_DIR_DEFAULT`
  (the original OMOMO release holding `data/{train,test}_diffusion_manip_seq_joints24.p`,
  NOT `OMOMO_new`), so the mesh gets the subject's real body, not the neutral mean:
  `/home/gbesset/Documents/wbt_rl/data/00_raw_datasets/OMOMO`

Override the OMOMO root with `--omomo_dir <path>` if needed; a missing file degrades
to the neutral shape.

```bash
# betas load automatically — no extra flag needed
python examples/view_stages.py --task-name sub3_largebox_003 --methods gmr_socp_v1
```

---

## 3. Contact field viewer — `view_contact.py`

Standalone viser app for the per-frame contact field. Loads the bundled demo
field (`assets/contact/contact_sub3_largebox_003.npz`) and colors human witness
points by signed distance to the object (red = touching/penetrating, blue = at the
margin). Data only — runs no solve, pure numpy.

```bash
python examples/view_contact.py
# Viewer at http://localhost:8080, scrub the Frame slider.
```

---

## 4. Human → G1 OT correspondence

### Viewer — `view_correspondence.py`

Shows the G1 surface points colored by body segment (always available from the
bundled `assets/correspondence/corr_neutral.npz`). If a SMPL-X model dir is
available it also draws the human rest surface.

```bash
python examples/view_correspondence.py
```

### Build a correspondence table — `build_correspondence.py`

Requires a local SMPL-X models directory (not bundled).

```bash
python -m HoloNew.src.gmr_socp_v2.correspondence.build_correspondence \
  --model-dir /path/to/models/smplx \
  --gender neutral \
  --out assets/correspondence/corr_neutral.npz
# Optional: --urdf, --human-density, --g1-density 3000, --reg 0.005
```

### Quality report — `quality.py`

Prints axial-monotonicity / neighbour-preservation / coverage per body segment
(no ground-truth pairing needed).

```bash
python -m HoloNew.src.gmr_socp_v2.correspondence.quality \
  --model-dir /path/to/models/smplx --gender neutral
# Optional: --g1-density 50000, --reg 0.005
```

---

## 5. Body-velocity player — `viser_body_vel_player.py`

Replays a MuJoCo-converted `.npz` (with `body_pos_w` / `body_lin_vel_w`) and draws
per-body velocity arrows. Useful to sanity-check converted RL-training data.

```bash
python data_conversion/viser_body_vel_player.py \
  --npz_path ../converted_res/robot_only/sub3_largebox_003_mj.npz \
  --robot_urdf models/g1/g1_29dof.urdf
# Optional: --vel_scale 0.1, --vel_min_norm 0.01, --fps_override, --no-loop
```

For replaying plain retargeted results (robot + optional object), use
`viser_player.py` (documented in [README.md](README.md)).

---

## 6. Tests & parity

```bash
# Everything
pytest -q

# Parity: native vs vanilla holosoma, and GMR-SOCP vs mink GMR
pytest tests/test_parity_native_vs_holosoma.py \
       tests/test_parity_gmr_socp_vs_mink.py -q

# Golden retarget trajectory (regression on saved qpos)
pytest tests/test_retarget_golden.py -q

# Stage registry / viewer
pytest tests/test_stages.py tests/test_holosoma_stages.py tests/test_viewer.py -q

# Contact field
pytest tests/test_contact_field.py tests/test_contact_sdf.py \
       tests/test_contact_io.py tests/test_contact_backends.py \
       tests/test_contact_combined.py tests/test_contact_v2.py -q

# Correspondence
pytest tests/test_correspondence_build.py tests/test_correspondence_ot.py \
       tests/test_correspondence_g1.py tests/test_correspondence_human.py \
       tests/test_correspondence_segments.py tests/test_correspondence_v2.py -q
```

---

## 7. Design docs

Specs and implementation plans for the additions:

- `docs/specs/2026-06-11-gmr-socp-retargeter-design.md`
- `docs/specs/2026-06-12-three-solver-folders-design.md`
- `docs/specs/2026-06-12-per-method-stage-viz-design.md`
- `docs/specs/2026-06-11-ot-correspondence-v2-design.md`
- `docs/specs/2026-06-11-contact-fields-v2-design.md`
- `docs/plans/` — matching step-by-step implementation plans.
