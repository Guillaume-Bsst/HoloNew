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
| Holosoma native SOCP | `src/holosoma/` | Original interaction-mesh (Laplacian) retargeter |
| GMR-SOCP | `src/gmr_socp/` | Autonomous GMR-SOCP solver, mink-aligned preprocessing |
| TEST-SOCP | `src/test_socp/` | Paper-formulation per-frame SQP: pelvis-relative Style + interaction D/X/P (SDF + OT correspondence) + centroidal W^c/W^L + movable object W^o + inertia mode |

### Holosoma native — standalone CLI

```bash
# Robot-only (OMOMO)
python examples/robot_retarget.py \
  --data_path demo_data/OMOMO_new --task-type robot_only \
  --task-name sub3_largebox_003 --data_format smplh \
  --retargeter.debug --retargeter.visualize
```

(Full task-type / dataset matrix is in [README.md](README.md).)

#### 3-path façade (OMOMO mixed / HOI-M3)

The same script also accepts a `--dataset` key plus three explicit, no-default paths
(`--model-path` / `--motion-path` / `--obj-path`). When `--dataset` is set it replaces
`--data_path`/`--task-name`/`--data_format`; the legacy flags above keep working when it
is omitted. The paths below are filled in for **our current local config** (adjust to
your own layout — these are the only concrete paths in this cheat sheet).

```bash
# OMOMO robot_only — motion from the new .pt, betas from the non-new pickle.
# --smpl-model-dir is explicit (no default): the SMPL-H body model that turns the
# betas into a stature for the scale factor.
python examples/robot_retarget.py --dataset omomo --task-type robot_only \
  --model-path     /home/gbesset/Documents/wbt_rl/data/00_raw_datasets/OMOMO/data/train_diffusion_manip_seq_joints24.p \
  --motion-path    demo_data/OMOMO_new/sub3_largebox_003.pt \
  --smpl-model-dir /home/gbesset/Documents/wbt_rl/data/00_raw_datasets/models/smplh

# OMOMO object_interaction — same, plus the object mesh via --obj-path
# (the sibling largebox_cleaned_simplified.urdf is picked up automatically).
python examples/robot_retarget.py --dataset omomo --task-type object_interaction \
  --model-path     /home/gbesset/Documents/wbt_rl/data/00_raw_datasets/OMOMO/data/train_diffusion_manip_seq_joints24.p \
  --motion-path    demo_data/OMOMO_new/sub3_largebox_003.pt \
  --obj-path       /home/gbesset/Documents/wbt_rl/data/00_raw_datasets/OMOMO/data/captured_objects/largebox_cleaned_simplified.obj \
  --smpl-model-dir /home/gbesset/Documents/wbt_rl/data/00_raw_datasets/models/smplh

# HOI-M3 robot_only — raw SMPL-X .npz; --model-path is the SMPL-X body-model dir.
python examples/robot_retarget.py --dataset hoim3 --task-type robot_only \
  --model-path  /home/gbesset/Documents/wbt_rl/data/00_raw_datasets/models/models_smplx_v1_1/models/smplx \
  --motion-path /home/gbesset/Documents/wbt_rl/data/00_raw_datasets/HOI-M3/smplx/subject01_baseball.npz

# HOI-M3 object_interaction — plus the object 6DoF .npz via --obj-path.
python examples/robot_retarget.py --dataset hoim3 --task-type object_interaction \
  --model-path  /home/gbesset/Documents/wbt_rl/data/00_raw_datasets/models/models_smplx_v1_1/models/smplx \
  --motion-path /home/gbesset/Documents/wbt_rl/data/00_raw_datasets/HOI-M3/smplx/subject01_baseball.npz \
  --obj-path    /home/gbesset/Documents/wbt_rl/data/00_raw_datasets/HOI-M3/object/subject01_baseball.npz
```

### GMR-SOCP / TEST-SOCP

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
from HoloNew.src.gmr_socp.gmr_socp import GmrSocpRetargeter
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
```

#### Optional holosoma-style constraints

GMR-SOCP and TEST-SOCP carry holosoma's constraint helpers — object/ground
non-penetration, self-collision, foot-sticking, and foot-lock — **copied verbatim
from `src/holosoma/interaction_mesh_retargeter.py`** but **disabled by default**.
The default solve is bit-identical to the unconstrained baseline.

To opt in, pass the solver-specific config class with the desired flags set:

```python
from HoloNew.config_types.retargeting import RetargetingConfig
from HoloNew.src.gmr_socp.config import GmrSocpRetargeterConfig   # or TestSocpRetargeterConfig
from HoloNew.src.gmr_socp.gmr_socp import GmrSocpRetargeter

cfg = RetargetingConfig(
    task_type="robot_only",
    task_name="sub3_largebox_003",
    data_format="smplh",
    retargeter=GmrSocpRetargeterConfig(
        activate_obj_non_penetration=True,   # ground non-penetration on robot_only
        activate_foot_sticking=True,         # foot-sticking on g1 (left/right ankle links)
    ),
)
rt = GmrSocpRetargeter.from_config(cfg)
res = rt.retarget()
```

Constraint applicability:

| Flag | Works on | Notes |
|------|----------|-------|
| `activate_obj_non_penetration` | `robot_only` (ground) and object tasks | Ground non-pen uses the `ground` geom in the g1 xml; object non-pen requires an object task + scene xml |
| `activate_foot_sticking` | `robot_only` and any task with g1 | Uses g1 `FOOT_STICKING_LINKS` (left/right ankle links) |
| `activate_self_collision` | Any task | Requires `self_collision=SelfCollisionConfig(enable=True, pairs=[...])` to take effect |
| `foot_lock=FootLockConfig(enable=True, ...)` | `robot_only` and any task with g1 | Independent toggle (gated only by `foot_lock.enable`); pins the foot Z to the floor over the configured frame windows |

#### TEST-SOCP paper-formulation features

Beyond the holosoma-style constraints, TEST-SOCP carries the paper terms. They live
in `src/test_socp/config.py` and are set on `TestSocpRetargeterConfig`:

| Field (default) | Effect |
|-----------------|--------|
| `lambda_D` / `lambda_X` (`20.0`) | interaction distance / cross terms (SDF + OT correspondence) |
| `lambda_P` (`0.0`) | tangential band constraint is a hard constraint, on by default; this is the soft fallback weight |
| `inertia_mode` (`False`) / `floor_as_entity` (`False`) | contacts place the body (Style + contacts), weak `W^c` carries the ballistic phase when contacts are few/absent |
| `track_L_ref` (`False`) / `lambda_L_track` (`5.0`) | track the lumped reference orbital angular momentum `L^ref` (opt-in; can suppress jumps at high weight) |
| `lambda_object_floor` (`0.0`) | object↔floor contact term (object inertia) |
| `activate_obj_surface_nonpen` (`False`) | object surface non-penetration (opt-in; slow) |
| `scale_xy_robot` (`1.0`) / `scale_z_robot` (`None`) | world placement of the robot root inside the preprocess scale stage: multiplier on the raw grounded axis (1.0 = raw; `<1` pulls toward the origin/floor like holosoma). `None` = native morphological scaling. TEST keeps raw XY so targets and the contact field share one frame |
| `scale_xy_object` (`1.0`) / `scale_z_object` (`1.0`) | same, applied independently to the object pose (XY / Z). Defaults leave the object raw |

> GMR-SOCP carries the same four knobs on `GmrSocpRetargeterConfig`, all defaulting to
> `None` = GMR's native behaviour (XY → the holosoma scale factor `~0.68`; robot Z →
> native morphological; object Z → raw). Set `scale_xy_robot=1.0` to keep the raw root
> XY (this is what makes GMR-SOCP match the mink-GMR reference).

```python
from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
cfg = RetargetingConfig(
    task_type="object_interaction", task_name="sub3_largebox_003", data_format="smplh",
    retargeter=TestSocpRetargeterConfig(inertia_mode=True, floor_as_entity=True),
)
```

---

## 2. Stage visualization — `view_stages.py`

Runs holosoma native + GMR-SOCP + TEST-SOCP headless on one sequence, then
opens a viser viewer with **Method** and **Stage** dropdowns (per-method
preprocessing stages: Original → Mapped → Scaled → Offset → Ground).

The task defaults to `object_interaction` (so the manipulated object shows); pass
`--task-type robot_only` / `climbing` to override.

```bash
# All three optimizers, default task (object_interaction), default sequence
python examples/view_stages.py
# Viewer at http://localhost:8080 — Enter in the terminal to exit.
```

#### 3-path façade on the stage viewer (OMOMO mixed / HOI-M3)

`view_stages.py` accepts the same `--dataset` façade as `robot_retarget.py`. The
**simplest form is `--motion-name <seq>`**: with `--dataset`, the model / motion /
object files are resolved automatically from the global dataset roots (the
`WBT_OMOMO_DIR` / `WBT_OMOMO_NEW_DIR` / `WBT_HOIM3_DIR` / `WBT_SMPLX_DIR` /
`WBT_SMPLH_DIR` env vars, exported by `source_retargeting_setup.sh`). All selected
methods (holosoma / GMR-SOCP / TEST-SOCP) then run on that sequence. HOI-M3's raw
SMPL-X is prepped once (cached) into the processed format the smplx path consumes.

```bash
# By name — files resolved from the global roots, nothing else to type:
python examples/view_stages.py --dataset omomo --task-type robot_only \
  --motion-name sub3_largebox_003 --methods gmr_socp test_socp

python examples/view_stages.py --dataset hoim3 --task-type robot_only \
  --motion-name subject01_baseball --methods test_socp --max-frames 200
```

The explicit-path form still works (and overrides name resolution) when files live
elsewhere — `--model-path` / `--motion-path` (/ `--obj-path`, `--smpl-model-dir`).
`--motion-name` is supported for `omomo` and `hoim3`; the same flags work on
`robot_retarget.py`.

> Note: `--obj-path` is resolved by name too, but the **object overlay in the viewer
> is currently wired only for the OMOMO smplh path** — HOI-M3 shows human + robot, not
> the object.

Use `--methods` to solve only a subset instead of all three. Choices:
`holosoma`, `gmr_socp`, `test_socp` (space-separated, order preserved).

```bash
# Only GMR-SOCP
python examples/view_stages.py --task-name sub3_largebox_003 --methods gmr_socp

# holosoma + TEST-SOCP only
python examples/view_stages.py --task-name sub3_largebox_003 --methods holosoma test_socp
```

Takes the same `RetargetingConfig` flags as `robot_retarget.py`, plus `--methods`,
`--omomo-dir` (see SMPL-X mesh below) and `--max-frames`.

**`--max-frames N`** caps the solved/displayed frames on **all** selected methods
(SFU dance and OMOMO manipulation clips run to several hundred frames — see counts
in §2b). Use it to inspect a motion without waiting for the full solve:

```bash
# First 120 frames only, TEST-SOCP alone (fastest)
python examples/view_stages.py --task-name sub3_largebox_003 \
  --methods test_socp --max-frames 120
```

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
- **Test** — the TEST-SOCP interaction/contact visuals (only the `test_socp` method
  produces them), coloured by signed distance (red = penetration → white = contact →
  blue = far):
  - **SDF Object** / **SDF Floor** — the object SDF near-surface band shell (any
    stage) and the analytic floor band.
  - **Human contact** — SMPL-X surface probes at their object/floor signed distance
    (gated to the `Grounded` stage).
  - **Object contact** / **Floor contact** — the contact footprint (witness points of
    the active human probes on the object / floor), `Grounded` stage.
  - **G1 transport** — the correspondence points carried onto the solved robot,
    coloured by their human source's distance (gated to the `Robot` stage).
  - **Directions** — probe → witness lines (human on `Grounded`, G1 on `Robot`).
- **TEST diagnostics** — the per-frame solve state of the `test_socp` method (no-op
  for the others):
  - **Solved object pose** (on by default) — re-places the object mesh/points at
    TEST's **solved** object pose (movable / inertia), overriding the reference
    pose, so the solved-vs-reference gap is visible.
  - **CoM + trail** — magenta CoM marker + grey ground shadow (z=0) + the whole-clip
    CoM polyline: the ballistic arc and balance made geometric (W^c / inertia).
  - **Angular momentum L** — an arrow from the CoM along `L`, auto-scaled so the
    clip's peak `|L|` reads ~0.5 m (direction is the cue on spins / cartwheels — W^L).
  - **Solve state** — read-only text: `CoM z`, foot `slip` (mm), `|L|` for the frame.
- **Ghost** — pick a second `(Method, Stage)` to overlay faded for comparison
  (`Off` to disable). The ghost covers skeleton stages, not the robot. **The main
  tool for the pose/style and duck-walk comparisons** (overlay GMR or holosoma
  under TEST).

### SMPL-X mesh (Meshes → SMPL-X mesh)

The mesh is posed from the `.pt` per-joint quaternions on the same raw joints the
solvers receive (no extra re-grounding — the `Original` stage is the literal
preprocess input, so the mesh stays aligned with it). Two data sources, both used
automatically:

- **SMPL-X model dir** — `SMPLX_MODEL_DIR_DEFAULT` reads `$WBT_SMPLX_DIR`
  (`src/test_socp/correspondence/constants.py`), set by `source_retargeting_setup.sh`.
  Our current config: `/home/gbesset/Documents/wbt_rl/data/00_raw_datasets/models/models_smplx_v1_1/models`
- **Subject shape (betas + gender)** — loaded by default from `OMOMO_DIR_DEFAULT`, which
  reads `$WBT_OMOMO_DIR` (`src/test_socp/contact/constants.py` — the original OMOMO
  release holding `data/{train,test}_diffusion_manip_seq_joints24.p`, NOT `OMOMO_new`),
  so the mesh gets the subject's real body, not the neutral mean. Our current config:
  `/home/gbesset/Documents/wbt_rl/data/00_raw_datasets/OMOMO`

Override the OMOMO root with `--omomo_dir <path>` (or export `WBT_OMOMO_DIR`) if needed;
a missing file degrades to the neutral shape.

```bash
# betas load automatically — no extra flag needed
python examples/view_stages.py --task-name sub3_largebox_003 --methods gmr_socp
```

---

## 2b. Interesting motions to visualize (the three experiments)

The clips that exercise the thesis: TEST keeps a GMR-level **style** while enabling
**object interaction** without the holosoma failure modes (foot-lock blow-ups,
Laplacian "duck-walk" toward the object). All commands open the §2 viewer; use the
**Ghost** folder to overlay two methods and the **TEST diagnostics** folder for the
centroidal cues.

Prepped clips and their lengths:

| Clip | `--data-path` | `--data-format` | `--task-type` | Frames | Used for |
|------|---------------|-----------------|---------------|-------:|----------|
| `0005_2FeetJump001`  | `demo_data/SFU` | `smplx` | `robot_only` | 644 | flight / ballistic (W^c) |
| `0007_Cartwheel001`  | `demo_data/SFU` | `smplx` | `robot_only` | 528 | flight + angular momentum (W^L) |
| `0017_WushuKicks001` | `demo_data/SFU` | `smplx` | `robot_only` | 595 | complex foot motion (foot-lock) |
| `0018_ChineseDance001` | `demo_data/SFU` | `smplx` | `robot_only` | 342 | one-foot spin (foot-lock) |
| `0018_DanceTurns001` | `demo_data/SFU` | `smplx` | `robot_only` | 182 | dynamic style |
| `sub3_largebox_003`  | `demo_data/OMOMO_new` | `smplh` | `object_interaction` | 196 | object interaction / duck-walk |

### E1 — Style vs GMR on highly dynamic SFU clips

TEST should track the style as well as GMR (which reproduces it well). Run both,
then Ghost one under the other and compare the skeletons / robot pose.

```bash
python examples/view_stages.py \
  --data-path demo_data/SFU --data-format smplx --task-type robot_only \
  --task-name 0018_DanceTurns001 --methods gmr_socp test_socp
# (no cap needed at 182 frames; add --max-frames for the longer clips)
```

### E2 — Holosoma "duck-walk" toward the object (OMOMO largebox)

Holosoma's Laplacian pulls the body toward the object → the legs/feet splay
(abduction, "marche en canard"). TEST's SDF activation is local, so the legs stay
put. Ghost **holosoma** under **test_socp** and watch the legs/feet; toggle the
object to see how close the body is dragged.

```bash
python examples/view_stages.py \
  --task-name sub3_largebox_003 --methods holosoma test_socp --max-frames 150
# In the viewer: Ghost -> Method=holosoma, Stage=Robot. Watch hip abduction.
```

### E3 — Foot-lock is horrible (one-foot spin, complex footwork)

Holosoma's foot pinning explodes on a self-spin on one foot and on fast footwork;
TEST stays smooth. Ghost **holosoma** under **test_socp** and watch the feet.

```bash
# One-foot spin
python examples/view_stages.py \
  --data-path demo_data/SFU --data-format smplx --task-type robot_only \
  --task-name 0018_ChineseDance001 --methods holosoma test_socp --max-frames 200

# Complex footwork
python examples/view_stages.py \
  --data-path demo_data/SFU --data-format smplx --task-type robot_only \
  --task-name 0017_WushuKicks001 --methods holosoma test_socp --max-frames 200
```

### Flight / ballistic phase (W^c, W^L)

Jumps and cartwheels: with few/no contacts the centroidal terms carry the pelvis.
Enable **TEST diagnostics → CoM + trail** (ballistic arc) and **Angular momentum L**.

```bash
python examples/view_stages.py \
  --data-path demo_data/SFU --data-format smplx --task-type robot_only \
  --task-name 0005_2FeetJump001 --methods test_socp --max-frames 250
```

> The viewer CLI builds the **default** TEST config (inertia mode off). The
> CoM/trail/L diagnostics still visualise the ballistic phase of the default solve;
> to view the **inertia-mode** variant, solve via the Python API
> (`TestSocpRetargeterConfig(inertia_mode=True, floor_as_entity=True)`, §1) and
> replay the qpos with `viser_player.py`.

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
python -m HoloNew.src.test_socp.correspondence.build_correspondence \
  --model-dir /path/to/models/smplx \
  --gender neutral \
  --out assets/correspondence/corr_neutral.npz
# Optional: --urdf, --human-density, --g1-density 3000, --reg 0.005
```

### Quality report — `quality.py`

Prints axial-monotonicity / neighbour-preservation / coverage per body segment
(no ground-truth pairing needed).

```bash
python -m HoloNew.src.test_socp.correspondence.quality \
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

# Holosoma-style constraints (default-off parity + ON smoke for GMR/TEST)
pytest tests/test_holosoma_constraints.py -q

# TEST-SOCP paper formulation: inertia mode, centroidal W^c / W^L (L^ref),
# object↔floor, SMPL-X / AMASS loaders + probe (metric + golden regressions)
pytest tests/test_inertia_mode.py tests/test_inertia_mode_metric.py \
       tests/test_inertia_mode_golden.py tests/test_inertia_flight.py \
       tests/test_centroidal_metric.py tests/test_centroidal_lref.py \
       tests/test_pin_centroidal.py tests/test_object_floor.py \
       tests/test_interaction_floor_only.py \
       tests/test_smplx_loader.py tests/test_smplx_field.py \
       tests/test_smplx_probe.py tests/test_amass_prep_orientations.py -q

# Bit-exact parity of the TEST-SOCP baseline (guards the SQP step-break + caching)
pytest tests/test_test_socp_parity.py -q
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
