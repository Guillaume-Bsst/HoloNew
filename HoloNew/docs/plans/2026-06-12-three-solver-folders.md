# Three Autonomous Solver Folders — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the three retargeting methods into self-contained folders `src/holosoma/`, `src/gmr_socp_v1/`, `src/gmr_socp_v2/`, each owning its solver + preprocessing (+ GMR tables/targets), as a behaviour-preserving refactor.

**Architecture:** Pure move/rename refactor guarded by the existing parity + golden + full suite. The native retargeter moves to `src/holosoma/` content-unchanged; the shared `src/gmr_socp/` is split into two fully-duplicated GMR folders; holosoma's resolution/preprocessing helpers are extracted out of the shared `utils.py` into `src/holosoma/`. Generic loaders/config/viewer stay shared. The GMR `from_config` is decoupled from holosoma's preprocessing.

**Tech Stack:** Python, pytest, git mv. Runs in the `holonew` conda env.

**Reference spec:** `docs/specs/2026-06-12-three-solver-folders-design.md`

## Critical environment (every task)
- Package dir (cwd): `/home/gbesset/Documents/wbt_rl/modules/01_retargeting/HoloNew/HoloNew`.
- Package imports as `HoloNew`. Work on a feature branch off `main` (controller creates it).
- **Always use this Python:** `/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python` (never bare python).
- Commits: identity `Guillaume-Bsst`; `git commit` normally; **never add Co-Authored-By/Claude or any Claude mention**; comments/docs in **English**.
- **This is behaviour-preserving.** The guard after every task is the suite:
  `/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/ -q` (40 tests, ~3-4 min).
  In particular `tests/test_parity_native_vs_holosoma.py` (native qpos byte-identical),
  `tests/test_parity_gmr_socp_vs_mink.py` (GMR base RMSE < 0.1), `tests/test_retarget_golden.py`.

## Importer inventory (files that reference the moved modules — update each as its module moves)
`interaction_mesh_retargeter`: `examples/robot_retarget.py`, `examples/parallel_robot_retarget.py`.
`gmr_socp`: `examples/view_stages.py`, `src/stages.py` (only docstrings/labels — check), `src/contact/motion.py` (check), `tests/test_gmr_*.py`, `tests/test_parity_gmr_socp_vs_mink.py`, `tests/test_correspondence_v2.py`, `tests/test_contact_v2.py`.
Use `git grep -l` before each move to catch any missed importer.

## utils.py function membership (Task 3)
- **Move to `src/holosoma/` (holosoma resolution + preprocessing):**
  `preprocess_motion_data`, `calculate_scale_factor` (→ `holosoma/preprocess.py`);
  `create_interaction_mesh`, `get_adjacency_list`, `calculate_laplacian_coordinates`,
  `calculate_laplacian_matrix`, `transform_points_world_to_local`,
  `transform_points_local_to_world` (→ `holosoma/interaction_mesh.py`).
- **Stay shared in `src/utils.py`:** everything else (`load_intermimic_data`,
  `load_object_data`, `weighted_surface_sampling*`, `extract_*`, `augment_object_poses`,
  `transform_from_human_to_world`, `find_standing_pose`, `load_smpl_motion`,
  `create_top_surface_weight_function`, `scale_points_in_object_axes_frame`,
  `create_scaled_*`, `create_new_scene_xml_file`, `transform_y_up_to_z_up`,
  `estimate_human_orientation`).
- Before moving any function, `git grep -n "<func>"` to confirm it is NOT used by a
  shared consumer (examples loaders). If a "move" function is also used by
  `examples/robot_retarget.py`'s shared loader path, keep it shared and import it into
  holosoma instead — the suite will flag a wrong call.

---

## Task 1: Create `src/holosoma/`, move the holosoma-only support files

`mujoco_utils.py` and `viser_utils.py` are used ONLY by the native retargeter.

**Files:** Create `src/holosoma/__init__.py`; move `src/mujoco_utils.py`, `src/viser_utils.py` → `src/holosoma/`.

- [ ] **Step 1: Confirm they're holosoma-only**
```bash
git grep -n "mujoco_utils\|viser_utils" -- src examples tests
```
Expected: references only from `interaction_mesh_retargeter.py` (and the GMR copies' own helper text, not real imports). If any shared consumer uses them, stop and report.

- [ ] **Step 2: Move them**
```bash
mkdir -p src/holosoma && touch src/holosoma/__init__.py
git mv src/mujoco_utils.py src/holosoma/mujoco_utils.py
git mv src/viser_utils.py src/holosoma/viser_utils.py
```

- [ ] **Step 3: Fix the native retargeter's sys.path import block.**
`src/interaction_mesh_retargeter.py` adds `Path(__file__).parent.parent/"src"` to `sys.path` and does bare `from mujoco_utils import ...` / `from viser_utils import ...`. Since these now live in `src/holosoma/` and the retargeter is still in `src/` (it moves in Task 2), update the bare imports to package imports:
```python
from HoloNew.src.holosoma.mujoco_utils import _world_mesh_from_geom
from HoloNew.src.holosoma.viser_utils import create_motion_control_sliders
```
Leave the `from utils import (...)` block alone for now (Task 3). Keep the `sys.path` insert for `utils` only.

- [ ] **Step 4: Run the suite**
`/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/ -q`
Expected: 40 passed. (If an import fails, fix the path.)

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "refactor: move holosoma-only mujoco_utils/viser_utils into src/holosoma/"
```

---

## Task 2: Move the native retargeter into `src/holosoma/`

**Files:** `git mv src/interaction_mesh_retargeter.py src/holosoma/interaction_mesh_retargeter.py`; update importers.

- [ ] **Step 1: Move it**
```bash
git mv src/interaction_mesh_retargeter.py src/holosoma/interaction_mesh_retargeter.py
```

- [ ] **Step 2: Fix the moved file's internal imports.**
It is now at `src/holosoma/`. The `sys.path` block currently computes `Path(__file__).parent.parent/"src"` — from `holosoma/` that resolves to the package dir + `/src` = `src/` (correct for the still-shared `utils`). Verify `from utils import (...)` still resolves (utils.py is still in `src/`). The `mujoco_utils`/`viser_utils` imports were made package-qualified in Task 1 — keep them. `from HoloNew.config_types.retargeter import ...` is unaffected.

- [ ] **Step 3: Update external importers** (exact substitutions):
- `examples/robot_retarget.py`: `from HoloNew.src.interaction_mesh_retargeter import InteractionMeshRetargeter` → `from HoloNew.src.holosoma.interaction_mesh_retargeter import InteractionMeshRetargeter`.
- `examples/parallel_robot_retarget.py`: same substitution.
- Any test importing it (grep): update.
```bash
git grep -l "src.interaction_mesh_retargeter\|src\.interaction_mesh_retargeter" -- examples tests
```
Apply the path change to each hit.

- [ ] **Step 4: Run the suite** — Expected: 40 passed. The native parity test must pass (qpos byte-identical; only the file location changed).

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "refactor: move native retargeter into src/holosoma/"
```

---

## Task 3: Extract holosoma's preprocessing + interaction-mesh helpers out of utils.py

**Files:** Create `src/holosoma/preprocess.py`, `src/holosoma/interaction_mesh.py`; trim `src/utils.py`; update the native retargeter's imports.

- [ ] **Step 1: Create `src/holosoma/interaction_mesh.py`** by moving these functions VERBATIM from `src/utils.py`: `create_interaction_mesh`, `get_adjacency_list`, `calculate_laplacian_coordinates`, `calculate_laplacian_matrix`, `transform_points_world_to_local`, `transform_points_local_to_world` (plus any numpy/scipy imports they need). Remove them from `src/utils.py`.
- [ ] **Step 2: Create `src/holosoma/preprocess.py`** by moving `preprocess_motion_data` and `calculate_scale_factor` VERBATIM from `src/utils.py` (with their imports). Remove them from `src/utils.py`.
- [ ] **Step 3: Update the native retargeter's imports.** In `src/holosoma/interaction_mesh_retargeter.py` the `from utils import (create_interaction_mesh, calculate_laplacian_coordinates, calculate_laplacian_matrix, create_interaction_mesh, get_adjacency_list, transform_points_local_to_world, transform_points_world_to_local)` block now resolves to the holosoma module:
```python
from HoloNew.src.holosoma.interaction_mesh import (
    calculate_laplacian_coordinates, calculate_laplacian_matrix,
    create_interaction_mesh, get_adjacency_list,
    transform_points_local_to_world, transform_points_world_to_local,
)
```
Drop the now-unneeded `sys.path` hack and bare `from utils import ...` if nothing else in the file needs bare `utils`. (If the file still imports other `utils` functions that stayed shared, keep a `from HoloNew.src.utils import ...` for those.)
- [ ] **Step 4: Update other consumers of the moved functions.** `git grep -n "preprocess_motion_data\|calculate_scale_factor\|create_interaction_mesh\|calculate_laplacian"` — update each importer to the new holosoma path. Notably `examples/robot_retarget.py` imports `preprocess_motion_data` (used by both native prep and GMR `from_config`); point it at `HoloNew.src.holosoma.preprocess`.
- [ ] **Step 5: Run the suite** — Expected: 40 passed.
- [ ] **Step 6: Commit**
```bash
git add -A && git commit -m "refactor: move holosoma preprocessing + interaction-mesh helpers into src/holosoma/"
```

---

## Task 4: Split GMR v1 into `src/gmr_socp_v1/`

**Files:** Create `src/gmr_socp_v1/` with `gmr_socp_v1.py` + folder-local copies of `tables.py`, `targets.py`, `preprocess.py`.

- [ ] **Step 1: Create the folder with copies**
```bash
mkdir -p src/gmr_socp_v1 && touch src/gmr_socp_v1/__init__.py
git mv src/gmr_socp/gmr_socp_v1.py src/gmr_socp_v1/gmr_socp_v1.py
cp src/gmr_socp/tables.py     src/gmr_socp_v1/tables.py
cp src/gmr_socp/targets.py    src/gmr_socp_v1/targets.py
cp src/gmr_socp/preprocess.py src/gmr_socp_v1/preprocess.py
git add src/gmr_socp_v1/tables.py src/gmr_socp_v1/targets.py src/gmr_socp_v1/preprocess.py
```
(`tables.py`/`targets.py`/`preprocess.py` stay in `src/gmr_socp/` for v2 until Task 5.)

- [ ] **Step 2: Fix `gmr_socp_v1.py` internal imports.** Its `from .tables import ...`, `from .targets import ...`, `from .preprocess import ...` now resolve to the folder-local copies — no change needed (relative imports). Confirm it does not import from `src/gmr_socp/` absolutely anywhere; if it does (e.g. `from HoloNew.src.gmr_socp.X`), change to `.X`.

- [ ] **Step 3: Update external importers of v1:**
- `examples/view_stages.py`: `from HoloNew.src.gmr_socp.gmr_socp_v1 import GmrSocpRetargeterV1` → `from HoloNew.src.gmr_socp_v1.gmr_socp_v1 import GmrSocpRetargeterV1`.
- `tests/test_gmr_socp.py`, `tests/test_gmr_orientation.py`, `tests/test_parity_gmr_socp_vs_mink.py`: same substitution for the v1 import.
- `tests/test_gmr_tables.py`, `tests/test_gmr_targets.py`: they import `HoloNew.src.gmr_socp.tables` / `.targets`. Point them at `HoloNew.src.gmr_socp_v1.tables` / `.targets` (the v1 copy is the canonical one for these unit tests).
```bash
git grep -l "gmr_socp.gmr_socp_v1\|gmr_socp\.tables\|gmr_socp\.targets" -- examples tests
```

- [ ] **Step 4: Run the suite** — Expected: 40 passed. GMR parity (v1) must pass.

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "refactor: split GMR v1 into autonomous src/gmr_socp_v1/"
```

---

## Task 5: Split GMR v2 into `src/gmr_socp_v2/` and remove `src/gmr_socp/`

**Files:** Create `src/gmr_socp_v2/` with `gmr_socp_v2.py` + copies; delete the old `src/gmr_socp/`.

- [ ] **Step 1: Create the folder with copies**
```bash
mkdir -p src/gmr_socp_v2 && touch src/gmr_socp_v2/__init__.py
git mv src/gmr_socp/gmr_socp_v2.py src/gmr_socp_v2/gmr_socp_v2.py
git mv src/gmr_socp/tables.py     src/gmr_socp_v2/tables.py
git mv src/gmr_socp/targets.py    src/gmr_socp_v2/targets.py
git mv src/gmr_socp/preprocess.py src/gmr_socp_v2/preprocess.py
git rm src/gmr_socp/__init__.py
```
(`src/gmr_socp/` is now empty and removed.)

- [ ] **Step 2: Fix `gmr_socp_v2.py` internal imports** (relative `.tables`/`.targets`/`.preprocess` resolve to the folder-local copies). Confirm no leftover `HoloNew.src.gmr_socp.X` absolute imports.

- [ ] **Step 3: Update external importers of v2:**
- `examples/view_stages.py`: v2 import → `from HoloNew.src.gmr_socp_v2.gmr_socp_v2 import GmrSocpRetargeterV2`.
- `tests/test_gmr_socp.py`, `tests/test_correspondence_v2.py`, `tests/test_contact_v2.py`: v2 import substitution.
```bash
git grep -l "gmr_socp.gmr_socp_v2\|HoloNew.src.gmr_socp\b" -- examples tests src
```
Expected after fixes: no remaining reference to `HoloNew.src.gmr_socp.` (the old package).

- [ ] **Step 4: Run the suite** — Expected: 40 passed. GMR parity (v1 AND v2) must pass.

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "refactor: split GMR v2 into autonomous src/gmr_socp_v2/, remove src/gmr_socp/"
```

---

## Task 6: Decouple GMR `from_config` from holosoma's preprocessing

After the earlier preprocessing fix, GMR's targets come from its own `compute_stages`; the holosoma `preprocess_motion_data` / `initialize_robot_pose` calls only produced a `q_init` whose base is then overridden by the ground pelvis and whose joints are zero. Drop those calls and build `q_init` directly. **The GMR-vs-mink parity test (RMSE < 0.1) and the v1/v2 integration tests are the guard — qpos must not change.**

**Files:** Modify `src/gmr_socp_v1/gmr_socp_v1.py` and `src/gmr_socp_v2/gmr_socp_v2.py` (`from_config`).

- [ ] **Step 1: Capture current GMR qpos as a reference** (to prove the decoupling is qpos-neutral)
```bash
/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -c "
import numpy as np
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.gmr_socp_v1.gmr_socp_v1 import GmrSocpRetargeterV1
q = GmrSocpRetargeterV1.from_config(RetargetingConfig(task_type='robot_only', task_name='sub3_largebox_003', data_format='smplh')).retarget().qpos
np.save('/tmp/gmr_v1_before.npy', q); print('saved', q.shape)"
```

- [ ] **Step 2: Simplify `from_config` in `gmr_socp_v1.py`.** Replace the `preprocess_motion_data(...)` + `initialize_robot_pose(...)` block and the subsequent `q_init_full` construction with: build the retargeter (`rt = cls(**kwargs)`), load raw joints + quats, run `compute_stages` for the ground stage, and set
```python
        q_init_full = np.zeros(rt.nq)               # joints start at zero (as before)
        rt.gmr_ground = ground
        _pelvis_bi = MAPPED_BODY_NAMES.index(HUMAN_ROOT_NAME)
        q_init_full[:3] = ground["pos"][0, _pelvis_bi]
        q_init_full[3:7] = ground["quat"][0, _pelvis_bi]
        rt.q_init_full = q_init_full
        rt.human_quat = human_quat
```
Keep using the SHARED `load_motion_data` / `create_task_constants` / `build_retargeter_kwargs_from_config` (loading is not solver-specific). Remove the now-unused `preprocess_motion_data` / `initialize_robot_pose` imports. Keep the `human_joints` raw load only if still needed for `T` alignment; otherwise derive `T` from `human_quat`/`raw_joints`.

- [ ] **Step 3: Prove qpos unchanged**
```bash
/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -c "
import numpy as np
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.gmr_socp_v1.gmr_socp_v1 import GmrSocpRetargeterV1
q = GmrSocpRetargeterV1.from_config(RetargetingConfig(task_type='robot_only', task_name='sub3_largebox_003', data_format='smplh')).retarget().qpos
b = np.load('/tmp/gmr_v1_before.npy')
import numpy.testing as t; t.assert_allclose(q, b, atol=1e-9); print('GMR v1 qpos unchanged', q.shape)"
```
Expected: prints unchanged. If qpos differs, the decoupling changed behaviour — investigate `initialize_robot_pose`'s effect on `q_init` joints (they should be zeros) before proceeding.

- [ ] **Step 4: Apply the identical change to `gmr_socp_v2.py`** (its `from_config` has the same prep block before the correspondence/contact loading — keep those blocks).

- [ ] **Step 5: Run the suite** — Expected: 40 passed (GMR parity + integration confirm qpos identical).

- [ ] **Step 6: Commit**
```bash
git add -A && git commit -m "refactor: decouple GMR from_config from holosoma preprocessing (v1 + v2)"
```

---

## Task 7: Create the holosoma preprocessing exposure stub + final verification

This task only ensures `holosoma/preprocess.py` is the single home of holosoma's preprocessing (Task 3 placed `preprocess_motion_data`/`calculate_scale_factor` there) and the structure is clean. No behaviour change.

**Files:** none new; verification + cleanup.

- [ ] **Step 1: Confirm the final structure**
```bash
ls src/holosoma src/gmr_socp_v1 src/gmr_socp_v2
git grep -l "HoloNew.src.gmr_socp\b\|src.interaction_mesh_retargeter" -- src examples tests || echo "no stale references"
```
Expected: the three folders exist with their files; no stale references to the old paths.

- [ ] **Step 2: Full suite + the parity guards explicitly**
```bash
/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/ -q
/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_parity_native_vs_holosoma.py tests/test_parity_gmr_socp_vs_mink.py tests/test_retarget_golden.py -v -s
```
Expected: 40 passed; native parity max|diff|=0.0; GMR parity base-pos RMSE < 0.1.

- [ ] **Step 3: Commit any final import tidy-ups**
```bash
git add -A && git commit -m "refactor: finalize three-solver-folder structure" --allow-empty
```

---

## Self-Review notes
- **Spec coverage:** 3 folders → Tasks 1-5; native moved content-unchanged → Task 2 (guarded by native parity); GMR full duplication → Tasks 4-5; holosoma preprocessing file + interaction-mesh extraction → Task 3; decouple GMR from holosoma preprocessing → Task 6; shared plumbing untouched (viewer/stages/retarget_result/correspondence/contact/loaders).
- **Behaviour-preserving:** the only behaviour-touching task is Task 6, and it is explicitly proven qpos-neutral (Step 1/3 before/after compare) on top of the GMR parity test.
- **Open items for the implementer:** the exact `utils.py` function membership (the plan lists it; verify each with `git grep` before moving — keep shared if a shared loader uses it); whether `src/stages.py` / `src/contact/motion.py` actually import gmr_socp (grep showed them as matches — they may be only string/label matches; verify and update only real imports); the native `sys.path` hack may be fully removable after Task 3 (only if no bare `utils` import remains).
- **No new tests:** this is a refactor; the existing 40-test suite (esp. the two parity tests + golden) is the contract. Each task ends by running it.
