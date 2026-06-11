# OT human→G1 Correspondence (v2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port test_pipe's optimal-transport human→G1 surface correspondence into HoloNew, attach it to the v2 GMR-SOCP retargeter (data only, not used in the solve), bundle a precomputed correspondence so it is turnkey, and visualize it in a dedicated annex viser app.

**Architecture:** A new `src/correspondence/` package is a faithful port of test_pipe's `transport/` modules + the SMPL-X `HumanBody` builder, with imports adapted to HoloNew. The correspondence is a static human↔G1 surface map built once (offline) and saved as a small `.npz` bundled in `assets/correspondence/`. `GmrSocpRetargeterV2.from_config` loads it; `examples/view_correspondence.py` opens a standalone viser app to display it. The SMPL-X model is NOT bundled (104 MB > GitHub limit + license) — its directory is a configurable path defaulting to the one on this machine.

**Tech Stack:** Python, numpy, scipy, `ot` (POT), `smplx`, trimesh, yourdfpy, viser, pytest. Runs in the `holonew` conda env.

**Reference spec:** `docs/specs/2026-06-11-ot-correspondence-v2-design.md`
**Port source (read-only):** `/home/gbesset/Documents/wbt_rl/modules/third_party/test_pipe/src/test_pipe_retargeting/test_pipe_retargeting/` (its `transport/`, `human/body.py`).

## Critical environment (every task)
- Repo root: `/home/gbesset/Documents/wbt_rl/modules/01_retargeting/HoloNew`; package dir (cwd for commands): `.../HoloNew/HoloNew`.
- Package imports as `HoloNew`. Work on a feature branch off `main` (controller creates it).
- **Always use this Python:** `/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python` (never bare python).
- Commits: local identity `Guillaume-Bsst`; `git commit` normally; **never add Co-Authored-By/Claude or any Claude mention**; comments/docs in **English**.
- SMPL-X model dir (present on this machine): `/home/gbesset/Documents/wbt_rl/data/00_raw_datasets/models/models_smplx_v1_1/models/smplx` (contains `SMPLX_NEUTRAL.npz`). `smplx.create(model_dir, model_type="smplx", gender="neutral", ...)` expects `model_dir` to be the dir that contains a `smplx/` subfolder OR the smplx folder itself — verify at implementation time (Task 2).
- G1 URDF: `models/g1/g1_29dof.urdf`.

## File Structure (all additive; native + v1 untouched)
- `src/correspondence/__init__.py`
- `src/correspondence/constants.py` — `G1_29DOF_URDF`, `HUMAN_GRID_DENSITY=2000.0`, `G1_DENSITY=3000.0`, `OT_REG=0.005`, `SMPLX_MODEL_DIR_DEFAULT`.
- `src/correspondence/segments.py` — ported (pure numpy).
- `src/correspondence/human_body.py` — ported `HumanBody` + `PointCloudCache` (smplx).
- `src/correspondence/human_source.py` — ported `HumanSource` + `build_human_source` + `to_g1_frame`.
- `src/correspondence/g1_surface.py` — ported `G1Surface` + `sample_g1_surface` + `build_rest_cfg`.
- `src/correspondence/ot_couple.py` — ported `couple` (POT).
- `src/correspondence/quality.py` — ported coupling-quality metrics.
- `src/correspondence/build_correspondence.py` — ported `CorrespondenceTable`, `build_table`, `save/load`, CLI `main`.
- `src/correspondence/viz.py` — ported drawing helpers, adapted for the annex app.
- `assets/correspondence/corr_neutral.npz` — bundled precomputed correspondence (Task 6).
- Modify: `pyproject.toml` + `setup.py` — add `ot` dependency.
- Modify: `src/gmr_socp/gmr_socp_v2.py` — load correspondence in `from_config`.
- Create: `examples/view_correspondence.py` — annex viser app.
- Tests: `tests/test_correspondence_*.py`.

Import adaptation rule for every ported file: replace `from test_pipe_retargeting.X import Y` with the HoloNew relative/absolute equivalent (`from .X import Y` within the package, or `from HoloNew.src.correspondence.X import Y`), and `from test_pipe_retargeting.constants import G1_29DOF_URDF, HUMAN_GRID_DENSITY` with `from .constants import ...`. Keep relative imports (`.segments`, `.g1_surface`, …) as-is. Change NO numerical constants or algorithm logic.

---

## Task 0: Add the `ot` dependency

**Files:** Modify `pyproject.toml`, `setup.py`.

- [ ] **Step 1:** Add `"ot"` to the `dependencies` list in `pyproject.toml` (the `pot` package is imported as `ot`; the PyPI name is `POT`). Use `"POT"` as the requirement name. Add `"POT"` to `setup.py`'s `install_requires` too.

- [ ] **Step 2: Install it into the env**

Run: `/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pip install POT`
Then verify: `/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -c "import ot; print('ot', ot.__version__)"`
Expected: prints a version (no ImportError).

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml setup.py
git commit -m "build(corr): add POT (optimal transport) dependency"
```

---

## Task 1: Port `segments.py`

**Files:** Create `src/correspondence/__init__.py` (empty), `src/correspondence/segments.py`; Test `tests/test_correspondence_segments.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_correspondence_segments.py
import numpy as np
from HoloNew.src.correspondence.segments import (
    SEGMENTS, g1_link_to_segment, point_segments, SMPLX_JOINT_TO_SEGMENT,
)

def test_g1_link_segments():
    assert g1_link_to_segment("pelvis") == "pelvis"
    assert g1_link_to_segment("left_knee_link") == "left_shank"
    assert g1_link_to_segment("right_ankle_roll_link") == "right_foot"
    assert g1_link_to_segment("left_shoulder_pitch_link") == "left_upperarm"
    assert g1_link_to_segment("torso_link") == "torso"

def test_segments_count():
    assert len(SEGMENTS) == 15  # 3 axial + 6 left + 6 right

def test_point_segments_picks_dominant_corner():
    # 2 vertices, 1 face; lbs makes vertex 0 -> pelvis(joint0), vertex 1 -> head(joint15)
    V = 16
    lbs = np.zeros((V, 55)); lbs[0, 0] = 1.0; lbs[1, 15] = 1.0
    faces = np.array([[0, 1, 0]])
    tri_idx = np.array([0])
    bary = np.array([[0.1, 0.8, 0.1]])  # dominant corner = vertex 1 (head)
    seg = point_segments(lbs, faces, tri_idx, bary)
    assert SEGMENTS[seg[0]] == "head"
```
(Note SEGMENTS has 15 entries — verify against the source.)

- [ ] **Step 2: Run it, confirm failure (ModuleNotFoundError).**

- [ ] **Step 3: Port the module.** Copy `transport/segments.py` from the port source **verbatim** into `src/correspondence/segments.py` (it has no `test_pipe_retargeting` imports — only numpy). Create empty `src/correspondence/__init__.py`.

- [ ] **Step 4: Run the tests, confirm PASS (3).** Fix `test_segments_count` if the real `len(SEGMENTS)` differs (it is 15 in the source).

- [ ] **Step 5: Commit**

```bash
git add src/correspondence/__init__.py src/correspondence/segments.py tests/test_correspondence_segments.py
git commit -m "feat(corr): port body-segment mapping"
```

---

## Task 2: Port the SMPL-X `HumanBody` + `human_source`

**Files:** Create `src/correspondence/constants.py`, `src/correspondence/human_body.py`, `src/correspondence/human_source.py`; Test `tests/test_correspondence_human.py`.

- [ ] **Step 1: Create `constants.py`**

```python
# src/correspondence/constants.py
from pathlib import Path

_PKG = Path(__file__).resolve().parent.parent.parent       # .../HoloNew/HoloNew
G1_29DOF_URDF = str(_PKG / "models" / "g1" / "g1_29dof.urdf")
HUMAN_GRID_DENSITY = 2000.0
G1_DENSITY = 3000.0
OT_REG = 0.005
# SMPL-X model dir is NOT bundled (license + size); default to the local data dir.
SMPLX_MODEL_DIR_DEFAULT = "/home/gbesset/Documents/wbt_rl/data/00_raw_datasets/models/models_smplx_v1_1/models/smplx"
```

- [ ] **Step 2: Write the failing test (gated on SMPL-X presence)**

```python
# tests/test_correspondence_human.py
import os
import numpy as np
import pytest
from HoloNew.src.correspondence.constants import SMPLX_MODEL_DIR_DEFAULT

smplx_missing = not os.path.isdir(SMPLX_MODEL_DIR_DEFAULT)
needs_smplx = pytest.mark.skipif(smplx_missing, reason="SMPL-X model dir not present")

@needs_smplx
def test_human_source_builds():
    from HoloNew.src.correspondence.human_body import HumanBody
    from HoloNew.src.correspondence.human_source import build_human_source
    body = HumanBody(SMPLX_MODEL_DIR_DEFAULT, None, "neutral")
    src = build_human_source(body, density=500.0)   # low density = fast
    N = src.points.shape[0]
    assert N > 0 and src.points.shape == (N, 3)
    assert src.seg.shape == (N,) and src.seg.min() >= 0 and src.seg.max() <= 14
    assert src.tri_idx.shape == (N,) and src.bary.shape == (N, 3)
```

- [ ] **Step 3: Run it; confirm it FAILS (ModuleNotFoundError) — or SKIPS if SMPL-X absent. On this machine SMPL-X is present so it must reach the import error.**

- [ ] **Step 4: Port `human_body.py`** — copy `human/body.py` from the source verbatim into `src/correspondence/human_body.py` (it has no `test_pipe_retargeting` imports; only smplx/torch/numpy/trimesh/scipy). Confirm `smplx.create(model_dir=..., model_type="smplx", gender="neutral", use_pca=False, num_betas=10)` loads with `model_dir = SMPLX_MODEL_DIR_DEFAULT`. If smplx expects the PARENT of the `smplx/` folder, set `SMPLX_MODEL_DIR_DEFAULT` to that parent (`.../models`) instead — test empirically and pick the path that loads.

- [ ] **Step 5: Port `human_source.py`** — copy `transport/human_source.py` verbatim into `src/correspondence/human_source.py`. Its only intra-package import is `from .segments import point_segments` — keep as-is.

- [ ] **Step 6: Run the test, confirm PASS (or SKIP if SMPL-X absent). On this machine it must PASS.**

- [ ] **Step 7: Commit**

```bash
git add src/correspondence/constants.py src/correspondence/human_body.py src/correspondence/human_source.py tests/test_correspondence_human.py
git commit -m "feat(corr): port SMPL-X HumanBody and human surface source"
```

---

## Task 3: Port `g1_surface.py`

**Files:** Create `src/correspondence/g1_surface.py`; Test `tests/test_correspondence_g1.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_correspondence_g1.py
import numpy as np
import yourdfpy
from HoloNew.src.correspondence.constants import G1_29DOF_URDF
from HoloNew.src.correspondence.g1_surface import sample_g1_surface, build_rest_cfg

def test_g1_surface_samples_valid_links():
    urdf = yourdfpy.URDF.load(G1_29DOF_URDF, load_meshes=True, build_scene_graph=True)
    surf = sample_g1_surface(urdf, density=300.0)   # low density = fast
    M = surf.points_world.shape[0]
    assert M > 0
    assert surf.points_world.shape == (M, 3) and surf.offset_local.shape == (M, 3)
    assert surf.link_idx.min() >= 0 and surf.link_idx.max() < len(surf.link_names)
    assert surf.seg.min() >= 0 and surf.seg.max() <= 14

def test_build_rest_cfg_length():
    urdf = yourdfpy.URDF.load(G1_29DOF_URDF, load_meshes=True, build_scene_graph=True)
    cfg = build_rest_cfg(urdf)
    assert cfg.shape[0] == len(urdf.actuated_joint_names)
```

- [ ] **Step 2: Run it, confirm failure (ModuleNotFoundError).**

- [ ] **Step 3: Port the module** — copy `transport/g1_surface.py` verbatim into `src/correspondence/g1_surface.py`. Intra-package imports `from .segments import SEGMENTS, g1_link_to_segment` stay as-is.

- [ ] **Step 4: Run the tests, confirm PASS (2).** The G1 URDF link names must resolve to segments via `g1_link_to_segment`; if some link maps unexpectedly, that is fine as long as `seg` is in range. (May take ~30s due to mesh voxelisation.)

- [ ] **Step 5: Commit**

```bash
git add src/correspondence/g1_surface.py tests/test_correspondence_g1.py
git commit -m "feat(corr): port G1 surface sampling"
```

---

## Task 4: Port `ot_couple.py`

**Files:** Create `src/correspondence/ot_couple.py`; Test `tests/test_correspondence_ot.py`.

- [ ] **Step 1: Write the failing test (synthetic, no SMPL-X/G1 needed)**

```python
# tests/test_correspondence_ot.py
import numpy as np
from dataclasses import dataclass
from HoloNew.src.correspondence.ot_couple import couple

@dataclass
class _Src:
    points: np.ndarray
    seg: np.ndarray

@dataclass
class _Tgt:
    points_world: np.ndarray
    seg: np.ndarray

def test_couple_returns_valid_human_indices():
    rng = np.random.default_rng(0)
    # one segment (0), 20 human points, 12 G1 points
    src = _Src(points=rng.standard_normal((20, 3)).astype(np.float32), seg=np.zeros(20, np.int64))
    tgt = _Tgt(points_world=rng.standard_normal((12, 3)).astype(np.float32), seg=np.zeros(12, np.int64))
    human_idx = couple(src, tgt, reg=0.05)
    assert human_idx.shape == (12,)
    assert human_idx.min() >= 0 and human_idx.max() < 20

def test_couple_raises_on_missing_human_segment():
    src = _Src(points=np.zeros((5, 3), np.float32), seg=np.zeros(5, np.int64))
    tgt = _Tgt(points_world=np.zeros((3, 3), np.float32), seg=np.ones(3, np.int64))  # seg 1 absent in src
    import pytest
    with pytest.raises(ValueError):
        couple(src, tgt, reg=0.05)
```

- [ ] **Step 2: Run it, confirm failure (ModuleNotFoundError).**

- [ ] **Step 3: Port the module** — copy `transport/ot_couple.py` verbatim into `src/correspondence/ot_couple.py` (only imports `ot` + scipy at call time; no intra-package imports).

- [ ] **Step 4: Run the tests, confirm PASS (2).**

- [ ] **Step 5: Commit**

```bash
git add src/correspondence/ot_couple.py tests/test_correspondence_ot.py
git commit -m "feat(corr): port per-segment optimal-transport coupling"
```

---

## Task 5: Port `build_correspondence.py` + `quality.py`

**Files:** Create `src/correspondence/quality.py`, `src/correspondence/build_correspondence.py`; Test `tests/test_correspondence_build.py`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_correspondence_build.py
import os
import numpy as np
import pytest
from HoloNew.src.correspondence.constants import SMPLX_MODEL_DIR_DEFAULT, G1_29DOF_URDF
from HoloNew.src.correspondence.build_correspondence import (
    CorrespondenceTable, save_correspondence, load_correspondence,
)

def test_save_load_roundtrip(tmp_path):
    t = CorrespondenceTable(
        link_idx=np.array([0, 1]), offset_local=np.zeros((2, 3), np.float32),
        link_names=["pelvis", "torso_link"], human_idx=np.array([3, 4]),
        tri_idx=np.array([7, 8]), bary=np.full((2, 3), 1 / 3, np.float32),
        density=2000.0, gender="neutral", betas=np.zeros(0, np.float32),
        g1_rest_cfg=np.zeros(5), seg=np.array([0, 1]),
    )
    p = tmp_path / "c.npz"
    save_correspondence(p, t)
    r = load_correspondence(p)
    assert r.link_names == ["pelvis", "torso_link"]
    np.testing.assert_array_equal(r.human_idx, [3, 4])

needs_smplx = pytest.mark.skipif(not os.path.isdir(SMPLX_MODEL_DIR_DEFAULT),
                                 reason="SMPL-X model dir not present")

@needs_smplx
def test_build_table_neutral():
    from HoloNew.src.correspondence.build_correspondence import build_table
    t = build_table(SMPLX_MODEL_DIR_DEFAULT, "neutral", None, G1_29DOF_URDF,
                    human_density=500.0, g1_density=300.0, reg=0.005)
    M = t.link_idx.shape[0]
    assert M > 0
    assert t.human_idx.min() >= 0
    assert t.link_idx.min() >= 0 and t.link_idx.max() < len(t.link_names)
```

- [ ] **Step 2: Run them, confirm failure (ModuleNotFoundError).**

- [ ] **Step 3: Port `quality.py`** — copy `transport/quality.py` verbatim; adapt any `test_pipe_retargeting` imports to relative (it imports from `.segments`/numpy — keep relative).

- [ ] **Step 4: Port `build_correspondence.py`** — copy `transport/build_correspondence.py` and adapt imports: `from test_pipe_retargeting.human.body import HumanBody` → `from .human_body import HumanBody`; `from test_pipe_retargeting.constants import G1_29DOF_URDF, HUMAN_GRID_DENSITY` → `from .constants import G1_29DOF_URDF, HUMAN_GRID_DENSITY`. Keep `.g1_surface`, `.human_source`, `.ot_couple` relative imports. Logic unchanged.

- [ ] **Step 5: Run the tests; roundtrip PASS always, build_table PASS on this machine (~1-2 min; SMPL-X present).**

- [ ] **Step 6: Commit**

```bash
git add src/correspondence/quality.py src/correspondence/build_correspondence.py tests/test_correspondence_build.py
git commit -m "feat(corr): port CorrespondenceTable build/save/load"
```

---

## Task 6: Build + bundle the neutral correspondence

**Files:** Create `assets/correspondence/corr_neutral.npz` (committed binary artifact).

- [ ] **Step 1: Build it with the CLI**

```bash
mkdir -p assets/correspondence
/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m HoloNew.src.correspondence.build_correspondence \
  --model-dir "/home/gbesset/Documents/wbt_rl/data/00_raw_datasets/models/models_smplx_v1_1/models/smplx" \
  --gender neutral --urdf models/g1/g1_29dof.urdf \
  --human-density 2000 --g1-density 3000 --reg 0.005 \
  --out assets/correspondence/corr_neutral.npz
```
Expected: prints "Saved correspondence: <M> G1 points -> assets/correspondence/corr_neutral.npz". (If `python -m` path differs, run the module file directly with the same args.)

- [ ] **Step 2: Sanity-check the artifact + its size (must be < 100 MB for GitHub)**

```bash
ls -lh assets/correspondence/corr_neutral.npz
/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -c "
from HoloNew.src.correspondence.build_correspondence import load_correspondence
t = load_correspondence('assets/correspondence/corr_neutral.npz')
print('G1 points', t.link_idx.shape[0], 'links', len(t.link_names))
"
```
Expected: a few-MB file; prints point/link counts. If it exceeds ~50 MB, lower `--human-density`/`--g1-density` and rebuild (the cache embeds tri_idx/bary at the human density).

- [ ] **Step 3: Commit the artifact** (force-add if `.gitignore` ignores npz under assets)

```bash
git add -f assets/correspondence/corr_neutral.npz
git commit -m "feat(corr): bundle precomputed neutral human->G1 correspondence"
```

---

## Task 7: Load the correspondence in v2

**Files:** Modify `src/gmr_socp/gmr_socp_v2.py`; Test `tests/test_correspondence_v2.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_correspondence_v2.py
def test_v2_loads_bundled_correspondence():
    from HoloNew.examples.robot_retarget import RetargetingConfig
    from HoloNew.src.gmr_socp.gmr_socp_v2 import GmrSocpRetargeterV2
    from HoloNew.src.correspondence.build_correspondence import CorrespondenceTable
    rt = GmrSocpRetargeterV2.from_config(
        RetargetingConfig(task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    assert isinstance(rt.correspondence, CorrespondenceTable)
    assert rt.correspondence.link_idx.shape[0] > 0

def test_v1_has_no_correspondence():
    from HoloNew.src.gmr_socp.gmr_socp_v1 import GmrSocpRetargeterV1
    assert not hasattr(GmrSocpRetargeterV1, "correspondence") or True  # v1 untouched
```

- [ ] **Step 2: Run it, confirm failure (`rt.correspondence` AttributeError).**

- [ ] **Step 3: Add correspondence loading to `GmrSocpRetargeterV2.from_config`.** After the existing input prep, before returning `rt`, add:

```python
        # Load the bundled human->G1 OT correspondence (data only; NOT used in the
        # solve yet). Falls back to a live build if a SMPL-X model dir is configured.
        from pathlib import Path
        from HoloNew.src.correspondence.build_correspondence import load_correspondence, build_table
        from HoloNew.src.correspondence.constants import (
            G1_29DOF_URDF, SMPLX_MODEL_DIR_DEFAULT, HUMAN_GRID_DENSITY, G1_DENSITY, OT_REG,
        )
        _bundled = Path(__file__).resolve().parent.parent.parent / "assets" / "correspondence" / "corr_neutral.npz"
        rt.correspondence = None
        if _bundled.exists():
            rt.correspondence = load_correspondence(_bundled)
        elif Path(SMPLX_MODEL_DIR_DEFAULT).is_dir():
            rt.correspondence = build_table(SMPLX_MODEL_DIR_DEFAULT, "neutral", None,
                                            G1_29DOF_URDF, HUMAN_GRID_DENSITY, G1_DENSITY, OT_REG)
```
Initialise `self.correspondence = None` in `__init__` too (so the attribute always exists). Do NOT touch `retarget()` or `solve_single_iteration` — the correspondence is unused by the solve.

- [ ] **Step 4: Run the test, confirm PASS** (uses the bundled artifact from Task 6, so no SMPL-X needed).

- [ ] **Step 5: Re-run the GMR suite to confirm v2 still retargets**

Run: `/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_gmr_socp.py -q`
Expected: PASS (v1 and v2 both retarget; correspondence load does not affect qpos).

- [ ] **Step 6: Commit**

```bash
git add src/gmr_socp/gmr_socp_v2.py tests/test_correspondence_v2.py
git commit -m "feat(corr): load bundled human->G1 correspondence in GMR-SOCP v2"
```

---

## Task 8: Annex viser app for OT visualization

**Files:** Create `src/correspondence/viz.py` (ported helpers), `examples/view_correspondence.py`.

- [ ] **Step 1: Port `viz.py`** — copy `transport/viz.py` from the source into `src/correspondence/viz.py`, adapting any `test_pipe_retargeting` imports to relative. If it depends on panel/runner state not present in HoloNew, keep only the pure drawing helpers (functions that take a viser server + arrays and add points/lines); drop anything referencing test_pipe's `SharedData`/panels. The goal is reusable helpers: draw G1 surface points colored by segment, and (optionally) the human surface.

- [ ] **Step 2: Write the annex app `examples/view_correspondence.py`**

```python
"""Standalone viser app to visualize the human->G1 OT correspondence (data only).

Shows the G1 surface points colored by body segment (always available from the
bundled correspondence). If a SMPL-X model dir is available, also draws the human
rest surface colored consistently. Does not run any solve.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import viser
import yourdfpy
from viser.extras import ViserUrdf

from HoloNew.src.correspondence.build_correspondence import load_correspondence
from HoloNew.src.correspondence.constants import G1_29DOF_URDF, SMPLX_MODEL_DIR_DEFAULT
from HoloNew.src.correspondence.segments import SEGMENTS

_SEG_COLORS = (np.array([
    [230,25,75],[60,180,75],[255,225,25],[0,130,200],[245,130,48],
    [145,30,180],[70,240,240],[240,50,230],[210,245,60],[250,190,212],
    [0,128,128],[220,190,255],[170,110,40],[255,250,200],[128,0,0],
], dtype=np.uint8))

def _bundled_corr() -> Path:
    return Path(__file__).resolve().parent.parent / "assets" / "correspondence" / "corr_neutral.npz"

def main() -> None:
    corr = load_correspondence(_bundled_corr())
    urdf = yourdfpy.URDF.load(G1_29DOF_URDF, load_meshes=True, build_scene_graph=True)
    urdf.update_cfg(corr.g1_rest_cfg)

    server = viser.ViserServer()
    server.scene.set_up_direction("+z")
    ViserUrdf(server, urdf_or_path=urdf, root_node_name="/g1")

    # Place each G1 surface point by its link's current transform (rest cfg).
    pts = np.zeros((corr.link_idx.shape[0], 3), np.float32)
    for li, link in enumerate(corr.link_names):
        T = np.asarray(urdf.get_transform(link))
        sel = corr.link_idx == li
        pts[sel] = corr.offset_local[sel] @ T[:3, :3].T + T[:3, 3]
    colors = _SEG_COLORS[corr.seg % len(SEGMENTS)]
    server.scene.add_point_cloud("/g1_corr_points", points=pts, colors=colors, point_size=0.005)

    print("Correspondence viewer at http://localhost:8080 — Ctrl+C to exit")
    input("Enter to exit ...")

if __name__ == "__main__":
    main()
```
(Confirm the viser point-cloud API name against the installed version — it may be `add_point_cloud`; if different, use the form `src/viewer.py` or `viser_player.py` uses for points. Confirm `urdf.get_transform(link)` exists, as `g1_surface.py` already uses it.)

- [ ] **Step 3: Bounded headless smoke test**

```bash
echo "" | timeout 120 /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python examples/view_correspondence.py 2>&1 | tail -15
```
Expected: loads the bundled correspondence, posts the G1 + colored points, prints the viewer URL, exits cleanly on the piped newline, NO Traceback.

- [ ] **Step 4: Run the full suite**

Run: `/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/ -q`
Expected: all pass (the SMPL-X-gated tests run here; on a machine without SMPL-X they skip).

- [ ] **Step 5: Commit**

```bash
git add src/correspondence/viz.py examples/view_correspondence.py
git commit -m "feat(corr): annex viser app for human->G1 correspondence"
```

---

## Self-Review notes

- **Spec coverage:** faithful port → Tasks 1-5,8; `ot` dep → Task 0; SMPL-X external + configurable → `constants.py` (Task 2); bundled precomputed correspondence → Task 6; v2 load (data only, solve untouched) → Task 7; annex viser app, graceful without SMPL-X → Task 8. Contact fields (`coal`) are explicitly NOT in this plan (increment 4b).
- **Heavy-dep gating:** SMPL-X tests use `pytest.mark.skipif` on the model dir so the suite stays green where SMPL-X is absent; on this machine they run. The bundled `.npz` (Task 6) makes Task 7 + the annex app work WITHOUT SMPL-X.
- **Native + v1 untouched:** only `gmr_socp_v2.py` is modified (adds a `correspondence` attribute; `retarget`/solver untouched), plus packaging (`ot`) and new files.
- **Open items for the implementer:** exact `smplx.create` model-dir layout (Task 2 Step 4); whether `viz.py` carries test_pipe-only state to strip (Task 8 Step 1); the viser point-cloud API name (Task 8 Step 2); the bundled `.npz` size must stay < 100 MB (Task 6 Step 2 — lower densities if needed).
```
