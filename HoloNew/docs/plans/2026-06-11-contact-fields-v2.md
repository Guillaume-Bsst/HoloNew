# Contact Fields (v2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port test_pipe's SDF contact-field path into HoloNew, attach it to the v2 GMR-SOCP retargeter (data only, not used in the solve), bundle a precomputed object SDF + demo contact field so it is turnkey, and visualize it in a dedicated annex viser app with a frame slider.

**Architecture:** A new `src/contact/` package faithfully ports test_pipe's `fields/` SDF path (`contact_field`, `probes`, `object_input`, `backends/coal`, `backends/sdf`, `backends/floor`, `combined`) plus a small numpy `contact_io` (save/load the per-frame contact channels) and a `motion` loader (joints/obj_poses/quats from the `.pt`). The object SDF is built once offline (with `coal`) and bundled; the demo sequence's per-frame contact is precomputed (with SMPL-X) and bundled. `GmrSocpRetargeterV2.from_config` loads both; `examples/view_contact.py` shows them per frame. SMPL-X is external (reused from the 4a package); coal is needed only to build the SDF offline.

**Tech Stack:** Python, numpy, scipy, `coal` (3.0.3 pip wheels), `smplx`, trimesh, `joblib`, viser, pytest. Runs in the `holonew` conda env.

**Reference spec:** `docs/specs/2026-06-11-contact-fields-v2-design.md`
**Port source (read-only):** `/home/gbesset/Documents/wbt_rl/modules/third_party/test_pipe/src/test_pipe_retargeting/test_pipe_retargeting/` (its `fields/`, `human/motion.py`).

## Critical environment (every task)
- Repo root: `/home/gbesset/Documents/wbt_rl/modules/01_retargeting/HoloNew`; package dir (cwd): `.../HoloNew/HoloNew`.
- Package imports as `HoloNew`. Work on a feature branch off `main` (controller creates it).
- **Always use this Python:** `/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python` (never bare python).
- Commits: identity `Guillaume-Bsst`; `git commit` normally; **never add Co-Authored-By/Claude or any Claude mention**; comments/docs in **English**.
- From 4a (already on main): `src/correspondence/human_body.py` (`HumanBody`, `PointCloudCache`), `src/correspondence/constants.py` (`SMPLX_MODEL_DIR_DEFAULT` = parent-of-smplx dir), `src/correspondence/targets.py` (`load_pt_quaternions`), `assets/correspondence/corr_neutral.npz`. `gmr_socp_v2.py` already loads `self.correspondence`.
- OMOMO object data (for the largebox mesh): `/home/gbesset/Documents/wbt_rl/data/00_raw_datasets/OMOMO` (has `data/captured_objects/largebox_cleaned_simplified.obj` + `data/*_diffusion_manip_seq_joints24.p`).
- Demo `.pt`: `demo_data/OMOMO_new/sub3_largebox_003.pt` (T=196). Its layout (verified in 4a): joints at `[162:162+52*3]` (T,52,3); object pose at `[318:325]` (`trans(3), quat_xyzw(4)`); per-joint quats at `[383:383+52*4]` (xyzw, upright_start-baked).

**Import adaptation rule (every ported file):** replace `from ..human.body import HumanBody` → `from HoloNew.src.correspondence.human_body import HumanBody`; `from ..constants import X` → `from .constants import X`; keep intra-package relatives (`.backends.coal`, `.contact_field`, …). Change NO numbers/logic.

## File Structure (all additive; native + v1 untouched)
- `src/contact/__init__.py`, `src/contact/constants.py`
- `src/contact/contact_field.py`, `src/contact/contact_io.py` (NEW), `src/contact/motion.py` (port of load_pt)
- `src/contact/probes.py`, `src/contact/object_input.py`, `src/contact/combined.py`
- `src/contact/backends/__init__.py`, `src/contact/backends/coal.py`, `src/contact/backends/sdf.py`, `src/contact/backends/floor.py`
- `assets/contact/largebox_sdf.npz`, `assets/contact/contact_sub3_largebox_003.npz`
- Modify: `pyproject.toml`/`setup.py` (`coal`), `src/gmr_socp/gmr_socp_v2.py` (load contact)
- Create: `examples/view_contact.py`
- Tests: `tests/test_contact_*.py`

---

## Task 0: Add the `coal` dependency

**Files:** Modify `../pyproject.toml`, `../setup.py` (repo root, one level above the package dir).

- [ ] **Step 1:** Add `"coal"` to `[project] dependencies` in `../pyproject.toml` and to `install_requires` in `../setup.py`.
- [ ] **Step 2: Install + verify**
```bash
/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pip install coal
/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -c "import coal; print('coal OK', getattr(coal,'__version__','?'))"
```
Expected: prints OK, no ImportError. (These are manylinux wheels — no compilation.)
- [ ] **Step 3: Commit**
```bash
cd /home/gbesset/Documents/wbt_rl/modules/01_retargeting/HoloNew
git add pyproject.toml setup.py && git commit -m "build(contact): add coal dependency"
cd HoloNew
```

---

## Task 1: constants + contact_field + floor (numpy core)

**Files:** Create `src/contact/__init__.py`, `src/contact/constants.py`, `src/contact/contact_field.py`, `src/contact/backends/__init__.py`, `src/contact/backends/floor.py`; Test `tests/test_contact_field.py`.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_contact_field.py
import numpy as np
from HoloNew.src.contact.contact_field import ContactField
from HoloNew.src.contact.backends.floor import floor_field

def test_floor_field_signs_and_active():
    pts = np.array([[0, 0, -0.01], [0, 0, 0.02], [0, 0, 1.0]], float)
    f = floor_field(pts, margin=0.05)
    assert f.active.tolist() == [True, True, False]      # within 0.05 of z=0
    assert f.distance[2] == np.float32(0.05)             # inactive clamped to +margin
    assert f.direction[0, 2] == -1.0 and f.direction[1, 2] == 1.0

def test_contact_field_is_frozen_dataclass():
    f = ContactField(distance=np.zeros(2), direction=np.zeros((2, 3)),
                     witness=np.zeros((2, 3)), active=np.zeros(2, bool))
    import dataclasses, pytest
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.distance = np.ones(2)
```

- [ ] **Step 2: Run it, confirm failure (ModuleNotFoundError).**

- [ ] **Step 3: Create the modules**
- `src/contact/__init__.py`, `src/contact/backends/__init__.py`: empty.
- `src/contact/constants.py`:
```python
CONTACT_MARGIN_M = 0.05
OBJECT_FIELD_RESOLUTION = 0.01
FLOOR_GRID_SIZE = 4.0
FLOOR_GRID_DENSITY = 400.0
OBJECT_GRID_DENSITY = 3000.0
# OMOMO raw object data (for the largebox mesh) — external, not bundled.
OMOMO_DIR_DEFAULT = "/home/gbesset/Documents/wbt_rl/data/00_raw_datasets/OMOMO"
```
(Use the source's values if `constants.py` in test_pipe defines different numbers for `FLOOR_GRID_SIZE`/`*_DENSITY`/`OBJECT_FIELD_RESOLUTION`/`CONTACT_MARGIN_M` — read `test_pipe_retargeting/constants.py` and copy those exact values.)
- `src/contact/contact_field.py`: copy `fields/contact_field.py` VERBATIM (its only `coal` import is lazy inside `_probe_distance`; module-top is numpy only).
- `src/contact/backends/floor.py`: copy `fields/backends/floor.py` VERBATIM (imports `from ..contact_field import ContactField` — keep relative).

- [ ] **Step 4: Run the tests, confirm PASS (2).**

- [ ] **Step 5: Commit**
```bash
git add src/contact/__init__.py src/contact/constants.py src/contact/contact_field.py src/contact/backends/__init__.py src/contact/backends/floor.py tests/test_contact_field.py
git commit -m "feat(contact): port ContactField core + analytic floor field"
```

---

## Task 2: contact I/O + motion loader (numpy, turnkey)

`contact_io` lets us bundle/load the per-frame contact channels without coal. `motion` loads joints/obj_poses/quats from the `.pt`.

**Files:** Create `src/contact/contact_io.py`, `src/contact/motion.py`; Test `tests/test_contact_io.py`.

- [ ] **Step 1: Write the failing tests**
```python
# tests/test_contact_io.py
import numpy as np
from HoloNew.src.contact.contact_field import ContactField
from HoloNew.src.contact.contact_io import save_contact_fields, load_contact_fields

def _cf(T, N):
    return ContactField(distance=np.zeros((T, N), np.float32), direction=np.zeros((T, N, 3), np.float32),
                        witness=np.zeros((T, N, 3), np.float32), active=np.zeros((T, N), bool))

def test_contact_fields_roundtrip(tmp_path):
    fields = {"human_floor": _cf(3, 5), "human_object": _cf(3, 5)}
    p = tmp_path / "c.npz"
    save_contact_fields(p, fields)
    r = load_contact_fields(p)
    assert set(r) == {"human_floor", "human_object"}
    assert r["human_floor"].distance.shape == (3, 5)
    assert isinstance(r["human_object"], ContactField)

def test_motion_loads_demo_shapes():
    from HoloNew.src.contact.motion import load_pt_motion
    joints, obj_poses, quats = load_pt_motion("demo_data/OMOMO_new/sub3_largebox_003.pt")
    assert joints.shape[1:] == (52, 3)
    assert obj_poses.shape[1] == 7
    assert quats.shape[1:] == (52, 4)
    assert joints.shape[0] == obj_poses.shape[0] == quats.shape[0]
```

- [ ] **Step 2: Run them, confirm failure.**

- [ ] **Step 3: Implement `src/contact/contact_io.py`**
```python
"""Save/load per-frame contact channels (dict[str, ContactField]) as a numpy .npz.

Pure numpy — lets the bundled demo contact be loaded without coal or SMPL-X.
Each channel's four (T, ...)-stacked arrays are stored under prefixed keys.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .contact_field import ContactField


def save_contact_fields(path, fields: dict[str, ContactField]) -> None:
    arrays = {"channels": np.array(list(fields), dtype="<U32")}
    for name, f in fields.items():
        arrays[f"{name}/distance"] = f.distance
        arrays[f"{name}/direction"] = f.direction
        arrays[f"{name}/witness"] = f.witness
        arrays[f"{name}/active"] = f.active
    np.savez_compressed(str(Path(path)), **arrays)


def load_contact_fields(path) -> dict[str, ContactField]:
    d = np.load(str(Path(path)))
    out = {}
    for name in (str(c) for c in d["channels"]):
        out[name] = ContactField(
            distance=d[f"{name}/distance"], direction=d[f"{name}/direction"],
            witness=d[f"{name}/witness"], active=d[f"{name}/active"],
        )
    return out
```

- [ ] **Step 4: Implement `src/contact/motion.py`** by porting test_pipe's `human/motion.py` `load_pt`. Read that file for the exact slices and the `_undo_upright_start` handling, then expose `load_pt_motion(pt_path) -> (joints (T,52,3), obj_poses (T,7) [qw,qx,qy,qz,x,y,z], quats (T,52,4) wxyz)`. Reuse the quaternion handling already proven in `src/correspondence/targets.py:load_pt_quaternions` (slice `[383:383+52*4]`, `_undo_upright_start`, reorder `[3,0,1,2]`). joints = `data[:, 162:162+52*3].reshape(-1,52,3)`; obj_poses = reorder `data[:, 318:325]` (`[trans(3), quat_xyzw(4)]`) into `[qw,qx,qy,qz,x,y,z]` exactly as the source does.

- [ ] **Step 5: Run the tests, confirm PASS (2).**

- [ ] **Step 6: Commit**
```bash
git add src/contact/contact_io.py src/contact/motion.py tests/test_contact_io.py
git commit -m "feat(contact): numpy contact-field I/O and .pt motion loader"
```

---

## Task 3: port coal backend + probes + object_input

**Files:** Create `src/contact/backends/coal.py`, `src/contact/probes.py`, `src/contact/object_input.py`; Test `tests/test_contact_backends.py`.

- [ ] **Step 1: Write the failing tests**
```python
# tests/test_contact_backends.py
import numpy as np
from HoloNew.src.contact.object_input import parse_obj_name
from HoloNew.src.contact.probes import make_floor_grid, make_object_grid

def test_parse_obj_name():
    assert parse_obj_name("sub3_largebox_003") == "largebox"

def test_make_floor_grid():
    g = make_floor_grid(center_xy=(0.0, 0.0))
    assert g.ndim == 2 and g.shape[1] == 3
    assert np.allclose(g[:, 2], 0.0)

def test_coal_build_bvh_smoke():
    import trimesh
    from HoloNew.src.contact.backends.coal import build_bvh
    box = trimesh.creation.box(extents=(1, 1, 1))
    bvh = build_bvh(np.asarray(box.vertices), np.asarray(box.faces))
    assert bvh is not None
```

- [ ] **Step 2: Run them, confirm failure.**

- [ ] **Step 3: Port the modules**
- `src/contact/backends/coal.py`: copy `fields/backends/coal.py` VERBATIM (imports `from ..contact_field import ...` — keep relative; uses `coal`).
- `src/contact/probes.py`: copy `fields/probes.py`, adapt `from ..constants import FLOOR_GRID_DENSITY, FLOOR_GRID_SIZE, OBJECT_GRID_DENSITY` → `from .constants import ...`.
- `src/contact/object_input.py`: copy `fields/object_input.py` VERBATIM (uses `joblib`/`trimesh` lazily; no test_pipe imports). If `joblib` is not installed, `pip install joblib` into the env and report it.

- [ ] **Step 4: Run the tests, confirm PASS (3).**

- [ ] **Step 5: Commit**
```bash
git add src/contact/backends/coal.py src/contact/probes.py src/contact/object_input.py tests/test_contact_backends.py
git commit -m "feat(contact): port coal backend, probe grids, object input"
```

---

## Task 4: port the SDF backend

**Files:** Create `src/contact/backends/sdf.py`; Test `tests/test_contact_sdf.py`.

- [ ] **Step 1: Write the failing tests**
```python
# tests/test_contact_sdf.py
import numpy as np
import trimesh
from HoloNew.src.contact.backends.sdf import (
    build_object_field, sdf_surface_field, save_object_sdf, load_object_sdf,
)

def test_build_and_query_box_sdf(tmp_path):
    box = trimesh.creation.box(extents=(0.4, 0.4, 0.4))
    sdf = build_object_field(box, margin=0.05, resolution=0.02)
    # a point on the +x face is ~0 distance, a far point is clamped to +margin
    f = sdf_surface_field(np.array([[0.25, 0.0, 0.0], [5.0, 5.0, 5.0]], float), sdf, margin=0.05)
    assert f.distance.shape == (2,)
    assert f.active[0] and not f.active[1]
    assert f.distance[1] == np.float32(0.05)
    p = tmp_path / "sdf.npz"
    save_object_sdf(sdf, p)
    r = load_object_sdf(p)
    np.testing.assert_array_equal(r.data, sdf.data)
```

- [ ] **Step 2: Run it, confirm failure.**

- [ ] **Step 3: Port the module** — copy `fields/backends/sdf.py` VERBATIM into `src/contact/backends/sdf.py`. It imports `from ..contact_field import ContactField` and `from .coal import build_bvh, surface_field` (keep relative). `build_object_field` uses coal; `sdf_surface_field`/`save`/`load` are numpy.

- [ ] **Step 4: Run the tests, confirm PASS.** (Build over a small box at 2cm resolution is fast.)

- [ ] **Step 5: Commit**
```bash
git add src/contact/backends/sdf.py tests/test_contact_sdf.py
git commit -m "feat(contact): port object SDF backend (build/query/save/load)"
```

---

## Task 5: port the combined per-frame driver

**Files:** Create `src/contact/combined.py`; Test `tests/test_contact_combined.py`.

`combined.py` has a test fallback: when `human_body_params=None` and `hverts`/`hprobes` are supplied, it skips SMPL-X — so SDF mode is unit-testable without a model.

- [ ] **Step 1: Write the failing test (SDF mode, no SMPL-X)**
```python
# tests/test_contact_combined.py
import numpy as np
import trimesh
from HoloNew.src.contact.backends.sdf import build_object_field
from HoloNew.src.contact.combined import compute_contact_fields

def test_compute_contact_fields_sdf_mode():
    box = trimesh.creation.box(extents=(0.4, 0.4, 0.4))
    sdf = build_object_field(box, margin=0.05, resolution=0.02)
    T, N = 2, 6
    hverts = np.zeros((T, N, 3), float)       # unused by floor/sdf channels beyond probes
    hprobes = np.random.default_rng(0).standard_normal((T, N, 3)) * 0.1
    quats = np.tile(np.array([1.0, 0, 0, 0]), (T, 52, 1))
    pelvises = np.zeros((T, 3))
    obj_poses = np.tile(np.array([1.0, 0, 0, 0, 0, 0, 0]), (T, 1))
    out = compute_contact_fields(
        T=T, quats=quats, pelvises=pelvises, human_faces=np.zeros((0, 3), int),
        human_body_params=None, human_pc_cache=None, object_mesh=box,
        object_grid_local=None, obj_poses=obj_poses,
        floor_grid=np.zeros((4, 3), float), margin=0.05,
        hverts=hverts, hprobes=hprobes, object_sdf=sdf,
    )
    assert set(out) == {"human_floor", "human_object"}        # SDF mode = human-side only
    assert out["human_floor"].distance.shape == (T, N)
    assert out["human_object"].distance.shape == (T, N)
```

- [ ] **Step 2: Run it, confirm failure.**

- [ ] **Step 3: Port the module** — copy `fields/combined.py` into `src/contact/combined.py` and adapt: `from ..human.body import HumanBody` → `from HoloNew.src.correspondence.human_body import HumanBody`; keep `from .backends.coal import ...`, `from .backends.floor import floor_field`, `from .backends.sdf import sdf_surface_field`, `from .contact_field import ...` relative. Logic unchanged.

- [ ] **Step 4: Run the test, confirm PASS.**

- [ ] **Step 5: Commit**
```bash
git add src/contact/combined.py tests/test_contact_combined.py
git commit -m "feat(contact): port per-frame contact-field driver (SDF mode)"
```

---

## Task 6: build + bundle the object SDF

**Files:** Create `assets/contact/largebox_sdf.npz`; Test (smoke, no new test file required — verification inline).

- [ ] **Step 1: Build the largebox SDF** (loads the OMOMO mesh for correct scale/centering)
```bash
mkdir -p assets/contact
/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python - <<'PY'
from pathlib import Path
from HoloNew.src.contact.object_input import load_mesh
from HoloNew.src.contact.backends.sdf import build_object_field, save_object_sdf
from HoloNew.src.contact.constants import OMOMO_DIR_DEFAULT, CONTACT_MARGIN_M, OBJECT_FIELD_RESOLUTION
res = load_mesh(Path(OMOMO_DIR_DEFAULT), "largebox")
assert res is not None, "largebox mesh not found"
mesh, scale, origin = res
sdf = build_object_field(mesh, margin=CONTACT_MARGIN_M, resolution=OBJECT_FIELD_RESOLUTION)
save_object_sdf(sdf, "assets/contact/largebox_sdf.npz")
print("dims", sdf.dims.tolist())
PY
ls -lh assets/contact/largebox_sdf.npz
```
Expected: prints the SDF grid dims and a file size. **The file must be < 100 MB.** If it is large (a 1cm SDF over a big box can be tens of MB), increase `resolution` (e.g. 0.02) and rebuild until comfortably under 100 MB; record the resolution used.

- [ ] **Step 2: Verify it loads + queries**
```bash
/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -c "
import numpy as np
from HoloNew.src.contact.backends.sdf import load_object_sdf, sdf_surface_field
sdf = load_object_sdf('assets/contact/largebox_sdf.npz')
f = sdf_surface_field(np.zeros((3,3)), sdf, 0.05)
print('ok', f.distance.shape)
"
```

- [ ] **Step 3: Commit the artifact**
```bash
git add -f assets/contact/largebox_sdf.npz
git commit -m "feat(contact): bundle precomputed largebox object SDF"
```

---

## Task 7: build + bundle the demo contact field

**Files:** Create `assets/contact/contact_sub3_largebox_003.npz`.

- [ ] **Step 1: Build the demo per-frame contact** (SMPL-X + the bundled SDF + correspondence cache)
```bash
/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python - <<'PY'
from pathlib import Path
import numpy as np
from HoloNew.src.contact.motion import load_pt_motion
from HoloNew.src.contact.object_input import load_mesh
from HoloNew.src.contact.probes import make_floor_grid, make_object_grid
from HoloNew.src.contact.backends.sdf import load_object_sdf
from HoloNew.src.contact.combined import compute_contact_fields
from HoloNew.src.contact.contact_io import save_contact_fields
from HoloNew.src.contact.constants import OMOMO_DIR_DEFAULT, CONTACT_MARGIN_M
from HoloNew.src.correspondence.human_body import HumanBody, PointCloudCache
from HoloNew.src.correspondence.build_correspondence import load_correspondence
from HoloNew.src.correspondence.constants import SMPLX_MODEL_DIR_DEFAULT

joints, obj_poses, quats = load_pt_motion("demo_data/OMOMO_new/sub3_largebox_003.pt")
T = joints.shape[0]
pelvises = joints[:, 0]
mesh, _, _ = load_mesh(Path(OMOMO_DIR_DEFAULT), "largebox")
sdf = load_object_sdf("assets/contact/largebox_sdf.npz")
corr = load_correspondence("assets/correspondence/corr_neutral.npz")
pc_cache = PointCloudCache(tri_idx=corr.tri_idx, bary=corr.bary)   # keyed 1:1 to correspondence
body = HumanBody(SMPLX_MODEL_DIR_DEFAULT, None, "neutral")
fields = compute_contact_fields(
    T=T, quats=quats, pelvises=pelvises, human_faces=body.faces.astype(np.int64),
    human_body_params={"model_dir": SMPLX_MODEL_DIR_DEFAULT, "betas": None, "gender": "neutral"},
    human_pc_cache=pc_cache, object_mesh=mesh, object_grid_local=make_object_grid(mesh),
    obj_poses=obj_poses, floor_grid=make_floor_grid(center_xy=(float(joints[:,0,0].mean()), float(joints[:,0,1].mean()))),
    margin=CONTACT_MARGIN_M, object_sdf=sdf, progress=True,
)
save_contact_fields("assets/contact/contact_sub3_largebox_003.npz", fields)
print({k: v.distance.shape for k, v in fields.items()})
PY
ls -lh assets/contact/contact_sub3_largebox_003.npz
```
Expected: prints `{human_floor:(196,N), human_object:(196,N)}` and a file size. **Must be < 100 MB.** If large, reduce the human point-cloud density (rebuild the correspondence cache at a lower density is out of scope; instead, if needed, subsample the probes — but first just check the size: with N≈3673 and T=196, four float arrays are ~3673*196*(1+3+3+1)*4 bytes ≈ 25 MB before compression, comfortably under the limit).

- [ ] **Step 2: Verify it loads (numpy only, no coal/SMPL-X)**
```bash
/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -c "
from HoloNew.src.contact.contact_io import load_contact_fields
f = load_contact_fields('assets/contact/contact_sub3_largebox_003.npz')
print({k: v.distance.shape for k,v in f.items()})
"
```

- [ ] **Step 3: Commit the artifact**
```bash
git add -f assets/contact/contact_sub3_largebox_003.npz
git commit -m "feat(contact): bundle precomputed demo contact field (sub3_largebox_003)"
```

---

## Task 8: load contact in v2

**Files:** Modify `src/gmr_socp/gmr_socp_v2.py`; Test `tests/test_contact_v2.py`.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_contact_v2.py
def test_v2_loads_bundled_contact():
    from HoloNew.examples.robot_retarget import RetargetingConfig
    from HoloNew.src.gmr_socp.gmr_socp_v2 import GmrSocpRetargeterV2
    from HoloNew.src.contact.contact_field import ContactField
    rt = GmrSocpRetargeterV2.from_config(
        RetargetingConfig(task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    assert rt.object_sdf is not None
    assert isinstance(rt.contact_fields, dict)
    assert "human_object" in rt.contact_fields
    assert isinstance(rt.contact_fields["human_object"], ContactField)
```

- [ ] **Step 2: Run it, confirm failure (`rt.object_sdf` AttributeError).**

- [ ] **Step 3: Add contact loading to `GmrSocpRetargeterV2`.**
- In `__init__`, initialise `self.object_sdf = None` and `self.contact_fields = None`.
- In `from_config`, after the correspondence-loading block and before `return rt`, add:
```python
        from HoloNew.src.contact.backends.sdf import load_object_sdf
        from HoloNew.src.contact.contact_io import load_contact_fields
        _assets = Path(__file__).resolve().parent.parent.parent / "assets" / "contact"
        _sdf_path = _assets / "largebox_sdf.npz"
        _contact_path = _assets / f"contact_{cfg.task_name}.npz"
        if _sdf_path.exists():
            rt.object_sdf = load_object_sdf(_sdf_path)
        if _contact_path.exists():
            rt.contact_fields = load_contact_fields(_contact_path)
```
(`Path` is already imported in v2 from the 4a change. Do NOT touch `retarget()`/solver. The contact is data only.)

- [ ] **Step 4: Run the test, confirm PASS** (uses the bundled artifacts; no coal/SMPL-X needed).

- [ ] **Step 5: Confirm v2 still retargets + golden intact**
```bash
/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_gmr_socp.py tests/test_retarget_golden.py tests/test_correspondence_v2.py -q
```
Expected: PASS. Confirm `git status` shows only `gmr_socp_v2.py` + the new test changed (v1/native untouched).

- [ ] **Step 6: Commit**
```bash
git add src/gmr_socp/gmr_socp_v2.py tests/test_contact_v2.py
git commit -m "feat(contact): load bundled object SDF + demo contact in GMR-SOCP v2"
```

---

## Task 9: annex viser app for the contact field

**Files:** Create `examples/view_contact.py`.

- [ ] **Step 1: Write the annex app**
```python
"""Standalone viser app to visualize the per-frame contact field (data only).

Loads the bundled demo contact field and shows, for the selected frame, the human
contact probes colored by their signed distance to the object (human_object channel),
red = penetrating/touching, blue = far. Uses only numpy (no coal/SMPL-X). A frame
slider scrubs time. Does not run any solve.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import viser

from HoloNew.src.contact.contact_io import load_contact_fields
from HoloNew.src.contact.constants import CONTACT_MARGIN_M

def _bundled() -> Path:
    return Path(__file__).resolve().parent.parent / "assets" / "contact" / "contact_sub3_largebox_003.npz"

def _color(dist: np.ndarray, margin: float) -> np.ndarray:
    # near (small/negative) -> red, far (+margin) -> blue
    t = np.clip(dist / margin, 0.0, 1.0)
    c = np.zeros((dist.shape[0], 3), np.uint8)
    c[:, 0] = ((1 - t) * 255).astype(np.uint8)
    c[:, 2] = (t * 255).astype(np.uint8)
    return c

def main() -> None:
    fields = load_contact_fields(_bundled())
    ch = fields["human_object"]
    T = ch.distance.shape[0]
    server = viser.ViserServer()
    server.scene.set_up_direction("+z")
    slider = server.gui.add_slider("Frame", min=0, max=T - 1, step=1, initial_value=0)

    def redraw(t: int) -> None:
        # witness points sit on the object surface; probes carry the contact value.
        pts = ch.witness[t]
        colors = _color(ch.distance[t], CONTACT_MARGIN_M)
        server.scene.add_point_cloud("/contact", points=pts.astype(np.float32), colors=colors, point_size=0.01)

    slider.on_update(lambda _evt: redraw(int(slider.value)))
    redraw(0)
    print("Contact viewer at http://localhost:8080 — Enter to exit ...")
    input("Enter to exit ...")

if __name__ == "__main__":
    main()
```
Confirm `server.gui.add_slider`, `.on_update`, and `add_point_cloud` match the installed viser (the 4a annex app + `src/viewer.py` use these). If `witness` is all-zero for inactive probes, that is expected (the colored cloud still shows the active band). Adapt the visualized point set if `human_object.witness` is unhelpful — the probe positions are not stored in the contact field, so witness (object-surface closest points) is the available per-frame geometry; document the choice.

- [ ] **Step 2: Bounded headless smoke test**
```bash
echo "" | timeout 90 /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python examples/view_contact.py 2>&1 | tail -15
```
Expected: loads the bundled contact, builds the scene, prints the viewer URL, exits cleanly on the piped newline, NO Traceback.

- [ ] **Step 3: Full suite**
```bash
/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/ -q
```
Expected: all pass.

- [ ] **Step 4: Commit**
```bash
git add examples/view_contact.py
git commit -m "feat(contact): annex viser app for the per-frame contact field"
```

---

## Self-Review notes

- **Spec coverage:** SDF path port → Tasks 1,3,4,5; `coal` dep → Task 0; numpy contact I/O (turnkey load) → Task 2; bundled object SDF → Task 6; bundled demo contact → Task 7; v2 load (data only, solve untouched) → Task 8; annex app → Task 9. The differentiable `sdf_surface_distance_torch` rides along in Task 4's verbatim port (for future solve use; not wired).
- **Spec refinement:** the spec said `backends/coal.py` is not ported; in fact it IS ported (Task 3) because `build_object_field` and `combined.py` import `build_bvh`/`surface_field` from it. What is NOT exercised is the exact per-frame Coal query mode (the `object_sdf is None` branch). The turnkey-without-coal property holds for the *bundled-artifact load path* (Tasks 2/8/9 use only numpy loaders), not for importing `combined.py`.
- **Reuse:** `HumanBody`/`PointCloudCache` come from the 4a `src/correspondence/` package; the contact cache is built from the bundled correspondence's `tri_idx`/`bary` (Task 7) so contact is keyed 1:1 to the OT map.
- **Native + v1 untouched:** only `gmr_socp_v2.py` changes (adds `object_sdf`/`contact_fields`; solver untouched), plus packaging + new files.
- **Open items for the implementer:** exact `test_pipe_retargeting/constants.py` values for the contact constants (Task 1 Step 3); `joblib` install if missing (Task 3); the bundled artifact sizes must stay < 100 MB (Tasks 6/7 — raise SDF resolution / check contact size); the viz point set choice when `witness` is sparse (Task 9 Step 1).
```
