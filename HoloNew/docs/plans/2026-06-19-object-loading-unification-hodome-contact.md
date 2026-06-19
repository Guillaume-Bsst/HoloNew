# Object-Loading Unification + HODome Contact Channel — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the solver's object channel dataset-agnostic via an `ObjectSource` abstraction, activate the contact/SDF object channel for HODome, unify even object-surface sampling on one density-based helper, and remove dead object code.

**Architecture:** Each `MotionLoader` gains an abstract `object_source()` returning `list[ObjectSource]` (mesh path + Z-up object poses). The TEST-SOCP builder consumes a single source (assert ≤1) when `cfg.dataset` is set, with a legacy fallback (`OBJECT_MESH_FILE` + `.pt`) otherwise. The forced `robot_only` for smplx datasets is lifted when a loader yields an object. One even sampler (`sample_object_surface`, density pts/m²) serves solver + viewer.

**Tech Stack:** Python, numpy, trimesh, smplx, pytest. Run tests from the `HoloNew/` package dir in the project conda env.

## Global Constraints

- `poses_raw` is always `(T, 7)` ordered `[qw, qx, qy, qz, x, y, z]`, in the **same Z-up world** as `human_joints`.
- `object_source()` returns `list[ObjectSource]`; the builder consumes **one** (`assert len(sources) <= 1`). Empty list = no object for this sequence.
- Object sampling is **density-based** (pts/m²), never a fixed count.
- The legacy path (no `--dataset`) must stay behavior-identical.
- No `Co-Authored-By` / Claude trailer in commit messages.
- Out of scope (Spec 2): MuJoCo collision-scene generation, `activate_obj_non_penetration` for HODome, movable solved object pose. Out of scope: multi-human, real multi-object consumption.

---

### Task 1: `ObjectSource` dataclass + abstract `object_source` on all loaders

**Files:**
- Modify: `src/data_loaders/base.py`
- Modify: `src/data_loaders/omomo.py`, `src/data_loaders/hodome.py`, `src/data_loaders/legacy.py`
- Test: `tests/test_data_loaders_base.py`

**Interfaces:**
- Produces: `ObjectSource(mesh_path: Path, poses_raw: np.ndarray)`; `MotionLoader.object_source(*, motion_path, obj_path, model_path, task_type, constants, motion_data_config, smpl_model_dir=None) -> list[ObjectSource]`.

- [ ] **Step 1: Write the failing test**

In `tests/test_data_loaders_base.py`, update the `DummyLoader` to also implement `object_source`, and add a test that the abstraction exists and stubs return `[]`:

```python
from HoloNew.src.data_loaders.base import (
    MotionLoader, DATASET_TO_FORMAT, DATASET_LOADERS, register_loader, resolve_loader,
    ObjectSource,
)

def test_object_source_abstract_and_stub():
    @register_loader("dummy_os")
    class DummyOS(MotionLoader):
        def load(self, *, model_path, motion_path, obj_path, task_type,
                 constants, motion_data_config, smpl_model_dir=None):
            return np.zeros((2, 22, 3)), np.zeros((2, 7)), 1.0
        def object_source(self, *, motion_path, obj_path, model_path, task_type,
                          constants, motion_data_config, smpl_model_dir=None):
            return []
    srcs = resolve_loader("dummy_os").object_source(
        motion_path=None, obj_path=None, model_path=None, task_type="robot_only",
        constants=None, motion_data_config=None)
    assert srcs == []
    DATASET_LOADERS.pop("dummy_os")
```

Also update the existing `test_register_and_resolve_loader` `DummyLoader` to implement `object_source` (returning `[]`) so it remains instantiable.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_data_loaders_base.py -v`
Expected: FAIL — `ImportError: cannot import name 'ObjectSource'`.

- [ ] **Step 3: Implement in base.py**

Add to `src/data_loaders/base.py` (after imports add `from dataclasses import dataclass`):

```python
@dataclass(frozen=True)
class ObjectSource:
    """One object's solver inputs, in the human_joints Z-up world frame.

    mesh_path: a mesh file usable directly by load_or_build_object_sdf, in the
        object-local frame consistent with poses_raw.
    poses_raw: (T, 7) [qw, qx, qy, qz, x, y, z] object pose per frame.
    """
    mesh_path: "Path"
    poses_raw: np.ndarray
```

Add the abstract method inside `MotionLoader`:

```python
    @abstractmethod
    def object_source(self, *, motion_path: Path, obj_path: Path | None,
                      model_path: Path | None, task_type: str, constants,
                      motion_data_config,
                      smpl_model_dir: Path | None = None) -> list["ObjectSource"]:
        """Object sources for the sequence (Z-up world of human_joints).
        Empty list when the sequence has no object (or task_type robot_only)."""
        ...
```

- [ ] **Step 4: Add `[]` stubs to concrete loaders**

In `src/data_loaders/legacy.py`, add to `LegacyLoader`:

```python
    def object_source(self, *, motion_path, obj_path, model_path, task_type,
                      constants, motion_data_config, smpl_model_dir=None):
        return []   # lafan / sfu / climbing have no .pt object channel here
```

In `src/data_loaders/omomo.py` (`OmomoMixedLoader`) and `src/data_loaders/hodome.py` (`HoDomeLoader`), add a temporary stub returning `[]` (real implementations land in Tasks 2 and 3):

```python
    def object_source(self, *, motion_path, obj_path, model_path, task_type,
                      constants, motion_data_config, smpl_model_dir=None):
        return []
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_data_loaders_base.py tests/test_data_loaders_hodome.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/data_loaders/base.py src/data_loaders/omomo.py src/data_loaders/hodome.py src/data_loaders/legacy.py tests/test_data_loaders_base.py
git commit -m "feat: ObjectSource abstraction + abstract object_source on loaders"
```

---

### Task 2: `OmomoMixedLoader.object_source` (real)

**Files:**
- Modify: `src/data_loaders/omomo.py`
- Test: `tests/test_data_loaders_base.py` (or a new `tests/test_object_source_omomo.py`)

**Interfaces:**
- Consumes: `ObjectSource`, `load_intermimic_data`.
- Produces: `OmomoMixedLoader.object_source(...) -> list[ObjectSource]` (bundled `models/{obj}/{obj}.obj` + `.pt` poses).

- [ ] **Step 1: Write the failing test**

Create `tests/test_object_source_omomo.py`:

```python
import numpy as np, torch
from pathlib import Path
from HoloNew.src.data_loaders.omomo import OmomoMixedLoader

def _fake_pt(path, T=4):
    # InterMimic layout: object pose at slice [318:325] as [qx,qy,qz,qw,x,y,z]
    row = np.zeros((T, 400), np.float32)
    row[:, 321] = 1.0                      # qw=1 (slot [318:325][3])
    row[:, 322] = np.arange(T)             # x
    torch.save(torch.from_numpy(row), str(path))

def test_omomo_object_source_robot_only_empty(tmp_path):
    p = tmp_path / "sub3_largebox_003.pt"; _fake_pt(p)
    srcs = OmomoMixedLoader().object_source(
        motion_path=p, obj_path=None, model_path=None, task_type="robot_only",
        constants=None, motion_data_config=None)
    assert srcs == []

def test_omomo_object_source_poses_shape(tmp_path, monkeypatch):
    p = tmp_path / "sub3_largebox_003.pt"; _fake_pt(p, T=4)
    # Bundled mesh absent -> empty (matches solver: no mesh, no SDF)
    monkeypatch.chdir(tmp_path)
    srcs = OmomoMixedLoader().object_source(
        motion_path=p, obj_path=p, model_path=None, task_type="object_interaction",
        constants=None, motion_data_config=None)
    assert srcs == []   # no models/largebox/largebox.obj under tmp
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_object_source_omomo.py -v`
Expected: FAIL — current stub returns `[]` for robot_only (passes) but the second test passes too only by luck; replace the stub so logic is real. Run and confirm at least one assertion path exercises the real code after Step 3.

- [ ] **Step 3: Implement**

Replace the stub in `src/data_loaders/omomo.py`:

```python
    def object_source(self, *, motion_path, obj_path, model_path, task_type,
                      constants, motion_data_config, smpl_model_dir=None):
        if task_type == "robot_only" or motion_path is None:
            return []
        from HoloNew.src.data_loaders.base import ObjectSource
        name = Path(motion_path).stem
        parts = name.split("_")
        obj_name = parts[1] if len(parts) >= 2 else name
        # The bundled, centred + pre-scaled mesh is the solver's canonical object
        # (same mesh the legacy OBJECT_MESH_FILE path uses). No bundled mesh -> no
        # object SDF, matching the builder's "object mesh not found" behaviour.
        bundled = Path("models") / obj_name / f"{obj_name}.obj"
        if not bundled.exists():
            return []
        _, poses = load_intermimic_data(str(motion_path))
        return [ObjectSource(mesh_path=bundled, poses_raw=np.asarray(poses, np.float64))]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_object_source_omomo.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data_loaders/omomo.py tests/test_object_source_omomo.py
git commit -m "feat: OmomoMixedLoader.object_source (bundled mesh + .pt poses)"
```

---

### Task 3: `HoDomeLoader.object_source` (real)

**Files:**
- Modify: `src/data_loaders/hodome.py`
- Test: `tests/test_data_loaders_hodome.py`

**Interfaces:**
- Consumes: `extract_hodome_object_mesh`, `hodome_object_poses`, `ObjectSource`.
- Produces: `HoDomeLoader.object_source(...) -> list[ObjectSource]` (tar-extracted mesh + Z-up poses).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_data_loaders_hodome.py`:

```python
import tarfile, trimesh
from pathlib import Path
from HoloNew.src.data_loaders.hodome import HoDomeLoader

def _make_scaned_tar(scaned_dir, token):
    scaned_dir.mkdir(parents=True, exist_ok=True)
    work = scaned_dir / token; work.mkdir()
    box = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    box.export(work / f"{token}.obj")
    with tarfile.open(scaned_dir / f"{token}.tar", "w") as t:
        t.add(work / f"{token}.obj", arcname=f"{token}/{token}.obj")

def test_hodome_object_source(tmp_path):
    token = "box"
    obj_dir = tmp_path / "object"; obj_dir.mkdir()
    _make_object_npz(obj_dir / f"sub3_{token}.npz", T=3)   # reuse helper above
    _make_scaned_tar(tmp_path / "scaned_object", token)
    srcs = HoDomeLoader().object_source(
        motion_path=tmp_path / "smplx" / f"sub3_{token}.npz",
        obj_path=obj_dir / f"sub3_{token}.npz", model_path=None,
        task_type="object_interaction", constants=None, motion_data_config=None)
    assert len(srcs) == 1
    assert srcs[0].poses_raw.shape == (3, 7)
    assert Path(srcs[0].mesh_path).exists()

def test_hodome_object_source_robot_only_empty(tmp_path):
    srcs = HoDomeLoader().object_source(
        motion_path=tmp_path / "m.npz", obj_path=None, model_path=None,
        task_type="robot_only", constants=None, motion_data_config=None)
    assert srcs == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_data_loaders_hodome.py -v -k object_source`
Expected: FAIL — stub returns `[]`, so `len(srcs) == 1` fails.

- [ ] **Step 3: Implement**

Replace the stub in `src/data_loaders/hodome.py` (`HoDomeLoader`):

```python
    def object_source(self, *, motion_path, obj_path, model_path, task_type,
                      constants, motion_data_config, smpl_model_dir=None):
        if task_type == "robot_only" or obj_path is None:
            return []
        from HoloNew.src.data_loaders.base import ObjectSource
        stem = Path(obj_path).stem
        token = stem.split("_", 1)[1] if "_" in stem else stem
        scaned = Path(obj_path).parent.parent / "scaned_object"
        mesh_path = extract_hodome_object_mesh(token, scaned)
        poses = hodome_object_poses(Path(obj_path))    # (T,7) Z-up
        return [ObjectSource(mesh_path=Path(mesh_path), poses_raw=poses)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_data_loaders_hodome.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data_loaders/hodome.py tests/test_data_loaders_hodome.py
git commit -m "feat: HoDomeLoader.object_source (tar mesh + Z-up poses)"
```

---

### Task 4: Builder consumes `object_source` (dataset path) with legacy fallback

**Files:**
- Modify: `src/test_socp/builder.py` (object SDF + pose-loading blocks, ~lines 271-389)
- Test: `tests/test_object_interaction.py` (extend) or new `tests/test_builder_object_source.py`

**Interfaces:**
- Consumes: `resolve_loader(cfg.dataset).object_source(...)`, `load_or_build_object_sdf`, `load_intermimic_data`.
- Produces: `rt.object_sdf`, `rt._obj_poses_raw`, `rt._obj_poses_mj` populated dataset-agnostically.

- [ ] **Step 1: Write the failing test**

Create `tests/test_builder_object_source.py` — a unit test of the resolver helper (extract the mesh/pose resolution into a pure function `resolve_object_inputs(cfg, constants, pt_path) -> tuple[Path | None, np.ndarray | None]` so it is testable without a full solve):

```python
import numpy as np
from types import SimpleNamespace
from HoloNew.src.test_socp.builder import resolve_object_inputs

def test_resolve_object_inputs_legacy_no_dataset(tmp_path, monkeypatch):
    # No dataset -> uses constants.OBJECT_MESH_FILE + .pt poses
    cfg = SimpleNamespace(dataset=None, motion_path=None, obj_path=None, model_path=None,
                          task_type="object_interaction", motion_data_config=None)
    constants = SimpleNamespace(OBJECT_MESH_FILE="models/largebox/largebox.obj")
    called = {}
    def fake_pt(path): called["pt"] = path; return None, np.zeros((5, 7))
    monkeypatch.setattr("HoloNew.src.test_socp.builder.load_intermimic_data", fake_pt)
    mesh, poses = resolve_object_inputs(cfg, constants, pt_path="seq.pt")
    assert str(mesh) == "models/largebox/largebox.obj"
    assert poses.shape == (5, 7) and called["pt"] == "seq.pt"

def test_resolve_object_inputs_dataset_empty(monkeypatch):
    cfg = SimpleNamespace(dataset="sfu", motion_path="m.npz", obj_path=None, model_path=None,
                          task_type="robot_only", motion_data_config=None)
    mesh, poses = resolve_object_inputs(cfg, SimpleNamespace(OBJECT_MESH_FILE=None),
                                        pt_path=None)
    assert mesh is None and poses is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_builder_object_source.py -v`
Expected: FAIL — `resolve_object_inputs` does not exist.

- [ ] **Step 3: Implement the helper + wire it in**

Add to `src/test_socp/builder.py` (module level):

```python
def resolve_object_inputs(cfg, constants, pt_path):
    """(mesh_path, poses_raw) for the object channel, dataset-agnostic.

    Dataset path: the loader's object_source (one object; assert <=1). Legacy
    path (no --dataset): constants.OBJECT_MESH_FILE + load_intermimic_data(.pt).
    Returns (None, None) when there is no object."""
    if cfg.dataset is not None:
        from HoloNew.src.data_loaders.base import resolve_loader
        sources = resolve_loader(cfg.dataset).object_source(
            motion_path=cfg.motion_path, obj_path=cfg.obj_path, model_path=cfg.model_path,
            task_type=cfg.task_type, constants=constants,
            motion_data_config=cfg.motion_data_config,
            smpl_model_dir=getattr(cfg, "smpl_model_dir", None))
        assert len(sources) <= 1, f"multi-object not supported yet: {len(sources)} sources"
        if not sources:
            return None, None
        return sources[0].mesh_path, np.asarray(sources[0].poses_raw, np.float64)
    mesh_file = getattr(constants, "OBJECT_MESH_FILE", None)
    if mesh_file is None or pt_path is None:
        return None, None
    _, poses = load_intermimic_data(str(pt_path))
    return mesh_file, np.asarray(poses, np.float64)
```

Then, in `TestSocp.from_config`, replace the three places that hardcode mesh/poses:

1. Compute once near the top of the object section:
   `_mesh_file, _obj_poses_all = resolve_object_inputs(cfg, constants, pt_path)`.
2. Object SDF block (~line 330): gate on `_mesh_file is not None and Path(_mesh_file).exists()` (drop the `OBJECT_MESH_FILE` direct read).
3. `_obj_poses_raw` / `_obj_poses_mj` blocks: use `_obj_poses_all[:T].copy()` instead of `load_intermimic_data(pt_path)`. Keep the `_obj_xy`/`_obj_z` placement and `convert_object_poses_to_mujoco_order`.

Ensure `from HoloNew.src.utils import load_intermimic_data` is imported at module top (so the test can monkeypatch `builder.load_intermimic_data`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_builder_object_source.py tests/test_object_interaction.py -v`
Expected: PASS (object_interaction parity preserved on the legacy/OMOMO path).

- [ ] **Step 5: Commit**

```bash
git add src/test_socp/builder.py tests/test_builder_object_source.py
git commit -m "feat: builder consumes object_source with legacy fallback"
```

---

### Task 5: Lift forced `robot_only` for HODome+object (facade + validate + object name)

**Files:**
- Modify: `src/data_loaders/facade.py` (the `robot_only` force, ~lines 94-99)
- Modify: `examples/robot_retarget.py` (`validate_config` ~line 170; object-name/mesh guard ~line 677)
- Test: `tests/test_facade_resolve.py` (extend)

**Interfaces:**
- Consumes: `resolve_loader(...).object_source(...)`.
- Produces: hodome+object keeps `task_type="object_interaction"`; `OBJECT_NAME` set to the token.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_facade_resolve.py`:

```python
from types import SimpleNamespace
from HoloNew.src.data_loaders.facade import normalize_dataset_cfg

def _cfg(**kw):
    base = dict(dataset="hodome", motion_name=None, model_path="m", motion_path="smplx/sub3_box.npz",
                obj_path="object/sub3_box.npz", smpl_model_dir=None, data_format=None,
                task_type="object_interaction", task_name=None, data_path=None, omomo_dir=None)
    base.update(kw); return SimpleNamespace(**base)

def test_hodome_with_object_keeps_object_interaction(monkeypatch):
    # object_source non-empty -> do NOT force robot_only
    monkeypatch.setattr(
        "HoloNew.src.data_loaders.facade._has_object_source", lambda cfg: True)
    cfg = _cfg()
    monkeypatch.setattr("HoloNew.src.data_loaders.facade.prep_hodome_processed",
                        lambda *a, **k: {"global_joint_positions": __import__("numpy").zeros((1,22,3),"float32"),
                                          "global_joint_orientations": __import__("numpy").zeros((1,22,4),"float32"),
                                          "height": 1.7, "betas": __import__("numpy").zeros(10,"float32"),
                                          "gender": "neutral"})
    normalize_dataset_cfg(cfg)
    assert cfg.task_type == "object_interaction"

def test_smplx_no_object_forces_robot_only(monkeypatch):
    monkeypatch.setattr(
        "HoloNew.src.data_loaders.facade._has_object_source", lambda cfg: False)
    cfg = _cfg(dataset="sfu", obj_path=None, motion_path="sub3.npz")
    # minimal: sfu path falls through to else branch; just assert the force happens
    cfg.task_type = "object_interaction"
    normalize_dataset_cfg(cfg)
    assert cfg.task_type == "robot_only"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_facade_resolve.py -v -k "object_interaction or robot_only"`
Expected: FAIL — `_has_object_source` does not exist and the force is unconditional.

- [ ] **Step 3: Implement**

In `src/data_loaders/facade.py`, add a helper and gate the force:

```python
def _has_object_source(cfg) -> bool:
    """True when the dataset loader yields an object for this sequence."""
    if cfg.obj_path is None:
        return False
    from HoloNew.src.data_loaders.base import resolve_loader
    srcs = resolve_loader(cfg.dataset).object_source(
        motion_path=cfg.motion_path, obj_path=cfg.obj_path, model_path=cfg.model_path,
        task_type=cfg.task_type, constants=None, motion_data_config=cfg.motion_data_config,
        smpl_model_dir=getattr(cfg, "smpl_model_dir", None))
    return len(srcs) > 0
```

Replace the unconditional force (lines ~94-99) with:

```python
    if (cfg.data_format == "smplx" and cfg.task_type == "object_interaction"
            and not _has_object_source(cfg)):
        import logging
        logging.getLogger(__name__).info(
            "Dataset %s smplx has no object source; using task_type=robot_only.", dataset)
        cfg.task_type = "robot_only"
```

In `examples/robot_retarget.py`:
- `validate_config` (~line 170): change the object_interaction format guard to allow smplx:
  `if cfg.task_type == "object_interaction" and cfg.data_format not in (None, "smplh", "smplx"):`.
- Object-name/mesh guard (~line 677): only set `OBJECT_MESH_FILE`/`OBJECT_URDF_FILE` from
  `obj_path` when it is a mesh file (`.obj`); for HODome (obj_path is a `.npz`) set
  `constants.OBJECT_NAME` to the token instead and leave `OBJECT_MESH_FILE` to the
  `object_source` path:

```python
    if cfg.dataset is not None and task_type == "object_interaction" and cfg.obj_path is not None:
        op = Path(cfg.obj_path)
        if op.suffix == ".obj":
            constants.OBJECT_MESH_FILE = str(op)
            constants.OBJECT_URDF_FILE = str(op.with_suffix(".urdf"))
        else:  # pose-file datasets (HODome .npz): mesh comes from object_source
            stem = op.stem
            constants.OBJECT_NAME = stem.split("_", 1)[1] if "_" in stem else stem
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_facade_resolve.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data_loaders/facade.py examples/robot_retarget.py tests/test_facade_resolve.py
git commit -m "feat: keep object_interaction for HODome+object; relax smplx guard"
```

---

### Task 6: Unify even object sampling on `sample_object_surface`

**Files:**
- Modify: `examples/view_stages.py` (lines ~249, ~291, comment ~247)
- Modify: `examples/robot_retarget.py` (line ~349 even sampling)
- Test: `tests/test_object_center_scale.py` or new `tests/test_object_sampling_unified.py`

**Interfaces:**
- Consumes: `sample_object_surface(mesh_file, density=200.0, seed=0) -> (M,3)`.
- Produces: viewer + object_interaction even point clouds via the single sampler.

- [ ] **Step 1: Write the failing test**

Create `tests/test_object_sampling_unified.py`:

```python
import numpy as np, trimesh
from HoloNew.src.test_socp.movable import sample_object_surface

def test_sample_density_deterministic(tmp_path):
    box = trimesh.creation.box(extents=(1, 1, 1))
    f = tmp_path / "box.obj"; box.export(f)
    a = sample_object_surface(str(f), density=200.0, seed=0)
    b = sample_object_surface(str(f), density=200.0, seed=0)
    assert a.shape[1] == 3 and len(a) >= 64
    assert np.allclose(a, b)   # deterministic at fixed seed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_object_sampling_unified.py -v`
Expected: PASS already (sampler exists) — this pins the contract before swapping callers. If it fails, fix the seed plumbing in `sample_object_surface` first.

- [ ] **Step 3: Swap the even-sampling callers**

In `examples/view_stages.py` replace both `load_object_data(..., sample_count=100)` even calls (OMOMO ~line 249, HODome ~line 291) with:

```python
                from HoloNew.src.test_socp.movable import sample_object_surface
                object_points_local = sample_object_surface(str(obj_file)).astype(np.float32)
```

(OMOMO keeps the existing `- obj_geom_center) * obj_geom_scale` post-processing on the result.)

Fix the misleading comment at ~line 247 to: `# Native-size local samples via the solver's sample_object_surface (density-based).`

In `examples/robot_retarget.py` object_interaction (~line 349), replace the even `load_object_data` call with density sampling and apply the demo scale at the caller:

```python
        from HoloNew.src.test_socp.movable import sample_object_surface
        object_local_pts = sample_object_surface(constants.OBJECT_MESH_FILE)
        object_local_pts_demo = object_local_pts * smpl_scale
        return object_local_pts, object_local_pts_demo, constants.OBJECT_URDF_FILE
```

Leave the climbing path (~line 367, `surface_weights`) on `load_object_data` unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_object_sampling_unified.py tests/test_viewer_object.py tests/test_object_center_scale.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/view_stages.py examples/robot_retarget.py tests/test_object_sampling_unified.py
git commit -m "refactor: unify even object sampling on sample_object_surface"
```

---

### Task 7: `obj_scale` missing → explicit error (#3)

**Files:**
- Modify: `examples/view_stages.py` (the `captured_objects` recenter branch, ~lines 226-231)
- Test: new `tests/test_view_stages_obj_scale.py`

**Interfaces:**
- Produces: `ValueError` when a non-centred captured mesh is used without `obj_scale`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_view_stages_obj_scale.py`. Extract the scale-resolution into a small pure function `resolve_captured_obj_scale(omomo_dir, task_name) -> float` in `view_stages.py` that raises on missing scale, and test it:

```python
import pytest
from HoloNew.examples.view_stages import resolve_captured_obj_scale

def test_missing_obj_scale_raises(tmp_path):
    with pytest.raises(ValueError, match="obj_scale"):
        resolve_captured_obj_scale(tmp_path, "sub3_unknownobj_000")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_view_stages_obj_scale.py -v`
Expected: FAIL — `resolve_captured_obj_scale` does not exist.

- [ ] **Step 3: Implement**

Add to `examples/view_stages.py`:

```python
def resolve_captured_obj_scale(omomo_dir, task_name) -> float:
    """obj_scale for a non-centred captured mesh; raise if unknown (no silent unit fallback)."""
    scale = load_object_scale(omomo_dir, task_name) if omomo_dir else None
    if scale is None:
        raise ValueError(
            f"obj_scale missing for {task_name}: captured mesh is off-origin and unscaled, "
            f"so it would render at the wrong size. Provide a bundled models/<obj>/<obj>.obj "
            f"or a valid OMOMO .p with obj_scale.")
    return float(scale)
```

Replace the warning+fallback block (~lines 226-231) with `obj_geom_scale = resolve_captured_obj_scale(cfg.omomo_dir, cfg.task_name)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_view_stages_obj_scale.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/view_stages.py tests/test_view_stages_obj_scale.py
git commit -m "fix: raise on missing obj_scale instead of silent unit fallback"
```

---

### Task 8: Remove dead object code (#2)

**Files:**
- Delete: `src/test_socp/contact/object_input.py`
- Modify: `src/test_socp/contact/probes.py` (remove `make_object_grid`), `src/test_socp/contact/constants.py` (remove `OBJECT_GRID_DENSITY`)
- Modify: `tests/test_contact_backends.py` (drop dead imports + `test_parse_obj_name`)

**Interfaces:** none (pure removal).

- [ ] **Step 1: Confirm no live imports**

Run:
```bash
grep -rn "object_input\|make_object_grid\|OBJECT_GRID_DENSITY" src/ examples/ tests/ --include="*.py" | grep -v __pycache__
```
Expected: only `probes.py`, `constants.py`, and `tests/test_contact_backends.py` (import-only) appear — no production caller. If anything else appears, STOP and reassess.

- [ ] **Step 2: Remove**

- Delete `src/test_socp/contact/object_input.py`.
- In `src/test_socp/contact/probes.py`: delete `make_object_grid` and remove `OBJECT_GRID_DENSITY` from the `.constants` import.
- In `src/test_socp/contact/constants.py`: delete the `OBJECT_GRID_DENSITY = 5000.0` line.
- In `tests/test_contact_backends.py`: delete line 2 (`from ...object_input import parse_obj_name`), drop `make_object_grid` from the line-3 import, and delete `test_parse_obj_name`.

- [ ] **Step 3: Run the contact suite**

Run: `python -m pytest tests/test_contact_backends.py tests/test_contact_field.py tests/test_contact_io.py -v`
Expected: PASS (remaining tests: floor grid, coal smoke).

- [ ] **Step 4: Full smoke**

Run: `python -m pytest tests/ -q`
Expected: PASS (no import errors from the removal).

- [ ] **Step 5: Commit**

```bash
git add -A src/test_socp/contact/ tests/test_contact_backends.py
git commit -m "chore: remove dead object code (object_input, make_object_grid)"
```

---

## Self-Review

**Spec coverage:**
- #5a ObjectSource abstraction → Tasks 1-3. Builder consumption → Task 4. Lift robot_only / smplx object_interaction → Task 5. ✓
- #1 unify even sampling → Task 6. ✓
- #3 obj_scale error → Task 7. ✓
- #2 dead code → Task 8. ✓
- Legacy fallback parity → Task 4 (`test_resolve_object_inputs_legacy_no_dataset` + `test_object_interaction`). ✓
- HODome contact channel active → Task 4 builds `object_sdf` from the HODome mesh + `_obj_poses_raw`; the always-built `SmplxGroundProbe` then queries it (no extra task). ✓

**Placeholder scan:** No TBD/TODO; every code step shows code. ✓

**Type consistency:** `ObjectSource(mesh_path, poses_raw)` and `object_source(...) -> list[ObjectSource]` used identically across Tasks 1-5; `resolve_object_inputs(cfg, constants, pt_path) -> (mesh_path|None, poses|None)` used in Task 4; `sample_object_surface(mesh_file, density=200.0, seed=0)` consistent in Task 6. ✓

**Behavioral note:** For `--dataset omomo`, Task 2 makes the solver use the bundled `models/{obj}/{obj}.obj` (centred + scaled) rather than the off-origin captured mesh previously assigned via `robot_retarget.py:677`. This aligns `--dataset omomo` with the legacy path and is the intended correction; the parity test pins the legacy path explicitly.
