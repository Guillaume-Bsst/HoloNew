# HODome MuJoCo Collision Scene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate the robot+object MuJoCo scene xml for HODome objects on the fly (mirroring `g1_29dof_w_largebox.xml`) and wire `object_name`/scene so the existing scene swap activates object non-penetration + movable object for HODome, with no solver change.

**Architecture:** A generator copies the base robot MJCF and injects an object `<body>` (freejoint + convex-hull `type="mesh"` geom, fixed largebox-style inertia) referencing the tar-extracted `.obj` by absolute path, written next to the robot meshes at `models/<robot>/<stem>_w_<token>.xml` so the historical `.replace(".urdf","_w_<token>.xml")` swap finds it. The façade sets `task_config.object_name=token` so `create_task_constants` propagates `OBJECT_NAME=token`; `build_from_config` generates the scene before the model is built.

**Tech Stack:** Python, MuJoCo (mujoco.MjModel), trimesh (tests only), pytest. Run from `HoloNew/HoloNew/` with `/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python`.

## Global Constraints

- Convex hull only: object geom is a single `<geom type="mesh">` (MuJoCo convex hull), exactly like `largebox.xml`. No convex decomposition (coacd/vhacd absent from the env).
- Fixed object inertia/mass copied verbatim from largebox: `mass="0.1"`, `diaginertia="0.002 0.002 0.002"`. No mesh-derived inertia. Add nothing not present in the base largebox pattern.
- Object mesh referenced by **absolute path** in the asset (robot meshes resolve via the base MJCF `meshdir="assets/"` relative to `models/<robot>/`, where the scene xml is written).
- Solver unchanged: scene swap, `_obj_poses_mj` drive, non-penetration, movable are reused as-is.
- No `Co-Authored-By` / Claude trailer in commit messages.
- Out of scope: convex decomposition, mesh-derived inertia, multi-object/multi-human.

---

### Task 1: `build_hodome_scene_xml` generator

**Files:**
- Create: `src/data_loaders/hodome_scene.py`
- Test: `tests/test_hodome_scene.py`

**Interfaces:**
- Produces: `build_hodome_scene_xml(robot_xml_path: str|Path, token: str, mesh_obj_path: str|Path, output_path: str|Path|None = None) -> Path`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hodome_scene.py`:

```python
from pathlib import Path
import trimesh
import mujoco
from HoloNew.src.data_loaders.hodome_scene import build_hodome_scene_xml

_G1 = Path("models/g1/g1_29dof.xml")


def test_scene_content(tmp_path):
    mesh = tmp_path / "box.obj"
    trimesh.creation.box(extents=(0.2, 0.2, 0.2)).export(mesh)
    out = tmp_path / "scene.xml"
    p = build_hodome_scene_xml(_G1, "baseball", mesh, output_path=out)
    txt = Path(p).read_text()
    assert '<mesh name="baseball_mesh"' in txt
    assert str(mesh.resolve()) in txt              # absolute object-mesh path
    assert '<body name="baseball_link">' in txt
    assert "<freejoint/>" in txt
    assert 'diaginertia="0.002 0.002 0.002"' in txt


def test_scene_parses_and_adds_free_joint(tmp_path):
    # Written next to the robot meshes so meshdir="assets/" resolves; cleaned up after.
    mesh = tmp_path / "box.obj"
    trimesh.creation.box(extents=(0.3, 0.3, 0.3)).export(mesh)
    base_nq = mujoco.MjModel.from_xml_path(str(_G1)).nq
    out = _G1.with_name("g1_29dof_w_pytesttoken.xml")
    try:
        build_hodome_scene_xml(_G1, "pytesttoken", mesh, output_path=out)
        m = mujoco.MjModel.from_xml_path(str(out))
        assert m.nq == base_nq + 7                 # object free joint adds 7 qpos
    finally:
        out.unlink(missing_ok=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_hodome_scene.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'HoloNew.src.data_loaders.hodome_scene'`.

- [ ] **Step 3: Implement the generator**

Create `src/data_loaders/hodome_scene.py`:

```python
"""Generate a robot+object MuJoCo scene xml for an arbitrary (HODome) object,
mirroring the bundled models/g1/g1_29dof_w_largebox.xml: a single convex-hull mesh
geom + free joint, fixed largebox-style inertia. Written next to the robot meshes so
the base MJCF meshdir resolves; the object mesh is referenced by absolute path."""
from __future__ import annotations

from pathlib import Path

# Object body block, copied verbatim from g1_29dof_w_largebox.xml (convex-hull mesh
# geom, fixed mass/inertia). {token} fills the object name.
_OBJECT_BLOCK = """    <body name="{token}_link">
        <freejoint/>
        <inertial pos="0 0 0" mass="0.1" diaginertia="0.002 0.002 0.002"/>
        <geom name="{token}" type="mesh" mesh="{token}_mesh"
                contype="1" conaffinity="1"
                pos="0 0 0" quat="1 0 0 0"
                rgba="0.7 0.8 0.9 0.7"
                friction="0.9 0.5 0.5"
                solref="0.02 1"
                solimp="0.9 0.95 0.001"/>
    </body>
"""


def build_hodome_scene_xml(robot_xml_path, token, mesh_obj_path, output_path=None) -> Path:
    """Robot+object scene xml for `token`, mesh = `mesh_obj_path` (convex hull).

    Injects the object mesh asset (absolute path) before the first </asset> and the
    object body before </worldbody>. Default output is next to the robot xml as
    <robot_stem>_w_<token>.xml (the path the solver scene swap expects)."""
    robot_xml_path = Path(robot_xml_path)
    mesh_abs = str(Path(mesh_obj_path).resolve())
    content = robot_xml_path.read_text()

    asset = f'    <mesh name="{token}_mesh" file="{mesh_abs}" scale="1 1 1"/>\n'
    i = content.index("</asset>")               # first </asset> = the mesh-asset block
    content = content[:i] + asset + content[i:]

    j = content.index("</worldbody>")
    content = content[:j] + _OBJECT_BLOCK.format(token=token) + content[j:]

    if output_path is None:
        output_path = robot_xml_path.with_name(f"{robot_xml_path.stem}_w_{token}.xml")
    output_path = Path(output_path)
    output_path.write_text(content)
    return output_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_hodome_scene.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/data_loaders/hodome_scene.py tests/test_hodome_scene.py
git commit -m "feat: build_hodome_scene_xml (robot+object MuJoCo scene generator)"
```

---

### Task 2: Façade sets `object_name=token` for HODome

**Files:**
- Modify: `src/data_loaders/facade.py` (the hodome branch of `normalize_dataset_cfg`, ~lines 109-117)
- Test: `tests/test_facade_resolve.py`

**Interfaces:**
- Consumes: `dataclasses.replace`, the resolved `cfg.task_name` token.
- Produces: after `normalize_dataset_cfg`, `cfg.task_config.object_name == token` for HODome with an object; unchanged otherwise.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_facade_resolve.py`:

```python
def test_hodome_sets_object_name_token(monkeypatch):
    import numpy as np
    monkeypatch.setattr(facade, "_has_object_source", lambda cfg: True)
    monkeypatch.setattr(facade, "prep_hodome_processed", lambda *a, **k: {
        "global_joint_positions": np.zeros((1, 22, 3), "float32"),
        "global_joint_orientations": np.zeros((1, 22, 4), "float32"),
        "height": np.float32(1.7), "betas": np.zeros(10, "float32"), "gender": "neutral"})
    cfg = _dataset_cfg(motion_path="smplx/subject01_baseball.npz",
                       obj_path="object/subject01_baseball.npz")
    facade.normalize_dataset_cfg(cfg)
    assert cfg.task_config.object_name == "baseball"
```

(`_dataset_cfg` from Spec 1's test already builds a hodome SimpleNamespace cfg; ensure it
carries a real `task_config`. If it uses a SimpleNamespace stub for task_config, give it a
`TaskConfig()` instance so `dataclasses.replace` works — update `_dataset_cfg` to set
`task_config=TaskConfig()` importing `from HoloNew.config_types.task import TaskConfig`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_facade_resolve.py::test_hodome_sets_object_name_token -q`
Expected: FAIL — `object_name` is `None` (façade does not set it yet).

- [ ] **Step 3: Implement**

In `src/data_loaders/facade.py`, add the import at the top:

```python
from dataclasses import replace
```

In the `elif dataset == "hodome":` branch of `normalize_dataset_cfg`, after `cfg.task_name = stem`, set the object name from the token when an object is present:

```python
        cfg.data_path = _HODOME_CACHE_DIR
        cfg.task_name = stem
        # Object name = token (2nd "_"-segment), so create_task_constants propagates
        # OBJECT_NAME=token to the builder gate AND the scene-swap path (_w_<token>.xml).
        if cfg.task_type == "object_interaction" and _has_object_source(cfg):
            token = stem.split("_", 1)[1] if "_" in stem else stem
            cfg.task_config = replace(cfg.task_config, object_name=token)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_facade_resolve.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data_loaders/facade.py tests/test_facade_resolve.py
git commit -m "feat: facade sets task_config.object_name=token for HODome objects"
```

---

### Task 3: `ensure_object_scene_xml` + wire into `build_from_config`

**Files:**
- Modify: `src/data_loaders/hodome_scene.py` (add `ensure_object_scene_xml`)
- Modify: `src/test_socp/builder.py` (call it after `create_task_constants`, before `cls(**kwargs)`)
- Test: `tests/test_hodome_scene.py`

**Interfaces:**
- Consumes: `build_hodome_scene_xml`, `resolve_loader(cfg.dataset).object_source(...)`.
- Produces: `ensure_object_scene_xml(cfg, constants) -> Path | None` — generates `<robot>_w_<OBJECT_NAME>.xml` for HODome object_interaction when the scene swap will fire; returns the path, or `None` (no-op).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_hodome_scene.py`:

```python
from types import SimpleNamespace
from HoloNew.src.data_loaders import hodome_scene
from HoloNew.src.data_loaders.base import ObjectSource
import numpy as np


def test_ensure_object_scene_xml(tmp_path, monkeypatch):
    mesh = tmp_path / "box.obj"
    trimesh.creation.box(extents=(0.2, 0.2, 0.2)).export(mesh)

    class _FakeLoader:
        def object_source(self, **kw):
            return [ObjectSource(mesh_path=mesh, poses_raw=np.zeros((2, 7)))]

    monkeypatch.setattr(hodome_scene, "resolve_loader", lambda d: _FakeLoader())
    cfg = SimpleNamespace(dataset="hodome", task_type="object_interaction",
                          motion_path="m.npz", obj_path="o.npz", model_path=None,
                          motion_data_config=None,
                          retargeter=SimpleNamespace(activate_obj_non_penetration=True,
                                                     load_object_scene=True))
    constants = SimpleNamespace(ROBOT_URDF_FILE="models/g1/g1_29dof.urdf",
                                OBJECT_NAME="baseball")
    out = _G1.with_name("g1_29dof_w_baseball.xml")
    try:
        p = hodome_scene.ensure_object_scene_xml(cfg, constants)
        assert p is not None and Path(p) == out and out.exists()
        assert '<body name="baseball_link">' in out.read_text()
    finally:
        out.unlink(missing_ok=True)


def test_ensure_object_scene_xml_noop_when_flag_off():
    cfg = SimpleNamespace(dataset="hodome", task_type="object_interaction",
                          motion_path="m.npz", obj_path="o.npz", model_path=None,
                          motion_data_config=None,
                          retargeter=SimpleNamespace(activate_obj_non_penetration=False,
                                                     load_object_scene=True))
    constants = SimpleNamespace(ROBOT_URDF_FILE="models/g1/g1_29dof.urdf",
                                OBJECT_NAME="baseball")
    assert hodome_scene.ensure_object_scene_xml(cfg, constants) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_hodome_scene.py -q`
Expected: FAIL — `ensure_object_scene_xml` / `resolve_loader` not defined in `hodome_scene`.

- [ ] **Step 3: Implement `ensure_object_scene_xml`**

In `src/data_loaders/hodome_scene.py`, add the import at the top:

```python
from HoloNew.src.data_loaders.base import resolve_loader
```

and the function:

```python
def ensure_object_scene_xml(cfg, constants):
    """Generate <robot>_w_<OBJECT_NAME>.xml for HODome object_interaction so the solver
    scene swap finds it. Gated on the same condition as the swap (object non-penetration
    + object scene). Returns the scene path, or None when not applicable."""
    if getattr(cfg, "dataset", None) != "hodome" or cfg.task_type != "object_interaction":
        return None
    sc = getattr(cfg, "retargeter", None)
    if not (getattr(sc, "activate_obj_non_penetration", False)
            and getattr(sc, "load_object_scene", True)):
        return None
    token = getattr(constants, "OBJECT_NAME", None)
    if token in (None, "ground"):
        return None
    srcs = resolve_loader(cfg.dataset).object_source(
        motion_path=cfg.motion_path, obj_path=cfg.obj_path, model_path=cfg.model_path,
        task_type=cfg.task_type, constants=constants,
        motion_data_config=cfg.motion_data_config,
        smpl_model_dir=getattr(cfg, "smpl_model_dir", None))
    if not srcs:
        return None
    robot_xml = str(constants.ROBOT_URDF_FILE).replace(".urdf", ".xml")
    return build_hodome_scene_xml(robot_xml, token, srcs[0].mesh_path)
```

- [ ] **Step 4: Wire into `build_from_config`**

In `src/test_socp/builder.py`, right after the `create_task_constants(...)` call (the
`constants = create_task_constants(...)` assignment, ~line 79-83) and before `rt = cls(**kwargs)`,
insert:

```python
    # HODome object_interaction: generate the robot+object MuJoCo scene xml before the
    # model is built, so __init__'s scene swap (_w_<OBJECT_NAME>.xml) finds it. No-op for
    # OMOMO (committed scene) and when object non-penetration is off.
    from HoloNew.src.data_loaders.hodome_scene import ensure_object_scene_xml
    ensure_object_scene_xml(cfg, constants)
```

(Place it after `constants` exists and before `kwargs`/`cls(**kwargs)`. `cfg` is the
`build_from_config` argument; `constants` is the local just built.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_hodome_scene.py tests/test_object_interaction.py -q`
Expected: PASS (scene tests green; OMOMO object_interaction unaffected — `ensure_` is a no-op for non-hodome).

- [ ] **Step 6: Commit**

```bash
git add src/data_loaders/hodome_scene.py src/test_socp/builder.py tests/test_hodome_scene.py
git commit -m "feat: generate HODome object scene before model build (ensure_object_scene_xml)"
```

---

### Task 4: End-to-end integration (skipif on HODome data)

**Files:**
- Test: `tests/test_hodome_object_scene_integration.py`

**Interfaces:**
- Consumes: `normalize_dataset_cfg`, `TestSocpRetargeter.from_config`, `RetargetingConfig`, `TestSocpRetargeterConfig`.

- [ ] **Step 1: Write the integration test**

Create `tests/test_hodome_object_scene_integration.py`:

```python
import numpy as np
import pytest
from pathlib import Path
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.src.data_loaders.facade import normalize_dataset_cfg

_NAME = "subject01_baseball"


def _hodome_seq_present() -> bool:
    try:
        from HoloNew.src.paths import get_path
        r = Path(get_path("hodome"))
        return (r / "smplx" / f"{_NAME}.npz").exists() \
            and (r / "object" / f"{_NAME}.npz").exists() \
            and (r / "scaned_object" / "baseball.tar").exists()
    except Exception:
        return False


@pytest.mark.skipif(not _hodome_seq_present(), reason="HODome baseball sequence not present")
def test_hodome_object_scene_end_to_end():
    cfg = RetargetingConfig(
        dataset="hodome", motion_name=_NAME, task_type="object_interaction",
        retargeter=TestSocpRetargeterConfig(activate_obj_non_penetration=True))
    normalize_dataset_cfg(cfg)                       # façade: paths, object_name=baseball
    assert cfg.task_config.object_name == "baseball"
    rt = TestSocpRetargeter.from_config(cfg)
    assert rt.object_name == "baseball"
    assert rt.has_dynamic_object is True             # scene swap added the object free joint
    res = rt.retarget(max_frames=3)
    assert np.all(np.isfinite(res.qpos))
    assert res.qpos.shape[1] >= 7 + rt.nq_a          # trailing object DOFs present
```

- [ ] **Step 2: Run the integration test**

Run: `/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_hodome_object_scene_integration.py -q`
Expected: PASS if the HODome `subject01_baseball` sequence is present; otherwise SKIPPED.
(Use a low `max_frames` — see the memory note on capping frames.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_hodome_object_scene_integration.py
git commit -m "test: HODome object-scene end-to-end (skipif on dataset)"
```

---

## Self-Review

**Spec coverage:**
- Composant A (scene generator, convex hull, fixed inertia, absolute mesh path) → Task 1. ✓
- Composant B (wiring: object_name + scene generated before model build) → Task 2 (object_name via façade) + Task 3 (ensure_object_scene_xml in build_from_config). ✓
- Composant C (mesh asset = tar-extracted .obj via object_source) → Task 3 (`srcs[0].mesh_path`). ✓
- Tests (generation, MuJoCo parse, has_dynamic_object, short finite solve) → Tasks 1, 3, 4. ✓
- Risk "scene xml naming/priority": resolved by writing at the historical `.replace` path in `models/<robot>/` (Task 1 default output + Task 3), so no solver change and OMOMO largebox is untouched. ✓
- Risk "meshdir": object mesh absolute path (Task 1); scene written beside robot meshes (Task 1/3 default path). ✓

**Placeholder scan:** No TBD/TODO; every code step shows code. The `_dataset_cfg` note in Task 2 is a concrete instruction (give task_config a real `TaskConfig()`), not a placeholder. ✓

**Type consistency:** `build_hodome_scene_xml(robot_xml_path, token, mesh_obj_path, output_path=None) -> Path` and `ensure_object_scene_xml(cfg, constants) -> Path | None` used identically across Tasks 1, 3, 4; `ObjectSource(mesh_path, poses_raw)` matches Spec 1. ✓

**Note (Spec 1 label correction):** before this plan, HODome object_interaction ran with `object_name` defaulting to `"largebox"` (≠ "ground", so the gate passed and the contact channel worked via `resolve_object_inputs`/`object_source`), but the label was wrong and the scene-swap path would have pointed at the committed `g1_29dof_w_largebox.xml`. Task 2 corrects the label to the token, which both names the object correctly and makes the scene swap select the generated `_w_<token>.xml`.
