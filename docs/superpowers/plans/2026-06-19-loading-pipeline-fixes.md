# Loading-pipeline fixes (#2–#5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the OMOMO object SDF available for every object (not just `largebox`),
remove CWD-relative asset paths, unify OMOMO scale on betas-FK, and pose the SMPL-X
hands in the contact probe for all three motion sources.

**Architecture:** A single package-anchored OMOMO object-mesh resolver shared by the
solver loader and the viewer; one canonical betas-FK scale function shared by the
loader and the TEST-SOCP builder; a single SMPL-X forward helper in `HumanBody` that
poses hands, fed by source-correct parent trees (SMPL-H for OMOMO, SMPL-X for
HODome/AMASS, which the prep now emits as 55-joint orientations).

**Tech Stack:** Python, numpy, smplx, trimesh, scipy, pinocchio/cvxpy (solve), pytest.

## Global Constraints

- Python env: `/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python` (has
  cvxpy, smplx, trimesh, pinocchio). The repo default `python` lacks cvxpy.
- Run pytest/eval from `HoloNew/HoloNew/` (model paths are relative to it).
- Commits carry **no** Claude trailer (no `Co-Authored-By`, no `Claude-Session`).
- Branch already created: `fix/loading-pipeline-omomo-hands-inertia`.
- Finding #6 (MJCF inertia) is out of scope — do not touch `hodome_scene.py`.
- Verified SMPL-X parents[:55] and SMPL-H parents[:52] (used in Task 4):
  - SMPL-X: `[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,15,15,15,20,25,26,20,28,29,20,31,32,20,34,35,20,37,38,21,40,41,21,43,44,21,46,47,21,49,50,21,52,53]`
  - SMPL-H: `[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,22,23,20,25,26,20,28,29,20,31,32,20,34,35,21,37,38,21,40,41,21,43,44,21,46,47,21,49,50]`
  - SMPL-X hands: left `25:40`, right `40:55`. SMPL-H hands: left `22:37`, right `37:52`.
    Same MANO order (verified), so hand pose vectors are interchangeable.

---

### Task 1: Shared OMOMO object-mesh resolver (#2) + package-anchored bundled path (#3a)

**Files:**
- Modify: `HoloNew/src/data_loaders/omomo.py` (add resolver; rewrite `object_source`)
- Modify: `HoloNew/examples/view_stages.py:222-282` (delegate to the resolver)
- Test: `HoloNew/tests/test_object_source_omomo.py` (rewrite CWD-coupled tests)
- Test: `HoloNew/tests/test_omomo_mesh_resolver.py` (new)

**Interfaces:**
- Produces: `resolve_omomo_object_mesh(seq_name: str, omomo_dir: Path | str | None = None, cache_dir: Path | str | None = None) -> Path | None`
  — bundled `<pkg>/models/<obj>/<obj>.obj` if present; else the captured unit mesh
  recentred on its vertex mean × `obj_scale`, written to a derived `.obj` and returned;
  else `None`. Raises `ValueError` if a captured mesh exists but `obj_scale` is missing.
- Produces: `_omomo_obj_name(seq_name) -> str` (2nd `_`-token, else the whole name).

- [ ] **Step 1: Write the failing resolver tests**

Create `HoloNew/tests/test_omomo_mesh_resolver.py`:

```python
import numpy as np
import joblib
import trimesh
import pytest
from pathlib import Path
from HoloNew.src.data_loaders.omomo import (
    resolve_omomo_object_mesh, _omomo_obj_name,
)


def test_obj_name_second_token():
    assert _omomo_obj_name("sub3_largebox_003") == "largebox"
    assert _omomo_obj_name("solo") == "solo"


def test_bundled_found_regardless_of_cwd(tmp_path, monkeypatch):
    # largebox is bundled in the package; resolver must find it from any cwd.
    monkeypatch.chdir(tmp_path)
    p = resolve_omomo_object_mesh("sub3_largebox_003")
    assert p is not None
    assert p.name == "largebox.obj"
    assert p.is_absolute()


def test_captured_fallback_recenters_and_scales(tmp_path):
    # Fake OMOMO release: captured unit mesh (off-origin) + a .p carrying obj_scale.
    omomo = tmp_path / "OMOMO"
    cap = omomo / "data" / "captured_objects"
    cap.mkdir(parents=True)
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    box.apply_translation([5.0, 0.0, 0.0])  # off-origin
    box.export(cap / "widget_cleaned_simplified.obj")
    (omomo / "data").mkdir(exist_ok=True)
    joblib.dump({0: {"seq_name": "sub9_widget_001", "obj_scale": np.array([2.0, 2.0])}},
                omomo / "data" / "train_diffusion_manip_seq_joints24.p")

    out = resolve_omomo_object_mesh("sub9_widget_001", omomo_dir=omomo,
                                    cache_dir=tmp_path / "cache")
    assert out is not None and out.exists()
    m = trimesh.load(str(out), force="mesh", process=False)
    v = np.asarray(m.vertices)
    # Recentred on its own centroid (~origin) and scaled ×2 -> extents 2.0.
    assert np.allclose(v.mean(0), 0.0, atol=1e-6)
    assert np.allclose(v.max(0) - v.min(0), 2.0, atol=1e-6)


def test_captured_missing_scale_raises(tmp_path):
    omomo = tmp_path / "OMOMO"
    cap = omomo / "data" / "captured_objects"
    cap.mkdir(parents=True)
    trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(
        cap / "widget_cleaned_simplified.obj")
    (omomo / "data").mkdir(exist_ok=True)
    joblib.dump({0: {"seq_name": "other"}},  # no obj_scale for our seq
                omomo / "data" / "train_diffusion_manip_seq_joints24.p")
    with pytest.raises(ValueError):
        resolve_omomo_object_mesh("sub9_widget_001", omomo_dir=omomo,
                                  cache_dir=tmp_path / "cache")


def test_no_omomo_dir_no_bundled_returns_none():
    assert resolve_omomo_object_mesh("sub9_widget_001", omomo_dir=None) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_omomo_mesh_resolver.py -q`
Expected: FAIL with `ImportError: cannot import name 'resolve_omomo_object_mesh'`.

- [ ] **Step 3: Implement the resolver in `omomo.py`**

Add near the top of `HoloNew/src/data_loaders/omomo.py` (after the existing imports):

```python
import tempfile

_OMOMO_MESH_CACHE = Path(tempfile.gettempdir()) / "holonew_omomo_meshes"


def _omomo_obj_name(seq_name: str) -> str:
    """Object token = 2nd '_'-segment of the sequence name (sub3_largebox_003 -> largebox)."""
    parts = str(seq_name).split("_")
    return parts[1] if len(parts) >= 2 else str(seq_name)


def resolve_omomo_object_mesh(seq_name, omomo_dir=None, cache_dir=None):
    """Object mesh path for an OMOMO sequence, in the centroid-centred frame the .pt
    poses expect.

    1. Bundled <pkg>/models/<obj>/<obj>.obj (already recentred + pre-scaled): returned
       as-is. Package-anchored, so independent of the current working directory.
    2. Else the captured unit mesh data/captured_objects/<obj>_cleaned_simplified.obj,
       recentred on its vertex mean and scaled by the per-sequence obj_scale, written to
       a derived .obj in cache_dir and returned (mirrors the HODome mesh cache).
    3. Else None.

    Raises ValueError if a captured mesh exists but obj_scale is unavailable (a wrong
    size is worse than a clear failure).
    """
    import trimesh
    obj = _omomo_obj_name(seq_name)
    pkg_models = Path(__file__).resolve().parents[2] / "models"
    bundled = pkg_models / obj / f"{obj}.obj"
    if bundled.exists():
        return bundled
    if omomo_dir is None:
        return None
    captured = (Path(omomo_dir) / "data" / "captured_objects"
                / f"{obj}_cleaned_simplified.obj")
    if not captured.exists():
        return None
    from HoloNew.src.test_socp.correspondence.human_metadata import load_object_scale
    scale = load_object_scale(Path(omomo_dir), str(seq_name))
    if scale is None:
        raise ValueError(
            f"obj_scale missing for {seq_name}: captured mesh {captured.name} is "
            f"off-origin and unscaled, so it cannot be sized correctly.")
    cache_dir = Path(cache_dir) if cache_dir is not None else _OMOMO_MESH_CACHE
    out = cache_dir / f"{seq_name}.obj"
    if out.exists():
        return out
    mesh = trimesh.load(str(captured), force="mesh", process=False)
    verts = np.asarray(mesh.vertices, np.float64)
    verts = (verts - verts.mean(0)) * float(scale)
    cache_dir.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=verts, faces=np.asarray(mesh.faces), process=False).export(str(out))
    return out
```

- [ ] **Step 4: Run the resolver tests to verify they pass**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_omomo_mesh_resolver.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Rewrite `object_source` to use the resolver**

Replace the body of `OmomoMixedLoader.object_source` in `HoloNew/src/data_loaders/omomo.py` (currently lines ~62-78) with:

```python
    def object_source(self, *, motion_path, obj_path, model_path, task_type,
                      constants, motion_data_config, smpl_model_dir=None):
        if task_type == "robot_only" or motion_path is None:
            return []
        from HoloNew.src.data_loaders.base import ObjectSource
        from HoloNew.src.utils import load_intermimic_data
        seq = Path(motion_path).stem
        # OMOMO release root (holds data/captured_objects + the betas/obj_scale pickle)
        # is two levels up from the pickle passed as model_path. None when unavailable
        # (then only the bundled mesh can resolve).
        omomo_dir = Path(model_path).parent.parent if model_path is not None else None
        try:
            mesh_path = resolve_omomo_object_mesh(seq, omomo_dir)
        except ValueError:
            # Captured mesh present but no obj_scale: degrade to no-object (floor-only),
            # consistent with the builder's "object mesh not found" behaviour.
            return []
        if mesh_path is None:
            return []
        _, poses = load_intermimic_data(str(motion_path))
        return [ObjectSource(mesh_path=Path(mesh_path), poses_raw=np.asarray(poses, np.float64))]
```

- [ ] **Step 6: Rewrite the CWD-coupled object_source tests**

In `HoloNew/tests/test_object_source_omomo.py`, replace
`test_omomo_object_source_no_bundled_mesh_empty` and
`test_omomo_object_source_with_bundled_mesh` with (the bundled mesh is now found
package-anchored regardless of cwd; "empty" needs a genuinely unknown object):

```python
def test_omomo_object_source_unknown_object_empty(tmp_path):
    # An object with no bundled mesh and no omomo_dir -> no source.
    p = tmp_path / "sub3_nosuchobj_003.pt"
    _fake_pt(p)
    srcs = OmomoMixedLoader().object_source(
        motion_path=p, obj_path=p, model_path=None, task_type="object_interaction",
        constants=None, motion_data_config=None)
    assert srcs == []


def test_omomo_object_source_bundled_largebox(tmp_path):
    # largebox is bundled in the package; resolved regardless of cwd, model_path=None.
    p = tmp_path / "sub3_largebox_003.pt"
    _fake_pt(p, T=5)
    srcs = OmomoMixedLoader().object_source(
        motion_path=p, obj_path=p, model_path=None, task_type="object_interaction",
        constants=None, motion_data_config=None)
    assert len(srcs) == 1
    assert srcs[0].poses_raw.shape == (5, 7)
    assert np.allclose(srcs[0].poses_raw[:, 0], 1.0)            # qw
    assert np.allclose(srcs[0].poses_raw[:, 4], np.arange(5))   # x
    assert srcs[0].mesh_path.name == "largebox.obj"
```

- [ ] **Step 7: Run the object_source tests**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_object_source_omomo.py -q`
Expected: PASS (3 passed: the robot_only test plus the two rewritten ones).

- [ ] **Step 8: Delegate the viewer to the resolver (DRY)**

In `HoloNew/examples/view_stages.py`, replace the object-file resolution block (lines
~222-240, from `parts = cfg.task_name.split("_")` through the
`obj_file = None`/`obj_geom_scale, obj_recenter` branch) with a call to the resolver,
and drop the now-redundant recenter/scale (the resolver returns an already-transformed
mesh, so `obj_geom_center = 0`, `obj_geom_scale = 1`):

```python
        from HoloNew.src.data_loaders.omomo import resolve_omomo_object_mesh
        try:
            obj_file = resolve_omomo_object_mesh(cfg.task_name, cfg.omomo_dir)
        except ValueError:
            obj_file = None
        # The resolver returns a mesh already centred on its centroid and at the
        # sequence's real size, so no further recenter/scale is applied here.
        obj_geom_scale, obj_recenter = 1.0, False
```

The `resolve_captured_obj_scale` helper (lines ~62-70) and its import of
`load_object_scale` become unused — delete the helper and the import. The downstream
code that uses `obj_geom_center`/`obj_geom_scale` is unchanged (they are now identity).

- [ ] **Step 9: Smoke-check the viewer import + resolver delegation**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -c "import HoloNew.examples.view_stages as v; from HoloNew.src.data_loaders.omomo import resolve_omomo_object_mesh; print('import OK', resolve_omomo_object_mesh('sub3_largebox_003').name)"`
Expected: `import OK largebox.obj`.

- [ ] **Step 10: Commit**

```bash
cd HoloNew
git add src/data_loaders/omomo.py examples/view_stages.py \
        tests/test_object_source_omomo.py tests/test_omomo_mesh_resolver.py
git commit -m "feat(omomo): shared package-anchored object-mesh resolver (captured fallback + obj_scale)"
```

---

### Task 2: Anchor `height_dict.pkl` to the package dir (#3b)

**Files:**
- Modify: `HoloNew/src/holosoma/preprocess.py:14-20`
- Test: `HoloNew/tests/test_paths.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces: `calculate_scale_factor(task_name, robot_height)` unchanged signature, now
  CWD-independent.

- [ ] **Step 1: Write the failing test**

Append to `HoloNew/tests/test_paths.py`:

```python
def test_calculate_scale_factor_is_cwd_independent(tmp_path, monkeypatch):
    import os
    from HoloNew.src.holosoma.preprocess import calculate_scale_factor
    # height_dict.pkl lives under the package demo_data; calling from an unrelated cwd
    # must still resolve it (sub3 is a known OMOMO subject in the bundled dict).
    monkeypatch.chdir(tmp_path)
    scale = calculate_scale_factor("sub3_largebox_003", 1.0)
    assert scale > 0.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_paths.py::test_calculate_scale_factor_is_cwd_independent -q`
Expected: FAIL with `FileNotFoundError: ... 'demo_data/height_dict.pkl'`.

- [ ] **Step 3: Anchor the pickle path to the package directory**

In `HoloNew/src/holosoma/preprocess.py`, replace:

```python
def calculate_scale_factor(task_name, robot_height):
    """Calculate scale factor based on human height."""
    with open("demo_data/height_dict.pkl", "rb") as f:
        height_dict = pickle.load(f)
```

with:

```python
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parents[2]   # .../HoloNew/HoloNew


def calculate_scale_factor(task_name, robot_height):
    """Calculate scale factor based on human height.

    Reads the bundled height table at a package-anchored path so it works from any
    working directory. (Legacy callers only; OMOMO scale is now betas-FK, see
    omomo_scale_factor.)
    """
    with open(_PKG_DIR / "demo_data" / "height_dict.pkl", "rb") as f:
        height_dict = pickle.load(f)
```

(Confirm `parents[2]` resolves to `.../HoloNew/HoloNew`: `preprocess.py` is at
`HoloNew/src/holosoma/preprocess.py`, so `[0]=holosoma, [1]=src, [2]=HoloNew`.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_paths.py::test_calculate_scale_factor_is_cwd_independent -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd HoloNew
git add src/holosoma/preprocess.py tests/test_paths.py
git commit -m "fix(scale): anchor height_dict.pkl to the package dir (CWD-independent)"
```

---

### Task 3: Canonical betas-FK OMOMO scale (#4)

**Files:**
- Modify: `HoloNew/src/data_loaders/omomo.py` (add `omomo_scale_factor`; route `.load`)
- Modify: `HoloNew/src/test_socp/builder.py:253-261`
- Test: `HoloNew/tests/test_omomo_scale.py` (new)
- Regenerate: `HoloNew/tests/golden/inertia_mode_qpos.npz`

**Interfaces:**
- Consumes: `omomo_height_from_betas` and `_betas_for_seq` (existing in `omomo.py`),
  `load_human_metadata` (`correspondence/human_metadata.py`), `get_path`.
- Produces: `omomo_scale_factor(seq: str, robot_height: float, omomo_dir: Path | str | None, smplh_model_dir: Path | str | None, default_human_height: float = 1.78) -> float`
  — `robot_height / betas_FK_height`; falls back to `robot_height / default_human_height`
  when betas or the SMPL-H model are unavailable.

- [ ] **Step 1: Write the failing test**

Create `HoloNew/tests/test_omomo_scale.py`:

```python
import os
import pytest
from pathlib import Path
from HoloNew.src.data_loaders.omomo import omomo_scale_factor
from HoloNew.src.paths import get_path

_SMPLH = get_path("smplh_models")
_OMOMO = get_path("omomo")
_HAVE = _SMPLH.is_dir() and (_OMOMO / "data").is_dir()


def test_scale_fallback_when_model_missing(tmp_path):
    # No SMPL-H model dir -> fall back to robot_height/default_human_height.
    s = omomo_scale_factor("sub3_largebox_003", robot_height=1.4,
                           omomo_dir=tmp_path, smplh_model_dir=tmp_path / "nope",
                           default_human_height=1.75)
    assert s == pytest.approx(1.4 / 1.75)


@pytest.mark.skipif(not _HAVE, reason="OMOMO + SMPL-H assets not present")
def test_scale_is_betas_fk_and_positive():
    s = omomo_scale_factor("sub3_largebox_003", robot_height=1.4,
                           omomo_dir=_OMOMO, smplh_model_dir=_SMPLH)
    assert 0.5 < s < 1.2   # G1 ~1.4 m vs adult human height
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_omomo_scale.py -q`
Expected: FAIL with `ImportError: cannot import name 'omomo_scale_factor'`.

- [ ] **Step 3: Implement `omomo_scale_factor` and route `.load` through it**

Add to `HoloNew/src/data_loaders/omomo.py`:

```python
def omomo_scale_factor(seq, robot_height, omomo_dir, smplh_model_dir,
                       default_human_height: float = 1.78) -> float:
    """Canonical OMOMO scale = robot_height / stature(seq), where stature comes from
    SMPL-H rest-pose FK on the subject's betas (omomo_height_from_betas).

    Falls back to robot_height / default_human_height when the betas metadata or the
    SMPL-H model directory is unavailable, so callers degrade rather than crash.
    """
    from HoloNew.src.test_socp.correspondence.human_metadata import load_human_metadata
    if smplh_model_dir is None or not Path(smplh_model_dir).is_dir():
        return float(robot_height) / float(default_human_height)
    betas, gender = load_human_metadata(Path(omomo_dir), str(seq)) if omomo_dir else (None, "neutral")
    if betas is None:
        return float(robot_height) / float(default_human_height)
    height = omomo_height_from_betas(betas, gender, Path(smplh_model_dir))
    return float(robot_height) / float(height)
```

Then route `OmomoMixedLoader.load` through it. Replace, in `load`:

```python
        betas, gender = _betas_for_seq(Path(model_path), Path(motion_path).stem)
        height = omomo_height_from_betas(betas, gender, Path(smpl_model_dir))
        smpl_scale = float(constants.ROBOT_HEIGHT) / height
```

with:

```python
        omomo_dir = Path(model_path).parent.parent
        smpl_scale = omomo_scale_factor(
            Path(motion_path).stem, float(constants.ROBOT_HEIGHT),
            omomo_dir, Path(smpl_model_dir))
```

- [ ] **Step 4: Switch the TEST-SOCP builder to the canonical scale**

In `HoloNew/src/test_socp/builder.py`, replace the OMOMO scale block (currently
lines ~253-261):

```python
    from HoloNew.src.holosoma.preprocess import calculate_scale_factor
    _robot_h = constants.ROBOT_HEIGHT
    if data_format == "smplx":
        smpl_scale = _robot_h / float(_smplx_height)
    else:
        try:
            smpl_scale = calculate_scale_factor(cfg.task_name, _robot_h)
        except Exception:  # noqa: BLE001 - height table may lack this subject
            smpl_scale = _robot_h / (cfg.motion_data_config.default_human_height or 1.78)
```

with:

```python
    _robot_h = constants.ROBOT_HEIGHT
    if data_format == "smplx":
        smpl_scale = _robot_h / float(_smplx_height)
    else:
        # Canonical OMOMO scale: betas-FK stature (single source shared with the
        # registry loader). smplh model dir from the config, else the path.yaml root.
        from HoloNew.src.data_loaders.omomo import omomo_scale_factor
        from HoloNew.src.paths import get_path
        _smplh_dir = getattr(cfg, "smpl_model_dir", None)
        if _smplh_dir is None:
            try:
                _smplh_dir = get_path("smplh_models")
            except Exception:  # noqa: BLE001 - path.yaml may lack the key
                _smplh_dir = None
        _omomo_dir = getattr(cfg, "omomo_dir", None)
        smpl_scale = omomo_scale_factor(
            cfg.task_name, _robot_h, _omomo_dir, _smplh_dir,
            default_human_height=(cfg.motion_data_config.default_human_height or 1.78))
```

- [ ] **Step 5: Run the scale tests**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_omomo_scale.py -q`
Expected: PASS (`test_scale_fallback_when_model_missing` passes; the betas-FK test
passes if assets present, else skips).

- [ ] **Step 6: Verify the golden drift cause, then regenerate**

First confirm the new vs old scale for the golden sequence (sanity, not a blind regen):

```bash
cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python - <<'PY'
from HoloNew.src.data_loaders.omomo import omomo_scale_factor
from HoloNew.src.holosoma.preprocess import calculate_scale_factor
from HoloNew.src.paths import get_path
rh = 1.4  # informational; the solve uses constants.ROBOT_HEIGHT
new = omomo_scale_factor("sub3_largebox_003", rh, get_path("omomo"), get_path("smplh_models"))
old = calculate_scale_factor("sub3_largebox_003", rh)
print(f"betas-FK scale={new:.5f}  height_dict scale={old:.5f}  ratio={new/old:.4f}")
PY
```

Expected: two positive numbers that differ by a few percent (betas-FK vs table). Record
the ratio in the commit message.

Then regenerate the golden:

```bash
cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python - <<'PY'
import numpy as np
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.tests.paper_placement import paper_placement_config
rt = TestSocpRetargeter.from_config(RetargetingConfig(
    task_type="object_interaction", task_name="sub3_largebox_003",
    data_format="smplh", retargeter=paper_placement_config()))
assert rt.correspondence is not None, "assets missing; cannot regenerate golden"
res = rt.retarget(max_frames=30)
np.savez("tests/golden/inertia_mode_qpos.npz", qpos=res.qpos)
print("regenerated inertia_mode_qpos.npz", res.qpos.shape)
PY
```

- [ ] **Step 7: Run the inertia golden test**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_inertia_mode_golden.py -q`
Expected: PASS (or SKIP if assets absent — then the golden was not regenerated and the
prior file is unchanged).

- [ ] **Step 8: Confirm the GMR golden is untouched**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_retarget_golden.py -q`
Expected: PASS (load_motion_data still uses the height_dict legacy path; baseline
unchanged).

- [ ] **Step 9: Commit**

```bash
cd HoloNew
git add src/data_loaders/omomo.py src/test_socp/builder.py tests/test_omomo_scale.py \
        tests/golden/inertia_mode_qpos.npz
git commit -m "feat(scale): canonical betas-FK OMOMO scale shared by loader+builder; regen inertia golden (ratio <fill in>)"
```

---

### Task 4: Pose SMPL-X hands in the contact probe (#5)

**Files:**
- Modify: `HoloNew/src/test_socp/correspondence/human_body.py` (shared forward + both posers)
- Modify: `HoloNew/data_utils/prep_amass_smplx_for_rt.py` (`compute_global_joint_orientations` N-joint; emit 55)
- Modify: `HoloNew/src/data_loaders/hodome.py` (`global_orientations_zup` + `prep_hodome_processed` emit 55)
- Test: `HoloNew/tests/test_hand_posing.py` (new)
- Regenerate: `HoloNew/tests/golden/inertia_mode_qpos.npz`

**Interfaces:**
- Consumes: SMPL-X model from `HumanBody` (built `use_pca=False`); the verified parent
  arrays from Global Constraints.
- Produces:
  - module constant `SMPLH_PARENTS` (np.int64 array, the verified 52-entry SMPL-H tree).
  - `HumanBody.placed_verts(quats_wxyz_52_mujoco, pelvis_target, frame_idx=None)` — now
    poses hands using `SMPLH_PARENTS` and SMPL-H hand slots `22:37` / `37:52`.
  - `HumanBody.placed_verts_smpl(quats_wxyz, pelvis_target, frame_idx=None)` — accepts
    `(J,4)` with `J ∈ {22, 55}`; `J>=55` poses hands using the model's native parents and
    SMPL-X hand slots `25:40` / `40:55`.
  - `compute_global_joint_orientations(aa_rot, parents)` — already N-joint via `parents`;
    callers now pass 55.
  - prep npz key `global_joint_orientations` is now `(T, 55, 4)` (was `(T, 22, 4)`).

- [ ] **Step 1: Write the failing tests**

Create `HoloNew/tests/test_hand_posing.py`:

```python
import numpy as np
import pytest
from HoloNew.src.test_socp.correspondence.constants import SMPLX_MODEL_DIR_DEFAULT
from HoloNew.src.test_socp.correspondence.human_body import HumanBody, SMPLH_PARENTS

_HAVE = __import__("os").path.isdir(SMPLX_MODEL_DIR_DEFAULT)
pytestmark = pytest.mark.skipif(not _HAVE, reason="SMPL-X model not present")


def _body():
    return HumanBody(SMPLX_MODEL_DIR_DEFAULT, betas=None, gender="neutral")


def test_smplh_parents_match_verified_array():
    expected = np.array(
        [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,
         20,22,23,20,25,26,20,28,29,20,31,32,20,34,35,
         21,37,38,21,40,41,21,43,44,21,46,47,21,49,50], dtype=np.int64)
    np.testing.assert_array_equal(SMPLH_PARENTS, expected)


def test_placed_verts_smpl_curled_hand_moves_fingertips():
    body = _body()
    pelvis = np.zeros(3)
    flat = np.zeros((55, 4), np.float32); flat[:, 0] = 1.0          # all identity
    curled = flat.copy()
    # Curl every left-hand joint (25..39) about local x by ~1 rad (global approx; the
    # test only needs a large, finite displacement vs the flat hand).
    from scipy.spatial.transform import Rotation as R
    q = R.from_rotvec([1.0, 0.0, 0.0]).as_quat()[[3, 0, 1, 2]]      # wxyz
    curled[25:40] = q
    v_flat = body.placed_verts_smpl(flat, pelvis)
    v_curled = body.placed_verts_smpl(curled, pelvis)
    disp = np.linalg.norm(v_curled - v_flat, axis=1)
    assert disp.max() > 0.02            # hand vertices moved a couple cm+
    assert np.isfinite(v_curled).all()  # mesh did not explode


def test_placed_verts_smpl_backward_compat_22():
    body = _body()
    q22 = np.zeros((22, 4), np.float32); q22[:, 0] = 1.0
    v = body.placed_verts_smpl(q22, np.zeros(3))
    assert v.shape[0] > 1000 and np.isfinite(v).all()   # still produces a body mesh


def test_placed_verts_omomo_poses_hands():
    body = _body()
    # 52 MuJoCo-order quats; identity everywhere, then curl the left-hand block.
    flat = np.zeros((52, 4), np.float32); flat[:, 0] = 1.0
    curled = flat.copy()
    from scipy.spatial.transform import Rotation as R
    q = R.from_rotvec([1.0, 0.0, 0.0]).as_quat()[[3, 0, 1, 2]]
    # MuJoCo order scatters into SMPL slots; curl a wide block to guarantee hand slots hit.
    curled[:] = q
    v_flat = body.placed_verts(flat, np.zeros(3))
    v_curled = body.placed_verts(curled, np.zeros(3))
    assert np.linalg.norm(v_curled - v_flat, axis=1).max() > 0.02
    assert np.isfinite(v_curled).all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_hand_posing.py -q`
Expected: FAIL with `ImportError: cannot import name 'SMPLH_PARENTS'`.

- [ ] **Step 3: Add `SMPLH_PARENTS`, `flat_hand_mean=True`, and the shared forward helper**

In `HoloNew/src/test_socp/correspondence/human_body.py`, add the constant near the top
(after `_SMPL_IDX`):

```python
# Verified SMPL-H 52-joint parent tree (body identical to SMPL-X; hands are the same
# MANO sub-tree, left 22:37 / right 37:52). Used to pose OMOMO hands, whose .pt quats
# are SMPL-H-ordered, with the correct parents (the SMPL-X model's parents place the
# 3 face joints at 22:25, which would misalign the hand chains).
SMPLH_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
     20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35,
     21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50], dtype=np.int64)
```

In `HumanBody.__init__`, change the model creation so provided hand poses are used as
absolute local rotations (no MANO mean offset):

```python
        self.model = smplx.create(
            model_dir, model_type="smplx", gender=gender or "neutral",
            use_pca=False, flat_hand_mean=True, num_betas=num_betas, batch_size=1,
        )
```

Add the shared forward helper as a method on `HumanBody`:

```python
    def _forward_posed(self, q_smpl_xyzw, parents, pelvis_target,
                       left_hand, right_hand):
        """Posed SMPL-X vertices for one frame from per-joint GLOBAL rotations.

        q_smpl_xyzw: (J,4) xyzw global per-joint rotations.
        parents: (J,) parent index per joint (root = -1) in the SAME order as q.
        left_hand / right_hand: (start, stop) slices selecting the 15 left/right hand
            joints in q, or None to leave that hand at the model default.
        """
        from scipy.spatial.transform import Rotation as R
        torch = self._torch
        q = np.asarray(q_smpl_xyzw, dtype=float)
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        q = np.where(norms > 1e-6, q / norms, np.array([0, 0, 0, 1.0]))
        global_rots = R.from_quat(q).as_matrix()                      # (J,3,3)
        par = np.asarray(parents)
        rel = np.matmul(np.transpose(global_rots[par], (0, 2, 1)), global_rots)
        rel[par == -1] = global_rots[par == -1]
        rotvec = R.from_matrix(rel).as_rotvec()                       # (J,3)
        kw = dict(
            global_orient=torch.from_numpy(rotvec[0]).float().view(1, 3),
            body_pose=torch.from_numpy(rotvec[1:22]).float().view(1, -1),
            betas=self._betas, return_verts=True, return_joints=True,
        )
        if left_hand is not None:
            s0, s1 = left_hand
            kw["left_hand_pose"] = torch.from_numpy(rotvec[s0:s1]).float().view(1, -1)
        if right_hand is not None:
            s0, s1 = right_hand
            kw["right_hand_pose"] = torch.from_numpy(rotvec[s0:s1]).float().view(1, -1)
        with torch.no_grad():
            output = self.model(**kw)
        verts = output.vertices[0].detach().cpu().numpy()
        pelvis = output.joints[0, 0].detach().cpu().numpy()
        return verts - pelvis + pelvis_target
```

- [ ] **Step 4: Rewrite `placed_verts` (OMOMO) to use the helper with SMPL-H hands**

Replace the body of `HumanBody.placed_verts` (after the cache check and the MuJoCo→SMPL
scatter that builds `q_smpl_xyzw`) so the final posing goes through the helper. The
scatter stays; replace everything from the normalisation down to `out = verts - pelvis +
pelvis_target` with:

```python
        out = self._forward_posed(
            q_smpl_xyzw, SMPLH_PARENTS, pelvis_target,
            left_hand=(22, 37), right_hand=(37, 52))
        if frame_idx is not None:
            self._cache_idx = frame_idx
            self._cache_verts = out
        return out
```

(`q_smpl_xyzw` is the (52,4) xyzw array the existing scatter already builds. The helper
re-normalises, so the old explicit normalisation lines are removed.)

- [ ] **Step 5: Rewrite `placed_verts_smpl` to accept J∈{22,55} and pose hands**

Replace the body of `HumanBody.placed_verts_smpl` (after the cache check) with:

```python
        q_in = np.asarray(quats_wxyz_22)
        J = q_in.shape[0]
        q_smpl_xyzw = np.zeros((max(J, 22), 4))
        q_smpl_xyzw[:, 3] = 1.0
        q_smpl_xyzw[:J] = q_in[:, [1, 2, 3, 0]]
        if J >= 55:
            parents = self.parents[:55]
            left_hand, right_hand = (25, 40), (40, 55)
        else:
            parents = self.parents[:max(J, 22)]
            left_hand = right_hand = None        # 22-joint legacy: hands at default
        out = self._forward_posed(q_smpl_xyzw, parents, pelvis_target,
                                  left_hand=left_hand, right_hand=right_hand)
        if frame_idx is not None:
            self._cache_idx = frame_idx
            self._cache_verts = out
        return out
```

Note: `self.parents` is the SMPL-X model's native parent tree (55), which is correct for
the SMPL-X hand slots `25:40` / `40:55`.

- [ ] **Step 6: Run the hand-posing tests**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_hand_posing.py -q`
Expected: PASS (4 passed, or skipped if SMPL-X model absent).

- [ ] **Step 7: Run the broader human-body / probe suite for regressions**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_correspondence_human.py tests/test_smplx_field.py tests/test_skeleton.py -q`
Expected: PASS (no regressions in correspondence/probe construction).

- [ ] **Step 8: Emit 55-joint orientations from the AMASS prep**

In `HoloNew/data_utils/prep_amass_smplx_for_rt.py`, the body of `main()` currently
computes `aa_body = aa_rot_52[...][:, :num_body_joints, :]` and 22-joint orientations.
Change it to use the full 55 joints. Replace:

```python
        bm_neutral = bm_dict["neutral"]
        kintree = getattr(bm_neutral, "kintree_table", None)
        if kintree is not None:
            parents = np.asarray(kintree)[0][:num_body_joints].astype(np.int64).copy()
            parents[0] = -1
        else:
            parents = _SMPLX_BODY_PARENTS
        aa_body = aa_rot_52.squeeze(0).detach().cpu().numpy()[:, :num_body_joints, :]  # T X 22 X 3
        global_joint_orientations = compute_global_joint_orientations(aa_body, parents)  # T X 22 X 4 wxyz
```

with:

```python
        # Full 55-joint SMPL-X axis-angle (body + face + both MANO hands) so the contact
        # probe can pose the hands. aa_rot_rep is the raw (T, 55, 3) pose; take all 55.
        bm_neutral = bm_dict["neutral"]
        kintree = getattr(bm_neutral, "kintree_table", None)
        if kintree is not None:
            parents55 = np.asarray(kintree)[0][:55].astype(np.int64).copy()
            parents55[0] = -1
        else:
            parents55 = _SMPLX_PARENTS_55
        aa_full = aa_rot_rep.reshape(-1, 55, 3)  # T X 55 X 3, native SMPL-X order
        global_joint_orientations = compute_global_joint_orientations(aa_full, parents55)  # T X 55 X 4 wxyz
```

Add the fallback constant near `_SMPLX_BODY_PARENTS`:

```python
# Full SMPL-X 55-joint parent tree (fallback when the body model does not expose a
# kintree). Body 0-21, face 22-24, left hand 25-39, right hand 40-54.
_SMPLX_PARENTS_55 = np.array([
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
    15, 15, 15, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38,
    21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53], dtype=np.int64)
```

(`aa_rot_rep` is `data["poses"]` reshaped; confirm it is the full 165=55×3 vector before
this point. The positions output `global_joint_positions[:, :num_body_joints]` is
unchanged.)

- [ ] **Step 9: Emit 55-joint orientations from the HODome prep**

In `HoloNew/src/data_loaders/hodome.py`, generalize `global_orientations_zup` to take the
full pose and FK 55 joints. Replace its signature/body:

```python
def global_orientations_zup(global_orient, body_pose, left_hand_pose=None,
                            right_hand_pose=None, jaw_pose=None,
                            leye_pose=None, reye_pose=None) -> np.ndarray:
    """Per-joint global orientations (T, 55, 4) WXYZ in Z-up from raw SMPL-X locals.

    Builds the full 55-joint axis-angle (body + face + both MANO hands), FK down the
    SMPL-X tree, then expresses each rotation in Z-up by conjugating with Q (R' = Q R Q^T),
    consistent with the joints/object/mesh transform. Missing optional components default
    to zero (identity local rotation)."""
    from HoloNew.data_utils.prep_amass_smplx_for_rt import (
        _SMPLX_PARENTS_55, compute_global_joint_orientations,
    )
    go = np.asarray(global_orient, np.float64).reshape(-1, 1, 3)
    T = go.shape[0]
    bp = np.asarray(body_pose, np.float64).reshape(T, 21, 3)

    def _opt(x, n):  # (T, n, 3) zeros if absent
        return (np.zeros((T, n, 3)) if x is None
                else np.asarray(x, np.float64).reshape(T, n, 3))

    jaw = _opt(jaw_pose, 1); leye = _opt(leye_pose, 1); reye = _opt(reye_pose, 1)
    lh = _opt(left_hand_pose, 15); rh = _opt(right_hand_pose, 15)
    aa = np.concatenate([go, bp, jaw, leye, reye, lh, rh], axis=1)   # (T,55,3)
    q_yup = compute_global_joint_orientations(aa, _SMPLX_PARENTS_55)  # (T,55,4) wxyz, Y-up
    t, j, _ = q_yup.shape
    Rm = R.from_quat(q_yup[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix()
    Rz = _YUP_TO_ZUP @ Rm @ _YUP_TO_ZUP.T
    q_xyzw = R.from_matrix(Rz).as_quat().reshape(t, j, 4)
    return q_xyzw[..., [3, 0, 1, 2]].astype(np.float32)              # -> wxyz
```

Then update `prep_hodome_processed` to pass the hand/face components:

```python
    quats = global_orientations_zup(
        d["global_orient"], d["body_pose"],
        left_hand_pose=d["left_hand_pose"], right_hand_pose=d["right_hand_pose"],
        jaw_pose=d["jaw_pose"], leye_pose=d["leye_pose"], reye_pose=d["reye_pose"])
```

- [ ] **Step 10: Test the prep orientation shape**

Append to `HoloNew/tests/test_hodome_prep.py` (or create a small test if the file gates
on the dataset) a shape check using the existing HODome prep fixture path; if that file
already `skipif`s on the dataset, add:

```python
def test_prep_hodome_emits_55_orientations(hodome_npz, smplx_model_dir):
    from HoloNew.src.data_loaders.hodome import prep_hodome_processed
    out = prep_hodome_processed(hodome_npz, smplx_model_dir)
    assert out["global_joint_orientations"].shape[1] == 55
    assert out["global_joint_positions"].shape[1] == 22
```

If `test_hodome_prep.py` has no such fixtures, instead assert the contract directly in
`tests/test_hand_posing.py` using a synthetic 55-joint call already covered in Step 1 —
in that case skip this step and note it in the commit.

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_hodome_prep.py -q`
Expected: PASS or SKIP (dataset-gated).

- [ ] **Step 11: Regenerate the inertia golden (hands now posed) with a sanity check**

The OMOMO probe now poses hands, changing the contact field near the hand links. Confirm
the change is localised (body away from hands ~unchanged) before blessing:

```bash
cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python - <<'PY'
import numpy as np
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.tests.paper_placement import paper_placement_config
old = np.load("tests/golden/inertia_mode_qpos.npz")["qpos"]
rt = TestSocpRetargeter.from_config(RetargetingConfig(
    task_type="object_interaction", task_name="sub3_largebox_003",
    data_format="smplh", retargeter=paper_placement_config()))
assert rt.correspondence is not None, "assets missing"
new = rt.retarget(max_frames=30).qpos
print("max |Δqpos| =", float(np.abs(new - old).max()))
np.savez("tests/golden/inertia_mode_qpos.npz", qpos=new)
print("regenerated; shape", new.shape)
PY
```

Expected: a finite, modest `max |Δqpos|` (hands changing contact, not an explosion).
Record it in the commit message.

- [ ] **Step 12: Run the inertia golden + hand tests together**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_hand_posing.py tests/test_inertia_mode_golden.py -q`
Expected: PASS (or SKIP where assets are absent).

- [ ] **Step 13: Commit**

```bash
cd HoloNew
git add src/test_socp/correspondence/human_body.py data_utils/prep_amass_smplx_for_rt.py \
        src/data_loaders/hodome.py tests/test_hand_posing.py tests/test_hodome_prep.py \
        tests/golden/inertia_mode_qpos.npz
git commit -m "feat(probe): pose SMPL-X hands for OMOMO+HODome+AMASS (source-correct parents); regen inertia golden (max dq <fill in>)"
```

---

### Task 5: Full-suite regression sweep

**Files:** none (verification only).

- [ ] **Step 1: Run the loading + probe + golden tests**

Run:
```bash
cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest \
  tests/test_omomo_mesh_resolver.py tests/test_object_source_omomo.py \
  tests/test_object_center_scale.py tests/test_paths.py tests/test_omomo_scale.py \
  tests/test_hand_posing.py tests/test_hodome_prep.py tests/test_data_loaders_hodome.py \
  tests/test_smplx_field.py tests/test_correspondence_human.py \
  tests/test_retarget_golden.py tests/test_inertia_mode_golden.py -q
```
Expected: all PASS or dataset-gated SKIP; no failures.

- [ ] **Step 2: Final completion check**

Confirm: every spec section maps to a task (resolver #2/#3a → T1, height_dict #3b → T2,
betas-FK #4 → T3, hands #5 → T4), no golden left red, `git status` clean.

## Self-Review

- **Spec coverage:** #2/#3 → Task 1 (+ Task 2 for the height_dict path); #4 → Task 3;
  #5 → Task 4; golden impact handled in Tasks 3 & 4; full sweep Task 5. No gaps.
- **Placeholder scan:** the only intentional `<fill in>` tokens are commit-message
  numbers measured at run time (drift ratio / max Δqpos); every code/test step contains
  real code. Step 10 has a documented conditional fallback, not a placeholder.
- **Type consistency:** `resolve_omomo_object_mesh` / `_omomo_obj_name` /
  `omomo_scale_factor` signatures match between definition (T1/T3) and callers
  (object_source, builder, viewer); `SMPLH_PARENTS` and `_SMPLX_PARENTS_55` are defined
  before use; `_forward_posed` hand-slot ranges match the verified layouts.
