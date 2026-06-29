# Paths-Config (`paths.toml`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop retyping absolute dataset/model paths on every CLI call by reading them from a machine-local `paths.toml`, consumed only at the CLI/entry edge.

**Architecture:** A pure stdlib resolver (`src/paths.py`) reads `HoloV2/paths.toml` via `tomllib`. A small argparse→`SceneSpec` glue (`src/viz/_scene_args.py`) fills `--model-dir`/`--dataset-root` defaults from it and resolves relative motion paths, then the viz `main()` functions delegate to it (removing duplicated `RobotSpec`/`SceneSpec` boilerplate and the placeholder URDF). The pure pipeline (`prepare`/`targets`) never imports any of this.

**Tech Stack:** Python 3.11, stdlib `tomllib` (no new dependency), pytest, argparse.

## Global Constraints

- **Python**: run everything with `~/.holonew_deps/miniconda3/envs/holonew/bin/python` (the env with deps). Shorthand below: `PY`.
- **Working directory**: all commands run from `HoloV2/` (`cd .../HoloNew/HoloV2`). The `conftest.py` puts the repo root on `sys.path` so tests import `from src.… import …`.
- **No new third-party dependency**: read TOML with stdlib `tomllib` only (`pyyaml` is NOT installed).
- **Edge-only**: `src/paths.py` and `src/viz/_scene_args.py` are imported ONLY by entry points; `prepare`/`targets` must never import them. Paths are NOT knobs → never added to `config.py`/`PrepareConfig`.
- **Commits**: conventional commits; **never** add a `Co-Authored-By`/Claude/Anthropic trailer. Author is the repo's configured `Guillaume-Bsst`.
- Set once per shell: `export PY=~/.holonew_deps/miniconda3/envs/holonew/bin/python && cd /home/gbesset/Documents/wbt_rl/modules/01_retargeting/HoloNew/HoloV2`

---

### Task 1: Pure path resolver `src/paths.py`

**Files:**
- Create: `src/paths.py`
- Test: `tests/test_paths.py`

**Interfaces:**
- Consumes: nothing (stdlib only).
- Produces:
  - `HOLOV2_ROOT: Path`, `PATHS_TOML: Path`, `PATHS_EXAMPLE: Path`
  - `load_paths(path: Path | None = None) -> dict`
  - `smplx_dir(cfg: dict | None = None, *, path: Path | None = None) -> Path`
  - `dataset_root(dataset: str, cfg: dict | None = None, *, path: Path | None = None) -> Path`
  - `resolve_motion(dataset: str, motion: str | Path, cfg: dict | None = None, *, path: Path | None = None) -> Path`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_paths.py`:

```python
from pathlib import Path

import pytest

from src import paths


def _write(tmp_path: Path) -> Path:
    p = tmp_path / "paths.toml"
    p.write_text(
        'smplx = "/models/smplx"\n'
        "[roots]\n"
        'hodome = "/data/HODome"\n'
    )
    return p


def test_load_paths_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        paths.load_paths(tmp_path / "nope.toml")


def test_load_paths_roundtrip(tmp_path):
    cfg = paths.load_paths(_write(tmp_path))
    assert cfg["smplx"] == "/models/smplx"
    assert cfg["roots"]["hodome"] == "/data/HODome"


def test_smplx_dir(tmp_path):
    cfg = paths.load_paths(_write(tmp_path))
    assert paths.smplx_dir(cfg) == Path("/models/smplx")


def test_smplx_dir_missing_key(tmp_path):
    p = tmp_path / "paths.toml"
    p.write_text('[roots]\nhodome = "/data/HODome"\n')
    with pytest.raises(ValueError):
        paths.smplx_dir(path=p)


def test_dataset_root(tmp_path):
    cfg = paths.load_paths(_write(tmp_path))
    assert paths.dataset_root("hodome", cfg) == Path("/data/HODome")


def test_dataset_root_missing_key(tmp_path):
    cfg = paths.load_paths(_write(tmp_path))
    with pytest.raises(ValueError):
        paths.dataset_root("omomo", cfg)


def test_resolve_motion_absolute_passthrough(tmp_path):
    cfg = paths.load_paths(_write(tmp_path))
    absolute = Path("/somewhere/seq.npz")
    assert paths.resolve_motion("hodome", absolute, cfg) == absolute


def test_resolve_motion_relative_joins_root(tmp_path):
    cfg = paths.load_paths(_write(tmp_path))
    assert paths.resolve_motion("hodome", "smplx/s01.npz", cfg) == Path("/data/HODome/smplx/s01.npz")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_paths.py -q`
Expected: FAIL (collection/import error: `ModuleNotFoundError: No module named 'src.paths'`).

- [ ] **Step 3: Write the resolver**

Create `src/paths.py`:

```python
"""Machine-local path registry for HoloV2 (dataset roots + SMPL-X model dir).

EDGE concern (effets de bord aux extrémités): read ONLY by CLI/entry points, never by the
pure pipeline (prepare/targets). Source of truth = HoloV2/paths.toml (gitignored,
machine-local; copy paths.example.toml). Parsed with the stdlib tomllib — no third-party dep.
These are environment paths, NOT algorithmic knobs, so they live here and never in config.py.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

HOLOV2_ROOT = Path(__file__).resolve().parents[1]   # .../HoloV2 (where paths.toml lives)
PATHS_TOML = HOLOV2_ROOT / "paths.toml"
PATHS_EXAMPLE = HOLOV2_ROOT / "paths.example.toml"


def load_paths(path: Path | None = None) -> dict:
    """Parse paths.toml -> dict, e.g. {"smplx": str, "roots": {dataset: str}}.

    Raises FileNotFoundError (pointing at the example template) when the file is absent.
    """
    p = Path(path) if path is not None else PATHS_TOML
    if not p.exists():
        raise FileNotFoundError(
            f"paths config not found: {p}. Copy the template: "
            f"`cp {PATHS_EXAMPLE.name} {p.name}` then edit your machine paths.")
    with open(p, "rb") as fh:
        return tomllib.load(fh)


def smplx_dir(cfg: dict | None = None, *, path: Path | None = None) -> Path:
    """SMPL-X model dir (folder holding SMPLX_NEUTRAL.npz). ValueError if the key is unset."""
    cfg = cfg if cfg is not None else load_paths(path)
    val = cfg.get("smplx")
    if not val:
        raise ValueError(f"paths.toml is missing 'smplx' (the SMPL-X model dir). Set it in {PATHS_TOML}.")
    return Path(val)


def dataset_root(dataset: str, cfg: dict | None = None, *, path: Path | None = None) -> Path:
    """Release root for `dataset` from the [roots] table. ValueError if the key is unset."""
    cfg = cfg if cfg is not None else load_paths(path)
    val = (cfg.get("roots") or {}).get(dataset)
    if not val:
        raise ValueError(
            f"paths.toml is missing roots.{dataset!r}. Add it under [roots] in {PATHS_TOML} "
            f"(or pass an absolute --motion-path / --dataset-root).")
    return Path(val)


def resolve_motion(dataset: str, motion: str | Path, cfg: dict | None = None,
                   *, path: Path | None = None) -> Path:
    """Resolve a motion path: absolute -> as-is; relative -> dataset_root(dataset)/motion."""
    m = Path(motion)
    if m.is_absolute():
        return m
    cfg = cfg if cfg is not None else load_paths(path)
    return dataset_root(dataset, cfg) / m
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_paths.py -q`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add src/paths.py tests/test_paths.py
git commit -m "feat(holov2): paths.py — machine-local TOML path resolver"
```

---

### Task 2: Config files + gitignore

**Files:**
- Create: `paths.example.toml` (committed template)
- Create: `paths.toml` (machine-local, gitignored — real paths)
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `src.paths.load_paths` (Task 1) for the smoke check.
- Produces: a real `paths.toml` so the viz CLIs work without `--model-dir`/`--dataset-root`.

- [ ] **Step 1: Create the committed template `paths.example.toml`**

```toml
# HoloV2 machine-local path registry, read at the CLI edge by src/paths.py.
# Setup: `cp paths.example.toml paths.toml` then edit the paths for your machine.
# A relative --motion-path on the CLI resolves against the dataset's [roots] entry;
# an absolute --motion-path is used as-is (use absolute for OMOMO — see note below).

smplx = "/abs/path/to/models_smplx_v1_1/models/smplx"   # folder holding SMPLX_NEUTRAL.npz

[roots]
# Dataset release roots. Used as the default --dataset-root (object/betas metadata)
# AND as the base for relative --motion-path.
hodome = "/abs/path/to/HODome"     # motion: smplx/<seq>.npz lives under this root
sfu    = "/abs/path/to/SFU/SFU"    # motion: <subject>/<seq>.npz lives under this root
omomo  = "/abs/path/to/OMOMO"      # metadata release (betas/scales/meshes).
# NOTE: OMOMO InterMimic .pt motion files live in a SEPARATE OMOMO_new/ tree, NOT under
# this root — pass an absolute --motion-path for OMOMO; roots.omomo stays the metadata root.
```

- [ ] **Step 2: Create the machine-local `paths.toml` (gitignored) with real paths**

```toml
smplx = "/home/gbesset/Documents/wbt_rl/data/00_raw_datasets/models/models_smplx_v1_1/models/smplx"

[roots]
hodome = "/home/gbesset/Documents/wbt_rl/data/00_raw_datasets/HODome"
sfu    = "/home/gbesset/Documents/wbt_rl/data/00_raw_datasets/SFU/SFU"
omomo  = "/home/gbesset/Documents/wbt_rl/data/00_raw_datasets/OMOMO"
```

- [ ] **Step 3: Add `paths.toml` to `.gitignore`**

Append these two lines to `.gitignore` (keep the example tracked):

```
# Machine-local path registry (copy from paths.example.toml). Keep the example tracked.
paths.toml
```

- [ ] **Step 4: Verify the resolver reads the real file and that paths.toml is ignored**

Run:
```bash
$PY -c "from src import paths; cfg = paths.load_paths(); print(paths.smplx_dir(cfg)); print(paths.dataset_root('hodome', cfg))"
git check-ignore paths.toml
git status --short
```
Expected: prints the two real absolute paths; `git check-ignore` prints `paths.toml`; `git status` shows `paths.example.toml` and `.gitignore` as changes but NOT `paths.toml`.

- [ ] **Step 5: Commit**

```bash
git add paths.example.toml .gitignore
git commit -m "feat(holov2): paths.example.toml template + gitignore machine-local paths.toml"
```

---

### Task 3: Argparse → SceneSpec glue `src/viz/_scene_args.py`

**Files:**
- Create: `src/viz/_scene_args.py`
- Test: `tests/test_scene_args.py`

**Interfaces:**
- Consumes: `src.paths` (Task 1); `src.prepare.contracts.{RobotSpec, SceneSpec}`.
- Produces:
  - `add_scene_args(ap: argparse.ArgumentParser) -> None`
  - `scene_from_args(a: argparse.Namespace, *, paths_file: Path | None = None) -> SceneSpec`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scene_args.py`:

```python
import argparse
from pathlib import Path

from src.viz._scene_args import add_scene_args, scene_from_args


def _toml(tmp_path: Path) -> Path:
    p = tmp_path / "paths.toml"
    p.write_text('smplx = "/models/smplx"\n[roots]\nhodome = "/data/HODome"\n')
    return p


def _parse(argv):
    ap = argparse.ArgumentParser()
    add_scene_args(ap)
    return ap.parse_args(argv)


def test_defaults_filled_from_paths(tmp_path):
    a = _parse(["--dataset", "hodome", "--motion-path", "smplx/s01.npz"])
    spec = scene_from_args(a, paths_file=_toml(tmp_path))
    assert spec.smpl_model_dir == Path("/models/smplx")
    assert spec.motion_path == Path("/data/HODome/smplx/s01.npz")
    assert spec.dataset_root == Path("/data/HODome")


def test_explicit_args_override_paths(tmp_path):
    a = _parse(["--dataset", "hodome", "--motion-path", "/abs/seq.npz",
                "--model-dir", "/m", "--dataset-root", "/r"])
    spec = scene_from_args(a, paths_file=_toml(tmp_path))
    assert spec.motion_path == Path("/abs/seq.npz")
    assert spec.smpl_model_dir == Path("/m")
    assert spec.dataset_root == Path("/r")


def test_object_names_split(tmp_path):
    a = _parse(["--dataset", "hodome", "--motion-path", "/abs/seq.npz",
                "--model-dir", "/m", "--dataset-root", "/r", "--object-names", "box,case"])
    spec = scene_from_args(a, paths_file=_toml(tmp_path))
    assert spec.object_names == ("box", "case")


def test_robot_urdf_is_real_repo_model(tmp_path):
    a = _parse(["--dataset", "hodome", "--motion-path", "/abs/seq.npz",
                "--model-dir", "/m", "--dataset-root", "/r"])
    spec = scene_from_args(a, paths_file=_toml(tmp_path))
    assert spec.robot.urdf_path.name == "g1_29dof.urdf"
    assert spec.robot.urdf_path.is_absolute()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_scene_args.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.viz._scene_args'`).

- [ ] **Step 3: Write the glue**

Create `src/viz/_scene_args.py`:

```python
"""Shared CLI glue for the viz entry points: declare the common scene flags and assemble a
fully-resolved ``SceneSpec`` from them (filling defaults from the machine-local paths.toml).

EDGE-only: imported by viz ``main()`` functions, never by the pure pipeline. Keeps the four
viewers from each duplicating the RobotSpec/SceneSpec construction.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .. import paths
from ..prepare.contracts import RobotSpec, SceneSpec


def add_scene_args(ap: argparse.ArgumentParser) -> None:
    """Add the scene-selection flags shared by the viz CLIs."""
    ap.add_argument("--dataset", default="hodome")
    ap.add_argument("--motion-path", required=True, type=Path,
                    help="absolute, or relative to the dataset's [roots] entry in paths.toml")
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="SMPL-X model dir; default: paths.toml 'smplx'")
    ap.add_argument("--dataset-root", type=Path, default=None,
                    help="release root for object/betas metadata; default: paths.toml roots[dataset]")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--frame-step", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--person-id", type=int, default=None, help="multi-person: which person to retarget")
    ap.add_argument("--object-names", default=None, help="comma-separated subset of objects to load")


def scene_from_args(a: argparse.Namespace, *, paths_file: Path | None = None) -> SceneSpec:
    """Build a fully-resolved SceneSpec, filling missing paths from paths.toml.

    Explicit CLI args always win; paths.toml is read only when a default is needed (so fully
    explicit, absolute invocations work even without a paths.toml).
    """
    need_cfg = (a.model_dir is None) or (a.dataset_root is None) or (not Path(a.motion_path).is_absolute())
    cfg = paths.load_paths(paths_file) if need_cfg else {}

    model_dir = a.model_dir if a.model_dir is not None else paths.smplx_dir(cfg)
    motion = paths.resolve_motion(a.dataset, a.motion_path, cfg)
    droot = a.dataset_root
    if droot is None:
        try:
            droot = paths.dataset_root(a.dataset, cfg)
        except ValueError:
            droot = None   # datasets without a configured root (e.g. hoim3) keep dataset_root unset

    objs = tuple(a.object_names.split(",")) if a.object_names else None
    robot = RobotSpec(name="g1", urdf_path=paths.HOLOV2_ROOT / "models" / "g1" / "g1_29dof.urdf",
                      link_names=("pelvis",), dof=29, height=1.3)
    return SceneSpec(dataset=a.dataset, motion_path=motion, robot=robot,
                     smpl_model_dir=model_dir, dataset_root=droot,
                     person_id=a.person_id, object_names=objs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_scene_args.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/viz/_scene_args.py tests/test_scene_args.py
git commit -m "feat(holov2): _scene_args — shared CLI->SceneSpec glue (paths defaults, real URDF)"
```

---

### Task 4: Wire the viz entry points to the glue

**Files:**
- Modify: `src/viz/viewer.py` (`main()` ~362-380; import ~42)
- Modify: `src/viz/scene.py` (`main()` ~207-225; import ~23)
- Modify: `src/viz/cloud.py` (`main()` ~130-148; import ~23)
- Modify: `src/viz/hoim3_multiperson.py` (`main()` ~102-112)

**Interfaces:**
- Consumes: `add_scene_args`, `scene_from_args` (Task 3); for hoim3, `src.paths.smplx_dir`.
- Produces: no new symbols (CLI behavior only).

- [ ] **Step 1: Rewire `viewer.py`**

Change the import line `from ..prepare.contracts import RobotSpec, SceneSpec` to:

```python
from ..prepare.contracts import SceneSpec
```

Add after the existing relative imports:

```python
from ._scene_args import add_scene_args, scene_from_args
```

Replace the body of `main()` (the argparse block + RobotSpec/objs/SceneSpec construction) with:

```python
def main() -> None:
    ap = argparse.ArgumentParser()
    add_scene_args(ap)
    a = ap.parse_args()
    spec = scene_from_args(a)
    view_trace(spec, port=a.port, frame_step=a.frame_step, max_frames=a.max_frames)
```

- [ ] **Step 2: Rewire `scene.py`**

Change `from ..prepare.contracts import RobotSpec, SceneSpec` to:

```python
from ..prepare.contracts import SceneSpec
```

Add with the other relative imports:

```python
from ._scene_args import add_scene_args, scene_from_args
```

Replace the body of `main()` with:

```python
def main() -> None:
    ap = argparse.ArgumentParser()
    add_scene_args(ap)
    a = ap.parse_args()
    spec = scene_from_args(a)
    view_scene(spec, port=a.port, frame_step=a.frame_step, max_frames=a.max_frames)
```

- [ ] **Step 3: Rewire `cloud.py` (keep the extra `--corr` flag)**

Change `from ..prepare.contracts import RobotSpec, SceneSpec` to:

```python
from ..prepare.contracts import SceneSpec
```

Add with the other relative imports:

```python
from ._scene_args import add_scene_args, scene_from_args
```

Replace the body of `main()` with (note `--corr` is added after the shared args):

```python
def main() -> None:
    ap = argparse.ArgumentParser()
    add_scene_args(ap)
    ap.add_argument("--corr", type=Path, default=_DEFAULT_CORR, help="correspondence cache (.npz)")
    a = ap.parse_args()
    spec = scene_from_args(a)
    view_cloud(spec, a.corr, port=a.port, frame_step=a.frame_step, max_frames=a.max_frames)
```

- [ ] **Step 4: Rewire `hoim3_multiperson.py` (minimal: optional `--model-dir` from paths)**

This CLI has a different shape (no `--dataset`/objects); only default `--model-dir` from paths.toml.

Add an import near the top (with the other relative imports, e.g. after `from ..prepare.contracts import RobotSpec, SceneSpec`):

```python
from .. import paths
```

Replace the body of `main()` with:

```python
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion-path", required=True, type=Path)
    ap.add_argument("--model-dir", type=Path, default=None, help="SMPL-X model dir; default: paths.toml 'smplx'")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--frame-step", type=int, default=30)
    ap.add_argument("--max-frames", type=int, default=150)
    a = ap.parse_args()
    model_dir = a.model_dir if a.model_dir is not None else paths.smplx_dir()
    robot = RobotSpec(name="g1", urdf_path=paths.HOLOV2_ROOT / "models" / "g1" / "g1_29dof.urdf",
                      link_names=("pelvis",), dof=29, height=1.3)
    spec = SceneSpec(dataset="hoim3", motion_path=a.motion_path, robot=robot, smpl_model_dir=model_dir)
    view(spec, port=a.port, frame_step=a.frame_step, max_frames=a.max_frames)
```

- [ ] **Step 5: Verify all four CLIs parse and the package still imports**

Run:
```bash
$PY -c "import src.viz.viewer, src.viz.scene, src.viz.cloud, src.viz.hoim3_multiperson; print('imports OK')"
$PY -m src.viz.viewer --help
$PY -m src.viz.cloud --help
$PY -m src.viz.hoim3_multiperson --help
```
Expected: `imports OK`; each `--help` exits 0 and shows `--model-dir` with the "default: paths.toml 'smplx'" help text (and `cloud --help` still lists `--corr`).

- [ ] **Step 6: Verify the existing viz/contract tests still pass**

Run: `$PY -m pytest tests/test_paths.py tests/test_scene_args.py tests/test_viewer_bake.py -q`
Expected: PASS (no regressions). If `test_viewer_bake.py` needs data and is skipped/xfailed on this machine, that's acceptable — confirm it is not a NEW failure introduced here by checking it behaves the same as on `git stash`.

- [ ] **Step 7: Commit**

```bash
git add src/viz/viewer.py src/viz/scene.py src/viz/cloud.py src/viz/hoim3_multiperson.py
git commit -m "refactor(holov2): viz CLIs delegate to _scene_args (paths defaults, drop placeholder URDF)"
```

---

### Task 5: Update `CHEATSHEET.md`

**Files:**
- Modify: `CHEATSHEET.md`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing (docs).

- [ ] **Step 1: Add a paths.toml setup note to §0**

In `CHEATSHEET.md` §0, after the `export SMPLX=…` block, add:

```markdown
> 💡 Plutôt que `$DATA`/`$SMPLX` à la main, copie une fois `paths.example.toml` → `paths.toml`
> (gitignoré) et édite tes chemins. Les viewers liront `--model-dir`/`--dataset-root` depuis là,
> et `--motion-path` peut alors être **relatif** à la racine du dataset.

```bash
cp paths.example.toml paths.toml      # une fois, puis éditer les chemins de ta machine
```
```

- [ ] **Step 2: Add the short (paths.toml) command forms to §1**

In `CHEATSHEET.md` §1, just under the "Viewer principal" heading, add a short form before the absolute example:

```bash
# Forme courte (lit smplx + racine depuis paths.toml ; motion relative à la racine du dataset)
$PY -m src.viz.viewer --dataset hodome --motion-path smplx/subject01_baseball.npz --max-frames 30
$PY -m src.viz.viewer --dataset sfu    --motion-path 0005/0005_Jogging001_stageii.npz --max-frames 30
# OMOMO : motion en chemin ABSOLU (les .pt vivent dans OMOMO_new/, hors racine metadata)
$PY -m src.viz.viewer --dataset omomo \
    --motion-path $DATA/OMOMO_new/OMOMO_new/sub10_clothesstand_000.pt --max-frames 30
```

Keep the existing absolute examples below (they still work — `--model-dir`/`--dataset-root` are now optional overrides).

- [ ] **Step 3: Verify the cheatsheet renders coherently**

Run: `$PY -c "print(open('CHEATSHEET.md').read().count('paths.toml'))"`
Expected: a count ≥ 3 (setup note + §1 references), confirming the edits landed.

- [ ] **Step 4: Commit**

```bash
git add CHEATSHEET.md
git commit -m "docs(holov2): cheatsheet — paths.toml short command forms"
```

---

## Self-Review

**Spec coverage:**
- TOML/`tomllib`, zero dep → Task 1 (`src/paths.py`). ✓
- `paths.toml` gitignored + `paths.example.toml` committed → Task 2. ✓
- Resolver API (`load_paths`/`smplx_dir`/`dataset_root`/`resolve_motion`) → Task 1. ✓
- `_scene_args` (`add_scene_args`/`scene_from_args`), optional `--model-dir`/`--dataset-root`, relative motion, real URDF fix → Task 3 + Task 4. ✓
- Backward-compat (absolute args still work, even without paths.toml) → Task 3 `need_cfg` short-circuit + `test_explicit_args_override_paths`. ✓
- Tests `tests/test_paths.py` → Task 1; `_scene_args` covered by `tests/test_scene_args.py` → Task 3. ✓
- Edge-only / not in config.py → enforced by module placement (`src/paths.py`, `src/viz/_scene_args.py`); no `prepare`/`targets`/`config.py` edits in any task. ✓
- Cheatsheet update → Task 5. ✓
- Error handling (missing file/keys; missing root only errs for relative motion) → covered by `test_load_paths_missing_file`, `test_*_missing_key`, and the absolute-passthrough test. ✓

**Placeholder scan:** No "TBD"/"add error handling"/"similar to Task N" — every code step shows full code. The `/abs/path/to/…` strings in `paths.example.toml` are intentional template placeholders (the file IS a template), not plan gaps. ✓

**Type consistency:** `load_paths`/`smplx_dir`/`dataset_root`/`resolve_motion`, `add_scene_args`/`scene_from_args`, and `paths.HOLOV2_ROOT` are named identically across Tasks 1, 3, 4. `SceneSpec`/`RobotSpec` field names (`smpl_model_dir`, `dataset_root`, `person_id`, `object_names`, `urdf_path`, `link_names`, `dof`, `height`) match `src/prepare/contracts.py`. ✓
</content>
