# Modular Retargeting Visualization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make holosoma's retargeter modular enough to show several retargeted trajectories/stages in one viser session, by extracting rendering from compute and adding a stage registry — without changing what the SOCP solver computes.

**Architecture:** Three coupling locks are undone. (1) A `StageSpec` registry becomes the single source of truth for displayable stages. (2) A `Viewer` class is extracted from `InteractionMeshRetargeter`: it owns the viser server, a dict of robot instances (one per `produces_qpos` stage), object, grid and named keypoint layers. (3) `retarget_motion` returns a structured `RetargetResult` instead of drawing inline. We edit holosoma core where it serves modularity; the cvxpy/CLARABEL solve (`iterate`, `solve_single_iteration`) is behaviourally untouched and guarded by a golden qpos test.

**Discipline (applies to every core edit):** Editing holosoma files is expected and fine, but **each edit must be justified and focused** — no gratuitous churn, follow existing conventions, and keep the diff reviewable. Every task that changes `interaction_mesh_retargeter.py` states *why* the edit is necessary. The golden test (Task 0) proves the solver's output is identical after each refactor.

**Tech Stack:** Python, numpy, cvxpy/CLARABEL (untouched), mujoco, viser, yourdfpy, trimesh, pytest.

**Scope:** Spec increments 1 and 2 (modular foundation). `GMR-SOCP` (increment 3) and contact/OT overlays (increment 4) are follow-up plans; `Viewer.stage_keys` and `RetargetResult.stages` are the seams they plug into.

**Reference spec:** `docs/specs/2026-06-11-holosoma-modular-retargeting-viz-design.md`

**Path note:** Paths are relative to the HoloNew package root: `modules/01_retargeting/HoloNew/HoloNew/`. Run commands from there.

---

## File Structure

- Create: `src/stages.py` — `StageSpec` + `STAGE_SPECS` registry + helpers. Responsibility: "what can be displayed".
- Create: `src/retarget_result.py` — `RetargetResult` dataclass: structured output of a retarget.
- Create: `src/viewer.py` — `Viewer` class: the viser scene (robots, object, grid, keypoints, playback). Responsibility: rendering.
- Modify: `src/interaction_mesh_retargeter.py` — delegate drawing to `Viewer`; return `RetargetResult` from `retarget_motion`. **Justification:** these are the three coupling locks; they cannot be removed without editing the class that owns them.
- Modify: `examples/robot_retarget.py` — consume the returned `RetargetResult` (its `retarget_motion` call currently ignores the return value). **Justification:** caller must receive structured data to feed the viewer.
- Create: `examples/view_stages.py` — entry point opening the multi-stage viewer.
- Create: `tests/__init__.py`, `tests/conftest.py`, `tests/golden/baseline_qpos.npz`, `tests/test_stages.py`, `tests/test_viewer.py`, `tests/test_retarget_golden.py`.

---

## Task 0: Golden regression baseline

The safety net that lets us refactor core with confidence: freeze the current qpos output and re-compare after every change.

**Files:** Create `tests/golden/`, `tests/conftest.py`, `tests/__init__.py`

- [ ] **Step 1: Pick the smallest demo sequence**

Run: `ls demo_data/OMOMO_new && ls demo_data/climb`
This plan assumes `robot_only / sub3_largebox_003 / smplh / demo_data/OMOMO_new`; adjust consistently if a smaller sequence exists.

- [ ] **Step 2: Produce the baseline with current code**

Run:
```bash
python examples/robot_retarget.py \
  --data_path demo_data/OMOMO_new --task-type robot_only \
  --task-name sub3_largebox_003 --data_format smplh --save_dir tests/golden
```
Expected: `tests/golden/sub3_largebox_003.npz` with keys `qpos`, `human_joints`.

- [ ] **Step 3: Freeze + inspect**

Run:
```bash
cp tests/golden/sub3_largebox_003.npz tests/golden/baseline_qpos.npz
python -c "import numpy as np; d=np.load('tests/golden/baseline_qpos.npz'); print({k:d[k].shape for k in d.files})"
```
Expected: prints `qpos (T, M)` and `human_joints (T, J, 3)`.

- [ ] **Step 4: conftest fixtures**

```python
# tests/conftest.py
from pathlib import Path
import numpy as np
import pytest

HERE = Path(__file__).parent
PKG = HERE.parent

@pytest.fixture(scope="session")
def golden_qpos():
    return np.load(HERE / "golden" / "baseline_qpos.npz")["qpos"]

@pytest.fixture(scope="session")
def robot_urdf():
    return str(PKG / "models" / "g1" / "g1_29dof.urdf")
```

- [ ] **Step 5: Commit**

```bash
git add tests/__init__.py tests/conftest.py tests/golden/baseline_qpos.npz
git commit -m "test: freeze golden qpos baseline for retargeting refactor"
```

---

## Task 1: StageSpec registry

Pure data, zero dependencies — the lowest-risk first unit. Mirrors test_pipe's `stage_registry.py`.

**Files:** Create `src/stages.py`; Test `tests/test_stages.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stages.py
from HoloNew.src.stages import (
    STAGE_SPECS, stage_labels, spec_for_label, key_for_label, produces_qpos,
)

def test_registry_has_native_socp_stage():
    assert "SOCP" in stage_labels()
    assert stage_labels()[0] == "Original"

def test_socp_drives_robot():
    assert produces_qpos("SOCP") is True
    assert key_for_label("SOCP") == "socp"

def test_original_is_skeleton_only():
    assert produces_qpos("Original") is False
    assert key_for_label("Original") is None

def test_lookup_roundtrip():
    for s in STAGE_SPECS:
        assert spec_for_label(s.label).key == s.key
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_stages.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'HoloNew.src.stages'`

- [ ] **Step 3: Write the registry**

```python
# src/stages.py
"""Single source of truth for the viewer's retargeting stages.

Adding a future stage (e.g. GMR-SOCP) is one StageSpec entry here plus a
producer that fills its data. Dropdown, ghost overlay and robot-mesh gating all
derive from this registry.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageSpec:
    label: str           # dropdown label
    key: str | None      # key into RetargetResult.stages; None = raw human skeleton
    produces_qpos: bool  # True => drives a robot mesh under /world/robot_<key>


STAGE_SPECS: tuple[StageSpec, ...] = (
    StageSpec("Original", None,        False),
    StageSpec("Mapped",   "mapped",    False),
    StageSpec("InObject", "in_object", False),
    StageSpec("SOCP",     "socp",      True),
)

_BY_LABEL: dict[str, StageSpec] = {s.label: s for s in STAGE_SPECS}


def stage_labels() -> list[str]:
    return [s.label for s in STAGE_SPECS]


def spec_for_label(label: str) -> StageSpec:
    return _BY_LABEL[label]


def key_for_label(label: str) -> str | None:
    return _BY_LABEL[label].key


def produces_qpos(label: str) -> bool:
    return _BY_LABEL[label].produces_qpos
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_stages.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/stages.py tests/test_stages.py
git commit -m "feat: add StageSpec registry for displayable retargeting stages"
```

---

## Task 2: Extract the Viewer (behaviour-preserving)

**Justification for core edit:** `_setup_visualization`, `draw_q`, `draw_keypoints` hard-wire the retargeter to a single viser robot — Lock 2. We move them into a `Viewer` and leave the retargeter's methods as thin delegating shims, so the many existing call sites are unchanged and the diff stays focused.

**Files:**
- Create: `src/viewer.py`
- Modify: `src/interaction_mesh_retargeter.py` (`_setup_visualization` ~243-312; `draw_q` ~890-915; `draw_keypoints` ~917-957)
- Test: `tests/test_viewer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_viewer.py
import numpy as np
from HoloNew.src.viewer import Viewer

def test_viewer_creates_named_robot_root(robot_urdf):
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None)
    assert "socp" in v.robots
    v.close()

def test_draw_q_sets_base_pose(robot_urdf):
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None)
    dof = v.robots["socp"].dof
    q = np.zeros(7 + dof)
    q[:3] = [1.0, 2.0, 3.0]
    q[3:7] = [1.0, 0.0, 0.0, 0.0]
    v.draw_q(q, stage="socp")
    np.testing.assert_allclose(v.robots["socp"].base.position, [1.0, 2.0, 3.0])
    v.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_viewer.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'HoloNew.src.viewer'`

- [ ] **Step 3: Write the Viewer** (bodies adapted verbatim from the current `draw_q`/`draw_keypoints`/`_setup_visualization`, preserving the qpos layout)

```python
# src/viewer.py
"""Owns the viser scene: robot instance(s), object, grid, keypoint layers.

Extracted from InteractionMeshRetargeter so rendering is separate from compute
and several trajectories can share one viser session. qpos layout matches
holosoma: [0:3] pos, [3:7] wxyz quat, [7:7+dof] actuated joints, optional
trailing [-7:] dynamic-object pose.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
import viser
from viser.extras import ViserUrdf
import yourdfpy

from .stages import STAGE_SPECS, spec_for_label, stage_labels


@dataclass
class RobotHandle:
    urdf: ViserUrdf
    base: object   # viser frame handle
    dof: int


class Viewer:
    def __init__(self, robot_model_path: str, object_model_path: str | None,
                 stage_keys: tuple[str, ...] = ("socp",),
                 has_dynamic_object: bool = False) -> None:
        self.robot_model_path = robot_model_path
        self.object_model_path = object_model_path
        self.has_dynamic_object = has_dynamic_object
        self.server = viser.ViserServer()
        self.server.scene.set_up_direction("+z")
        try:
            self.server.scene.add_frame("/world", show_axes=False)
        except Exception:
            pass

        self.robots: dict[str, RobotHandle] = {k: self._add_robot(k) for k in stage_keys}

        self.object_base = None
        self.viser_object = None
        if object_model_path:
            self.object_base = self.server.scene.add_frame("/world/object", show_axes=False)
            obj = yourdfpy.URDF.load(object_model_path, load_meshes=True, build_scene_graph=True)
            self.viser_object = ViserUrdf(self.server, urdf_or_path=obj, root_node_name="/world/object")

        self.server.scene.add_grid("/world/grid", width=8, height=8, position=(0.0, 0.0, 0.0))
        self._result = None

    def _add_robot(self, key: str) -> RobotHandle:
        root = f"/world/robot_{key}"
        base = self.server.scene.add_frame(root, show_axes=False)
        urdf = yourdfpy.URDF.load(self.robot_model_path, load_meshes=True, build_scene_graph=True)
        vu = ViserUrdf(self.server, urdf_or_path=urdf, root_node_name=root)
        dof = len(vu.get_actuated_joint_limits())
        vu.update_cfg(np.zeros(dof))
        return RobotHandle(urdf=vu, base=base, dof=dof)

    def draw_q(self, q: np.ndarray, stage: str = "socp") -> None:
        h = self.robots[stage]
        h.urdf.update_cfg(q[7:7 + h.dof])
        h.base.position = q[:3]
        h.base.wxyz = q[3:7]
        if self.viser_object is not None and self.object_base is not None:
            if self.has_dynamic_object:
                self.object_base.position = q[-7:-4]
                self.object_base.wxyz = q[-4:]
            else:
                self.object_base.position = np.zeros(3)
                self.object_base.wxyz = np.asarray([1.0, 0.0, 0.0, 0.0])

    def draw_keypoints(self, p: np.ndarray, name: str = "keypoint", rgba=(0, 0, 1, 1)):
        sphere = trimesh.primitives.Sphere(radius=0.02)
        color = tuple(int(c * 255) for c in rgba[:3])
        opacity = float(rgba[3])
        if p.ndim == 1:
            return self.server.scene.add_mesh_simple(
                f"/{name}", vertices=sphere.vertices, faces=sphere.faces,
                position=p, color=color, opacity=opacity)
        return self.server.scene.add_batched_meshes_simple(
            f"/{name}", vertices=sphere.vertices, faces=sphere.faces,
            batched_positions=p,
            batched_wxyzs=np.tile(np.array([1, 0, 0, 0]), (p.shape[0], 1)),
            batched_colors=color, opacity=opacity)

    def close(self) -> None:
        self.server.stop()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_viewer.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Delegate from the retargeter (focused edit, shims keep call sites stable)**

In `src/interaction_mesh_retargeter.py`, replace the body of `_setup_visualization` so it builds a `Viewer` and re-exposes the attributes the rest of the class already reads (`self.server`, `self.viser_robot`, `self.robot_base`, `self.viser_object`, `self.object_base`):

```python
    def _setup_visualization(self):
        """Setup viser via the extracted Viewer (single robot, unchanged behaviour)."""
        from .viewer import Viewer
        self.viewer = Viewer(
            robot_model_path=self.robot_model_path,
            object_model_path=self.object_model_path,
            stage_keys=("socp",),
            has_dynamic_object=self.has_dynamic_object,
        )
        self.server = self.viewer.server
        self.viser_robot = self.viewer.robots["socp"].urdf
        self.robot_base = self.viewer.robots["socp"].base
        self.viser_object = self.viewer.viser_object
        self.object_base = self.viewer.object_base
```

Replace `draw_q` and `draw_keypoints` with delegations:

```python
    def draw_q(self, q: np.ndarray):
        self.viewer.draw_q(q, stage="socp")

    def draw_keypoints(self, p, name="keypoint", rgba=(0, 0, 1, 1)):
        if not hasattr(self, "viewer"):
            return None
        return self.viewer.draw_keypoints(p, name=name, rgba=rgba)
```

Leave every other method (collision-debug draws at ~882/1021, `visualize_motion`, `visualize_tetrahedra`) as-is; they still use `self.server`, which now points at the Viewer's server.

- [ ] **Step 6: Verify the golden output is unchanged**

Run:
```bash
python examples/robot_retarget.py \
  --data_path demo_data/OMOMO_new --task-type robot_only \
  --task-name sub3_largebox_003 --data_format smplh --save_dir tests/_tmp
python -c "import numpy as np, numpy.testing as t; a=np.load('tests/golden/baseline_qpos.npz')['qpos']; b=np.load('tests/_tmp/sub3_largebox_003.npz')['qpos']; t.assert_allclose(a,b,atol=1e-9); print('GOLDEN OK', a.shape)"
```
Expected: `GOLDEN OK (T, M)` — solver behaviour identical.

- [ ] **Step 7: Commit**

```bash
git add src/viewer.py src/interaction_mesh_retargeter.py tests/test_viewer.py
git commit -m "refactor: extract Viewer from retargeter behind delegating shims"
```

---

## Task 3: `retarget_motion` returns a structured result

**Justification for core edit:** Lock 1 — `retarget_motion` interleaves drawing and saving with compute, so its results can't feed a separate viewer. We make it *return* a `RetargetResult` (it currently returns a loose 4-tuple that the example ignores) while keeping the inline `np.savez` for backward compatibility. Drawing stays exactly as-is (already routed through the Viewer via Task 2's shims).

**Files:**
- Create: `src/retarget_result.py`
- Modify: `src/interaction_mesh_retargeter.py` (`retarget_motion` ~367-551: collect stage lists; change the return)
- Modify: `examples/robot_retarget.py` (~707: capture the return)
- Test: `tests/test_retarget_golden.py`

- [ ] **Step 1: Write the result dataclass**

```python
# src/retarget_result.py
"""Structured output of retarget_motion: qpos trajectory + per-frame stage data.

Keys in `stages` match StageSpec.key in src/stages.py. 'socp' is the final robot
qpos sequence; skeleton stages ('mapped', 'in_object') hold (T, J, 3) points.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RetargetResult:
    qpos: np.ndarray                                            # (T, 7+dof[+7])
    stages: dict[str, np.ndarray] = field(default_factory=dict)
    cost: float = 0.0
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_retarget_golden.py
import numpy as np, numpy.testing as npt
from HoloNew.src.retarget_result import RetargetResult

def test_retarget_returns_result_matching_golden(golden_qpos):
    from HoloNew.examples.robot_retarget import run_headless  # added in Step 4
    result = run_headless(
        data_path="demo_data/OMOMO_new", task_type="robot_only",
        task_name="sub3_largebox_003", data_format="smplh",
    )
    assert isinstance(result, RetargetResult)
    npt.assert_allclose(result.qpos, golden_qpos, atol=1e-9)
    assert "mapped" in result.stages and "in_object" in result.stages
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_retarget_golden.py -v`
Expected: FAIL (`ImportError: cannot import name 'run_headless'`).

- [ ] **Step 4: Collect stage data + return RetargetResult**

In `retarget_motion`, before the frame loop (near line 407) add:

```python
        mapped_list = []
        in_object_list = []
```

Inside the loop, right after `human_mapped_joints_in_object` is computed (near line 427):

```python
                mapped_list.append(human_mapped_joints)
                in_object_list.append(human_mapped_joints_in_object)
```

Replace the final `return (...)` (lines ~546-551) with:

```python
        from .retarget_result import RetargetResult
        qpos = np.array(retargeted_motions)[1:]
        if dest_res_path is not None:
            np.savez(dest_res_path, qpos=qpos, human_joints=human_joint_motions, fps=30, cost=cost)
        return RetargetResult(
            qpos=qpos,
            stages={"mapped": np.stack(mapped_list), "in_object": np.stack(in_object_list)},
            cost=float(cost),
        )
```

(The original method already built `np.savez` and returned a tuple; we are changing the return type and adding two cheap lists. No other behaviour changes.)

Add a thin headless helper to `examples/robot_retarget.py` that runs the existing `main` wiring with viz disabled and returns the result. The simplest non-invasive form reuses `main`'s body by extracting the part from "Create retargeter" to the `retarget_motion` call into `run_headless(**overrides)` and having both `main` and the test call it. Keep `main`'s external behaviour identical.

At the existing call site (~707), capture the return:

```python
    result = retargeter.retarget_motion(
        human_joint_motions=human_joints,
        object_poses=object_poses,
        object_poses_augmented=object_poses_augmented,
        object_points_local_demo=object_local_pts_demo,
        object_points_local=object_local_pts,
        foot_sticking_sequences=foot_sticking_sequences,
        q_a_init=q_init,
        q_nominal_list=q_nominal,
        original=not cfg.augmentation,
        dest_res_path=dest_res_path,
    )
    return result
```

- [ ] **Step 5: Run the golden test**

Run: `python -m pytest tests/test_retarget_golden.py -v`
Expected: PASS — `result.qpos` equals the baseline within 1e-9.

- [ ] **Step 6: Commit**

```bash
git add src/retarget_result.py src/interaction_mesh_retargeter.py \
        examples/robot_retarget.py tests/test_retarget_golden.py
git commit -m "refactor: retarget_motion returns RetargetResult with stage data"
```

---

## Task 4: Multi-robot viewer + entry point

Add registry-driven playback to the Viewer (single slider, stage dropdown) and an entry point that runs a retarget and opens it. One robot per `produces_qpos` stage; today that is just `socp`, with skeleton stages selectable — ready for `GMR-SOCP` to drop in as a second robot.

**Files:**
- Modify: `src/viewer.py` (add `bind` + `_redraw`)
- Create: `examples/view_stages.py`
- Test: `tests/test_viewer.py` (append multi-stage construction test)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_viewer.py  (append)
from HoloNew.src.stages import STAGE_SPECS

def test_builds_one_robot_per_qpos_stage(robot_urdf):
    keys = tuple(s.key for s in STAGE_SPECS if s.produces_qpos)
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None, stage_keys=keys)
    assert set(v.robots) == set(keys)
    v.close()
```

- [ ] **Step 2: Run test to verify it fails (or passes construction)**

Run: `python -m pytest tests/test_viewer.py::test_builds_one_robot_per_qpos_stage -v`
Expected: PASS if Task 2 made `stage_keys` variadic (it did). This locks the seam before adding playback.

- [ ] **Step 3: Add registry-driven playback to the Viewer**

```python
# src/viewer.py  (add methods to Viewer)
    def bind(self, result) -> None:
        """Attach a RetargetResult and build slider + stage dropdown from the registry."""
        self._result = result
        with self.server.gui.add_folder("Playback"):
            self._slider = self.server.gui.add_slider(
                "Frame", min=0, max=max(0, result.qpos.shape[0] - 1), step=1, initial_value=0)
        with self.server.gui.add_folder("Display"):
            self._stage_dd = self.server.gui.add_dropdown(
                "Stage", options=stage_labels(), initial_value="SOCP")

        @self._slider.on_update
        def _(_evt): self._redraw(int(self._slider.value))

        @self._stage_dd.on_update
        def _(_evt): self._redraw(int(self._slider.value))

        self._redraw(0)

    def _redraw(self, frame: int) -> None:
        spec = spec_for_label(self._stage_dd.value)
        if spec.produces_qpos:
            self.draw_q(self._result.qpos[frame], stage=spec.key)
        elif spec.key is None:
            # 'Original' has no stored array here; skip unless a raw skeleton is bound.
            return
        elif spec.key in self._result.stages:
            self.draw_keypoints(self._result.stages[spec.key][frame], name=f"stage_{spec.key}")
```

- [ ] **Step 4: Create the entry point**

```python
# examples/view_stages.py
"""Run a retarget, then open the multi-stage viewer."""
from __future__ import annotations

import tyro

from HoloNew.examples.robot_retarget import RetargetingConfig, run_headless
from HoloNew.src.stages import STAGE_SPECS
from HoloNew.src.viewer import Viewer


def view(cfg: RetargetingConfig) -> None:
    cfg.retargeter.visualize = False   # only our viewer opens
    cfg.retargeter.debug = False
    result = run_headless(cfg=cfg)

    qpos_keys = tuple(s.key for s in STAGE_SPECS if s.produces_qpos)
    viewer = Viewer(
        robot_model_path=cfg.robot_config.robot_urdf_file,
        object_model_path=None,
        stage_keys=qpos_keys,
    )
    viewer.bind(result)
    input("Viewer at http://localhost:8080 — Enter to exit ...")


if __name__ == "__main__":
    view(tyro.cli(RetargetingConfig))
```

> Execution note: confirm `run_headless`'s exact signature (Task 3 Step 4) and the `cfg.robot_config.robot_urdf_file` field name against the real config (grep `robot_urdf_file`). Adjust if the config nests it differently.

- [ ] **Step 5: Run tests + manual smoke**

Run: `python -m pytest tests/ -v`
Expected: PASS (stages, viewer, retarget golden).
Manual:
```bash
python examples/view_stages.py --data_path demo_data/OMOMO_new \
  --task-type robot_only --task-name sub3_largebox_003 --data_format smplh
```
Expected: viser opens at `http://localhost:8080`; Frame slider scrubs the robot; Stage dropdown switches `SOCP` (robot) vs `Mapped`/`InObject` (skeletons).

- [ ] **Step 6: Commit**

```bash
git add src/viewer.py examples/view_stages.py tests/test_viewer.py
git commit -m "feat: registry-driven multi-stage viewer + view_stages entry point"
```

---

## Self-Review notes

- **Spec coverage:** Lock 1 → Task 3; Lock 2 → Tasks 2 & 4; Lock 3 → Task 1. Increment 1 → Tasks 2-3; Increment 2 → Tasks 1 & 4.
- **Core edits justified:** Task 2 (extract Viewer — Lock 2), Task 3 (structured return — Lock 1) each state why; both are guarded by the golden test (solver output identical). Drawing logic is moved verbatim, not rewritten.
- **Solver untouched:** No task edits `iterate`/`solve_single_iteration`/cvxpy.
- **Deferred seams:** `GMR-SOCP` = one `StageSpec(produces_qpos=True)` + a second qpos array + its key in `Viewer.stage_keys`. Overlays (contact/OT) = a separate layer; follow-up plan.
- **Open items for execution:** exact `run_headless` extraction in `robot_retarget.py` (preserve preprocessing order — golden test guards it); the `robot_urdf_file` config field name; `viser` GUI/handle API names for the installed version; whether to bind a raw-human array for the `Original` stage.
```
