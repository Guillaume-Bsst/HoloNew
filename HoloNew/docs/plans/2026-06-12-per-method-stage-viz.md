# Per-method Stage Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In `view_stages`, add Method + Stage dropdowns so the user can scrub each method's (holosoma / GMR v1 / GMR v2) source and every preprocessing step through to the solved robot.

**Architecture:** A per-method registry replaces the flat `STAGE_SPECS`. Each method is delivered to the viewer as a `MethodViz` bundle (its `qpos` + ordered `stages` skeleton arrays). holosoma stages are reproduced by a new `compute_holosoma_stages` (native qpos untouched); GMR stages are the full `compute_stages` dict stored on each GMR retargeter. The viewer gains `bind_methods` with Method + Stage dropdowns and draws exactly the selected (method, stage, frame).

**Tech Stack:** Python, numpy, viser, pytest. Runs in the `holonew` conda env.

**Reference spec:** `docs/specs/2026-06-12-per-method-stage-viz-design.md`

## Critical environment (every task)
- Package dir (cwd): `/home/gbesset/Documents/wbt_rl/modules/01_retargeting/HoloNew/HoloNew`.
- Package imports as `HoloNew`. Work on a feature branch off `main`.
- **Always use this Python:** `/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python` (never bare python). Run pytest from the package dir.
- Commits: identity `Guillaume-Bsst`; **never add Co-Authored-By/Claude or any Claude mention**; comments/docs in **English**.
- Guard: the existing parity/golden suite must stay green (native qpos untouched). Full suite: `/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/ -q`.

## File Structure
- Modify: `src/stages.py` — replace flat `STAGE_SPECS` with a per-method `MethodSpec` registry.
- Modify: `src/holosoma/preprocess.py` — add `compute_holosoma_stages(...)`.
- Modify: `src/gmr_socp_v1/gmr_socp_v1.py`, `src/gmr_socp_v2/gmr_socp_v2.py` — store `rt.gmr_stages` (full `compute_stages` dict) in `from_config`.
- Modify: `src/viewer.py` — add `MethodViz` dataclass + `bind_methods` + Method/Stage `_redraw`.
- Modify: `examples/view_stages.py` — build 3 `MethodViz`, call `bind_methods`.
- Modify: `tests/test_stages.py`, `tests/test_viewer.py`; Create `tests/test_holosoma_stages.py`, `tests/test_gmr_stages_exposed.py`.

---

## Task 1: Per-method registry in `src/stages.py`

**Files:** Modify `src/stages.py`; Modify `tests/test_stages.py`.

- [ ] **Step 1: Rewrite the failing test `tests/test_stages.py`** (replace its contents)
```python
from HoloNew.src.stages import (
    METHODS, ROBOT_STAGE, method_labels, robot_key_for_method, stages_for_method,
)

def test_method_labels():
    assert method_labels() == ["holosoma", "GMR-SOCP v1", "GMR-SOCP v2"]

def test_robot_keys():
    assert robot_key_for_method("holosoma") == "holosoma"
    assert robot_key_for_method("GMR-SOCP v1") == "gmr_socp_v1"
    assert robot_key_for_method("GMR-SOCP v2") == "gmr_socp_v2"

def test_stage_lists_end_with_robot():
    hs = stages_for_method("holosoma")
    assert hs == ["Original", "Grounded", "Scaled", "Mapped", ROBOT_STAGE]
    g1 = stages_for_method("GMR-SOCP v1")
    assert g1 == ["Original", "Mapped", "Scaled", "Offset", "Ground", ROBOT_STAGE]
    assert stages_for_method("GMR-SOCP v2") == g1
```

- [ ] **Step 2: Run it, confirm failure** (`ImportError`).

- [ ] **Step 3: Rewrite `src/stages.py`**
```python
"""Per-method registry for the annex stage viewer.

Each method declares its robot key and the ordered skeleton stages of its
pipeline; the implicit final "Robot" stage drives the solved robot mesh.
"""
from __future__ import annotations

from dataclasses import dataclass

ROBOT_STAGE = "Robot"


@dataclass(frozen=True)
class MethodSpec:
    label: str                     # dropdown label
    robot_key: str                 # robot instance key, /world/robot_<robot_key>
    skeleton_stages: tuple[str, ...]  # ordered preprocessing stages (skeletons)


METHODS: tuple[MethodSpec, ...] = (
    MethodSpec("holosoma", "holosoma", ("Original", "Grounded", "Scaled", "Mapped")),
    MethodSpec("GMR-SOCP v1", "gmr_socp_v1", ("Original", "Mapped", "Scaled", "Offset", "Ground")),
    MethodSpec("GMR-SOCP v2", "gmr_socp_v2", ("Original", "Mapped", "Scaled", "Offset", "Ground")),
)

_BY_LABEL = {m.label: m for m in METHODS}


def method_labels() -> list[str]:
    return [m.label for m in METHODS]


def method_for_label(label: str) -> MethodSpec:
    return _BY_LABEL[label]


def robot_key_for_method(label: str) -> str:
    return _BY_LABEL[label].robot_key


def stages_for_method(label: str) -> list[str]:
    """Ordered stage labels for a method: its skeleton stages + the Robot stage."""
    return list(_BY_LABEL[label].skeleton_stages) + [ROBOT_STAGE]
```

- [ ] **Step 4: Run the test, confirm PASS (3).** Then check what else imported the old API:
```bash
git grep -n "STAGE_SPECS\|stage_labels\|spec_for_label\|produces_qpos\|key_for_label" -- src examples tests
```
The only non-test consumers are `src/viewer.py` and `examples/view_stages.py` — they are rewritten in Tasks 4-5. Leave them for now (the suite will be red until Task 5; that is expected for this coupled UI change — note it and proceed). If `src/contact`/`src/correspondence` reference these, they should not; report if they do.

- [ ] **Step 5: Commit**
```bash
git add src/stages.py tests/test_stages.py
git commit -m "feat(viz): per-method stage registry"
```

---

## Task 2: `compute_holosoma_stages` in `src/holosoma/preprocess.py`

Reproduces holosoma's preprocessing (ground then scale) as skeleton arrays, WITHOUT mutating inputs or touching the retargeter.

**Files:** Modify `src/holosoma/preprocess.py`; Create `tests/test_holosoma_stages.py`.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_holosoma_stages.py
import numpy as np
from HoloNew.src.holosoma.preprocess import compute_holosoma_stages

def test_holosoma_stages_shapes_and_steps():
    T, J = 4, 52
    raw = np.zeros((T, J, 3), float)
    raw[:, :, 2] = 1.0                 # all joints at z=1
    raw[:, 3, 2] = 0.5; raw[:, 7, 2] = 0.6   # toes (indices 3,7) lowest
    out = compute_holosoma_stages(raw, scale=0.5, toe_indices=[3, 7],
                                  mapped_indices=[0, 1, 2, 3, 4])
    assert set(out) == {"Original", "Grounded", "Scaled", "Mapped"}
    assert out["Original"].shape == (T, J, 3)
    # Grounded: lowest toe (z=0.5) moved to 0
    np.testing.assert_allclose(out["Grounded"][:, 3, 2], 0.0, atol=1e-6)
    # Scaled = grounded * 0.5
    np.testing.assert_allclose(out["Scaled"], out["Grounded"] * 0.5)
    # Mapped = scaled at the 5 mapped indices
    assert out["Mapped"].shape == (T, 5, 3)
    np.testing.assert_allclose(out["Mapped"], out["Scaled"][:, [0, 1, 2, 3, 4]])
```
(`mat_height` defaults to 0; with z_min=0.5 >= 0 no mat adjustment. Match holosoma's
`preprocess_motion_data`: it subtracts `mat_height` from z_min only if `z_min >= mat_height`.)

- [ ] **Step 2: Run it, confirm failure** (`ImportError`).

- [ ] **Step 3: Implement `compute_holosoma_stages` in `src/holosoma/preprocess.py`** (append):
```python
def compute_holosoma_stages(raw_joints, scale, toe_indices, mapped_indices, mat_height=0.1):
    """Holosoma preprocessing as per-stage skeleton arrays (no mutation).

    Mirrors preprocess_motion_data: ground (drop lowest toe z to 0, less mat_height if
    on a mat) then scale (multiply all joints), then select the mapped joints. Returns
    {Original (T,52,3), Grounded (T,52,3), Scaled (T,52,3), Mapped (T,len(mapped),3)}.
    """
    import numpy as np
    raw = np.asarray(raw_joints, dtype=float)
    original = raw.copy()

    grounded = raw.copy()
    z_min = float(grounded[:, toe_indices, 2].min())
    if z_min >= mat_height:
        z_min -= mat_height
    grounded[:, :, 2] -= z_min

    scaled = grounded * float(scale)
    mapped = scaled[:, list(mapped_indices)]
    return {"Original": original, "Grounded": grounded, "Scaled": scaled, "Mapped": mapped}
```

- [ ] **Step 4: Run the test, confirm PASS.**

- [ ] **Step 5: Commit**
```bash
git add src/holosoma/preprocess.py tests/test_holosoma_stages.py
git commit -m "feat(viz): holosoma preprocessing stage producer"
```

---

## Task 3: Expose the full GMR stage dict on the retargeters

**Files:** Modify `src/gmr_socp_v1/gmr_socp_v1.py`, `src/gmr_socp_v2/gmr_socp_v2.py`; Create `tests/test_gmr_stages_exposed.py`.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_gmr_stages_exposed.py
import pytest
from HoloNew.examples.robot_retarget import RetargetingConfig

@pytest.mark.parametrize("modpath,cls", [
    ("HoloNew.src.gmr_socp_v1.gmr_socp_v1", "GmrSocpRetargeterV1"),
    ("HoloNew.src.gmr_socp_v2.gmr_socp_v2", "GmrSocpRetargeterV2"),
])
def test_gmr_exposes_full_stage_dict(modpath, cls):
    import importlib
    Cls = getattr(importlib.import_module(modpath), cls)
    rt = Cls.from_config(RetargetingConfig(task_type="robot_only",
                                           task_name="sub3_largebox_003", data_format="smplh"))
    assert set(rt.gmr_stages) == {"mapped", "scaled", "offset", "ground"}
    for k in rt.gmr_stages:
        assert rt.gmr_stages[k]["pos"].shape[1] == 14   # 14 mapped bodies
    # the solve still uses the ground stage
    assert rt.gmr_ground is rt.gmr_stages["ground"]
```

- [ ] **Step 2: Run it, confirm failure** (`AttributeError: gmr_stages`).

- [ ] **Step 3: In `gmr_socp_v1.py` `from_config`**, change the line that computes the ground stage:
```python
        rt.gmr_stages = compute_stages(raw_joints, human_quat, anchor_root_xy=True)
        rt.gmr_ground = rt.gmr_stages["ground"]
```
(Replace the existing `ground = compute_stages(...)["ground"]; rt.gmr_ground = ground`. Keep the base-init using `ground["pos"][0, _pelvis_bi]` — read it from `rt.gmr_ground`.) Add `self.gmr_stages = None` next to `self.gmr_ground = None` if the latter is initialised in `__init__` (else just set it in from_config).

- [ ] **Step 4: Apply the identical change to `gmr_socp_v2.py` `from_config`.**

- [ ] **Step 5: Run the test + the GMR parity/integration tests**
```bash
/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_gmr_stages_exposed.py tests/test_gmr_socp.py tests/test_parity_gmr_socp_vs_mink.py -q
```
Expected: PASS (qpos unchanged — only an extra attribute is stored).

- [ ] **Step 6: Commit**
```bash
git add src/gmr_socp_v1/gmr_socp_v1.py src/gmr_socp_v2/gmr_socp_v2.py tests/test_gmr_stages_exposed.py
git commit -m "feat(viz): expose full compute_stages dict on GMR retargeters"
```

---

## Task 4: Viewer `MethodViz` + `bind_methods` (Method + Stage)

**Files:** Modify `src/viewer.py`; Modify `tests/test_viewer.py`.

- [ ] **Step 1: Append the failing test to `tests/test_viewer.py`**
```python
def test_bind_methods_builds_method_and_stage(robot_urdf):
    import numpy as np
    from HoloNew.src.viewer import Viewer, MethodViz
    m = MethodViz(label="GMR-SOCP v1", robot_key="gmr_socp_v1",
                  qpos=np.zeros((3, 36)),
                  stages={"Original": np.zeros((3, 5, 3)), "Mapped": np.zeros((3, 5, 3))})
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None, stage_keys=("gmr_socp_v1",))
    v.bind_methods([m])
    assert v._method_dd.value == "GMR-SOCP v1"
    # selecting a skeleton stage and the Robot stage both redraw without error
    v._method_dd.value = "GMR-SOCP v1"; v._stage_dd.value = "Mapped"; v._redraw(0)
    v._stage_dd.value = "Robot"; v._redraw(0)
    v.close()
```

- [ ] **Step 2: Run it, confirm failure** (`ImportError: MethodViz` / no `bind_methods`).

- [ ] **Step 3: Add `MethodViz` + `bind_methods` + new `_redraw` to `src/viewer.py`.**
At the top add:
```python
from dataclasses import dataclass, field

from .stages import ROBOT_STAGE, method_labels, robot_key_for_method, stages_for_method


@dataclass
class MethodViz:
    label: str
    robot_key: str
    qpos: "np.ndarray"
    stages: dict = field(default_factory=dict)   # {stage_label: (T, B, 3)}
```
Add to `Viewer`:
```python
    def bind_methods(self, methods: list) -> None:
        """Bind a list of MethodViz and build Frame + Method + Stage dropdowns."""
        self._methods = {m.label: m for m in methods}
        T = min(len(m.qpos) for m in methods)

        with self.server.gui.add_folder("Playback"):
            self._slider = self.server.gui.add_slider("Frame", min=0, max=max(0, T - 1),
                                                      step=1, initial_value=0)
        with self.server.gui.add_folder("Display"):
            first = methods[0].label
            self._method_dd = self.server.gui.add_dropdown("Method", options=method_labels(),
                                                           initial_value=first)
            self._stage_dd = self.server.gui.add_dropdown("Stage", options=stages_for_method(first),
                                                          initial_value=ROBOT_STAGE)

        @self._method_dd.on_update
        def _(_evt):
            self._stage_dd.options = stages_for_method(self._method_dd.value)
            self._stage_dd.value = ROBOT_STAGE
            self._redraw(int(self._slider.value))

        @self._slider.on_update
        def _(_evt): self._redraw(int(self._slider.value))

        @self._stage_dd.on_update
        def _(_evt): self._redraw(int(self._slider.value))

        self._redraw(0)

    def _hide_all_robots(self) -> None:
        for h in self.robots.values():
            h.urdf.show_visual = False

    def _redraw(self, frame: int) -> None:
        method = self._methods[self._method_dd.value]
        stage = self._stage_dd.value
        # clear the skeleton layer each redraw
        self.draw_keypoints(np.zeros((0, 3), np.float32), name="stage_skeleton")
        if stage == ROBOT_STAGE:
            self._hide_all_robots()
            self.robots[method.robot_key].urdf.show_visual = True
            self.draw_q(method.qpos[frame], stage=method.robot_key)
        else:
            self._hide_all_robots()
            pts = method.stages[stage][frame]
            self.draw_keypoints(pts.astype(np.float32), name="stage_skeleton",
                                rgba=(1.0, 0.4, 0.0, 1.0))
```
Note: confirm `self._stage_dd.options = [...]` is settable in the installed viser (1.0.30 `GuiDropdownHandle` supports it). If not, recreate the dropdown handle. Confirm `ViserUrdf` has `.show_visual` (the existing `examples/robot_retarget.py` uses `viser_robot.show_visual` — it does). `draw_keypoints` with an empty `(0,3)` array must be a safe no-op clear — if `add_batched_meshes_simple` errors on 0 points, guard `_redraw` to only clear when there were prior points, or use a `point_size`-0 cloud; pick the simplest that runs.

- [ ] **Step 4: Run the viewer tests, confirm PASS.** Keep any existing `bind`-based viewer tests passing OR update them — if `bind`/`extra_qpos` are now unused everywhere (Task 5 switches view_stages to `bind_methods`), you may remove the old `bind`/`_redraw`-via-registry and its test; but do that in Task 5 once view_stages is switched. For Task 4, ADD the new API alongside the old so the suite for this task is green.

- [ ] **Step 5: Commit**
```bash
git add src/viewer.py tests/test_viewer.py
git commit -m "feat(viz): viewer Method+Stage dropdowns via bind_methods"
```

---

## Task 5: Wire `examples/view_stages.py` to build 3 MethodViz

**Files:** Modify `examples/view_stages.py`; remove the now-dead old `bind`/registry path in `src/viewer.py` + `tests`.

- [ ] **Step 1: Rewrite `examples/view_stages.py`'s `view(cfg)`**
```python
def view(cfg) -> None:
    import numpy as np
    from HoloNew.examples.robot_retarget import (
        run_headless, load_motion_data, create_task_constants,
    )
    from HoloNew.src.holosoma.preprocess import compute_holosoma_stages
    from HoloNew.src.gmr_socp_v1.gmr_socp_v1 import GmrSocpRetargeterV1
    from HoloNew.src.gmr_socp_v2.gmr_socp_v2 import GmrSocpRetargeterV2
    from HoloNew.src.viewer import Viewer, MethodViz
    from HoloNew.src.stages import METHODS

    # holosoma: native qpos + reproduced preprocessing stages
    native = run_headless(cfg)
    data_format = cfg.data_format or "smplh"
    constants = create_task_constants(robot_config=cfg.robot_config,
                                      motion_data_config=cfg.motion_data_config,
                                      task_config=cfg.task_config, task_type=cfg.task_type)
    raw_joints, _obj, smpl_scale = load_motion_data(cfg.task_type, data_format, cfg.data_path,
                                                    cfg.task_name, constants, cfg.motion_data_config)
    toe = [constants.DEMO_JOINTS.index(n) for n in cfg.motion_data_config.toe_names]
    mapped = [constants.DEMO_JOINTS.index(n) for n in constants.JOINTS_MAPPING]
    hs = compute_holosoma_stages(raw_joints, smpl_scale, toe, mapped)
    holosoma = MethodViz("holosoma", "holosoma", native.qpos, hs)

    # GMR v1 / v2: qpos + full stage dict (mapped-body pos per stage)
    def gmr_method(label, key, cls):
        rt = cls.from_config(cfg)
        res = rt.retarget()
        stages = {name.capitalize(): rt.gmr_stages[name]["pos"]
                  for name in ("mapped", "scaled", "offset", "ground")}
        stages = {"Original": raw_joints[:res.qpos.shape[0], :, :], **stages}  # 52-joint source
        return MethodViz(label, key, res.qpos, stages)

    methods = [holosoma,
               gmr_method("GMR-SOCP v1", "gmr_socp_v1", GmrSocpRetargeterV1),
               gmr_method("GMR-SOCP v2", "gmr_socp_v2", GmrSocpRetargeterV2)]

    keys = tuple(m.robot_key for m in methods)
    viewer = Viewer(robot_model_path=cfg.robot_config.ROBOT_URDF_FILE,
                    object_model_path=None, stage_keys=keys)
    viewer.bind_methods(methods)
    input("Stage viewer at http://localhost:8080 — Enter to exit ...")
```
Confirm the stage label casing matches `src/stages.py` (`Mapped/Scaled/Offset/Ground` — `.capitalize()` gives exactly those; `Original` is explicit). If `constants.JOINTS_MAPPING` keys are the human joint names, `mapped` indexing is correct; verify with a quick print and adjust if it is a dict of robot->human (use its human-side names).

- [ ] **Step 2: Remove the dead old API.** Delete `Viewer.bind` and the old registry-based `_redraw` if nothing references them; remove the old `bind`-based test in `tests/test_viewer.py`. `git grep -n "\.bind(\|extra_qpos\|STAGE_SPECS\|stage_labels"` — there should be no remaining real callers.

- [ ] **Step 3: Run the full suite**
`/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/ -q`
Expected: all pass (parity/golden unchanged; viewer + stage tests green).

- [ ] **Step 4: Bounded headless smoke**
```bash
echo "" | timeout 400 /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python examples/view_stages.py --data_path demo_data/OMOMO_new --task-type robot_only --task-name sub3_largebox_003 --data_format smplh 2>&1 | tail -20
```
Expected: runs native + v1 + v2 (a few minutes), opens viser, prints the viewer URL, exits cleanly on the piped newline, NO Traceback.

- [ ] **Step 5: Commit**
```bash
git add examples/view_stages.py src/viewer.py tests/test_viewer.py
git commit -m "feat(viz): view_stages shows per-method source + preprocessing stages"
```

---

## Self-Review notes
- **Spec coverage:** per-method registry → Task 1; holosoma stages (native qpos untouched) → Task 2; GMR full stage dict → Task 3; MethodViz + Method/Stage dropdowns + show-only-selection → Task 4; view_stages 3 methods + smoke → Task 5. Points-only skeletons (no bones) honored. Object/climbing out of scope.
- **Coupled UI change:** the suite is briefly red between Task 1 and Task 5 (the old `STAGE_SPECS` API is removed in Task 1 but `viewer`/`view_stages` switch in Tasks 4-5). Each task still runs its own targeted tests green; the FULL suite is green again at Task 5 Step 3. Note this when reviewing intermediate tasks.
- **Open items for the implementer:** viser `GuiDropdownHandle.options` settable (Task 4 — else recreate); empty-point-cloud clear behaviour (Task 4); `constants.JOINTS_MAPPING` human-name resolution for the holosoma `Mapped` indices (Task 5); `load_motion_data` returning RAW (un-preprocessed) joints (verify — it should, preprocessing is a separate call).
