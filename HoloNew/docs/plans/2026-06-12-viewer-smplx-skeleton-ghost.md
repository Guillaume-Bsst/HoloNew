# Stage Viewer: SMPL-X mesh, body/finger skeleton, ghost overlay — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the annex stage viewer so the original motion shows as a posed SMPL-X mesh + object mesh, the 52-joint source skeleton renders with body/finger bone+joint toggles, the `Original` stage is uniform across methods, and a ghost `(method, stage)` overlay can be drawn faded.

**Architecture:** Extend the existing lean `Viewer` (`src/viewer.py`) and its launcher (`examples/view_stages.py`) in place — no panel architecture. Reuse `correspondence/human_body.HumanBody` for SMPL-X posing. Add a small pure-data skeleton-topology module and one quaternion loader. All new viewer inputs are optional so existing callers/tests keep working.

**Tech Stack:** Python, numpy, viser, yourdfpy, smplx/torch/scipy (optional at runtime), pytest, tyro.

**Run all commands from** `modules/01_retargeting/HoloNew/HoloNew` **with the `holonew` conda env active:**
```bash
source ../scripts/source_retargeting_setup.sh
cd modules/01_retargeting/HoloNew/HoloNew
```

---

### Task 1: Skeleton topology module

**Files:**
- Create: `src/skeleton.py`
- Test: `tests/test_skeleton.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_skeleton.py
import numpy as np
from HoloNew.config_types.data_type import SMPLH_DEMO_JOINTS
from HoloNew.src import skeleton


def test_bone_and_joint_indices_in_range():
    n = len(SMPLH_DEMO_JOINTS)  # 52
    all_bones = skeleton.BODY_BONES + skeleton.FINGER_BONES
    for a, b in all_bones:
        assert 0 <= a < n and 0 <= b < n
    for i in skeleton.BODY_JOINT_INDICES + skeleton.FINGER_JOINT_INDICES:
        assert 0 <= i < n


def test_body_and_finger_joint_sets_disjoint_and_cover():
    body = set(skeleton.BODY_JOINT_INDICES)
    finger = set(skeleton.FINGER_JOINT_INDICES)
    assert body.isdisjoint(finger)
    assert body | finger == set(range(len(SMPLH_DEMO_JOINTS)))


def test_colors_are_uint8_rgb():
    for c in (skeleton.COLOR_BODY, skeleton.COLOR_FINGER,
              skeleton.COLOR_GHOST_BODY, skeleton.COLOR_GHOST_FINGER,
              skeleton.COLOR_STAGE, skeleton.COLOR_GHOST_STAGE):
        assert c.dtype == np.uint8 and c.shape == (3,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_skeleton.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'HoloNew.src.skeleton'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/skeleton.py
"""52-joint SMPLH skeleton topology and colours for the stage viewer.

Indices are positions into SMPLH_DEMO_JOINTS (intermimic MuJoCo order), so the
same bone/joint tables drive any (T, 52, 3) source-skeleton frame.
"""
from __future__ import annotations

import numpy as np

# Bones connecting the 18 body joints (+ the 4 right-arm joints at 33-36).
BODY_BONES: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (12, 13),
    (11, 14), (14, 15), (15, 16), (16, 17),
    (11, 33), (33, 34), (34, 35), (35, 36),
]

# Finger bones: wrist -> 5 finger roots, then each finger's two distal links,
# for both hands (left wrist = 17, right wrist = 36).
FINGER_BONES: list[tuple[int, int]] = (
    [(17, 18), (17, 21), (17, 24), (17, 27), (17, 30)]
    + [(18, 19), (19, 20), (21, 22), (22, 23), (24, 25), (25, 26),
       (27, 28), (28, 29), (30, 31), (31, 32)]
    + [(36, 37), (36, 40), (36, 43), (36, 46), (36, 49)]
    + [(37, 38), (38, 39), (40, 41), (41, 42), (43, 44), (44, 45),
       (46, 47), (47, 48), (49, 50), (50, 51)]
)

BODY_JOINT_INDICES: list[int] = list(range(18)) + [33, 34, 35, 36]
FINGER_JOINT_INDICES: list[int] = list(range(18, 33)) + list(range(37, 52))

COLOR_BODY = np.array([70, 130, 220], dtype=np.uint8)
COLOR_FINGER = np.array([120, 170, 230], dtype=np.uint8)
COLOR_GHOST_BODY = np.array([150, 180, 220], dtype=np.uint8)
COLOR_GHOST_FINGER = np.array([185, 205, 235], dtype=np.uint8)
COLOR_STAGE = np.array([230, 140, 30], dtype=np.uint8)
COLOR_GHOST_STAGE = np.array([240, 200, 150], dtype=np.uint8)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_skeleton.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/skeleton.py tests/test_skeleton.py
git commit -m "feat(viz): 52-joint SMPLH skeleton topology + colours"
```

---

### Task 2: Per-joint quaternion loader

**Files:**
- Modify: `src/utils.py` (add `load_intermimic_quats` next to `load_intermimic_data`)
- Test: `tests/test_intermimic_quats.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_intermimic_quats.py
from pathlib import Path

import numpy as np
from HoloNew.src.utils import load_intermimic_quats

_PT = Path("demo_data/OMOMO_new/sub3_largebox_003.pt")


def test_quats_shape_and_unit_norm():
    quats = load_intermimic_quats(str(_PT))
    assert quats.ndim == 3 and quats.shape[1:] == (52, 4)
    norms = np.linalg.norm(quats.reshape(-1, 4), axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_intermimic_quats.py -v`
Expected: FAIL with `ImportError: cannot import name 'load_intermimic_quats'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/utils.py` (top, alongside the existing imports, add `from scipy.spatial.transform import Rotation as R` only if not already imported), then add the function right after `load_intermimic_data`:

```python
# Right-multiplier that undoes intermimic's PHC "upright_start" twist baked into
# the stored global quats (interact2mimic.py writes global_rot * Q^-1, xyzw).
_UPRIGHT_START_FIX_XYZW = np.array([0.5, 0.5, 0.5, 0.5])


def load_intermimic_quats(file_path):
    """Per-joint global SMPL-X orientations from a .pt, MuJoCo order, wxyz.

    Returns (T, 52, 4) float32. The intermimic `upright_start` twist is undone so
    these are true SMPL-X global orientations (the form HumanBody.placed_verts
    expects). The .pt stores them xyzw at slice [383:383+52*4].
    """
    from scipy.spatial.transform import Rotation as R

    data = torch.load(file_path, map_location="cpu").detach().numpy()
    quats_xyzw = data[:, 383 : 383 + 52 * 4].reshape(-1, 52, 4)
    t, j, _ = quats_xyzw.shape
    fixed = R.from_quat(quats_xyzw.reshape(-1, 4)) * R.from_quat(_UPRIGHT_START_FIX_XYZW)
    quats_xyzw = fixed.as_quat().reshape(t, j, 4)
    return quats_xyzw[:, :, [3, 0, 1, 2]].astype(np.float32)
```

(`torch` and `np` are already imported at the top of `src/utils.py`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_intermimic_quats.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/utils.py tests/test_intermimic_quats.py
git commit -m "feat(viz): load per-joint SMPL-X quats from intermimic .pt"
```

---

### Task 3: Viewer accepts the original motion (backward-compatible)

**Files:**
- Modify: `src/viewer.py` (`Viewer.__init__`)
- Test: `tests/test_viewer.py` (add one test)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_viewer.py
def test_viewer_stores_original_motion(robot_urdf):
    import numpy as np
    from HoloNew.src.viewer import Viewer
    oj = np.zeros((4, 52, 3), dtype=np.float32)
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None,
               original_joints=oj)
    assert v.original_joints.shape == (4, 52, 3)
    assert v.original_quats is None and v.human_body is None
    v.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_viewer.py::test_viewer_stores_original_motion -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'original_joints'`

- [ ] **Step 3: Write minimal implementation**

In `src/viewer.py`, change the `Viewer.__init__` signature and add the stores. Replace:

```python
    def __init__(self, robot_model_path: str, object_model_path: str | None,
                 stage_keys: tuple[str, ...] = ("socp",),
                 has_dynamic_object: bool = False) -> None:
        self.robot_model_path = robot_model_path
        self.object_model_path = object_model_path
        self.has_dynamic_object = has_dynamic_object
```

with:

```python
    def __init__(self, robot_model_path: str, object_model_path: str | None,
                 stage_keys: tuple[str, ...] = ("socp",),
                 has_dynamic_object: bool = False,
                 original_joints: np.ndarray | None = None,
                 original_quats: np.ndarray | None = None,
                 object_poses: np.ndarray | None = None,
                 human_body: object | None = None) -> None:
        self.robot_model_path = robot_model_path
        self.object_model_path = object_model_path
        self.has_dynamic_object = has_dynamic_object
        # Original source motion, shared by every method's "Original" stage.
        self.original_joints = original_joints
        self.original_quats = original_quats
        self.object_poses = object_poses
        self.human_body = human_body
        self._smplx_handle = None
        self._object_handle = None
        self._dynamic_handles: list = []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_viewer.py::test_viewer_stores_original_motion -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/viewer.py tests/test_viewer.py
git commit -m "feat(viz): Viewer accepts shared original motion inputs"
```

---

### Task 4: Skeleton rendering + toggles for the Original stage

**Files:**
- Modify: `src/viewer.py` (add toggles in `bind_methods`, `_draw_skeleton`, `_draw_stage_points`, rework `_redraw`/clearing)
- Test: `tests/test_viewer.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_viewer.py
def test_original_stage_renders_with_toggles(robot_urdf):
    import numpy as np
    from HoloNew.src.viewer import Viewer, MethodViz
    oj = np.zeros((3, 52, 3), dtype=np.float32)
    m = MethodViz(label="GMR-SOCP v1", robot_key="gmr_socp_v1",
                  qpos=np.zeros((3, 36)),
                  stages={"Original": oj, "Mapped": np.zeros((3, 14, 3))})
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None,
               stage_keys=("gmr_socp_v1",), original_joints=oj)
    v.bind_methods([m])
    # the four skeleton toggles exist
    for cb in (v._tog_body_bones, v._tog_finger_bones,
               v._tog_body_joints, v._tog_finger_joints):
        assert cb.value in (True, False)
    # Original + Mapped both redraw without error under toggle changes
    v._stage_dd.value = "Original"; v._redraw(0)
    v._tog_finger_bones.value = False; v._redraw(0)
    v._stage_dd.value = "Mapped"; v._redraw(0)
    v.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_viewer.py::test_original_stage_renders_with_toggles -v`
Expected: FAIL with `AttributeError: 'Viewer' object has no attribute '_tog_body_bones'`

- [ ] **Step 3: Write minimal implementation**

In `src/viewer.py` add the skeleton import near the top:

```python
from . import skeleton
```

In `bind_methods`, after the existing `Display` folder block (which creates
`self._method_dd` / `self._stage_dd`), add a Skeleton folder:

```python
        with self.server.gui.add_folder("Skeleton"):
            self._tog_body_bones = self.server.gui.add_checkbox("Body bones", True)
            self._tog_finger_bones = self.server.gui.add_checkbox("Finger bones", True)
            self._tog_body_joints = self.server.gui.add_checkbox("Body joints", True)
            self._tog_finger_joints = self.server.gui.add_checkbox("Finger joints", False)

        for _cb in (self._tog_body_bones, self._tog_finger_bones,
                    self._tog_body_joints, self._tog_finger_joints):
            @_cb.on_update
            def _(_evt):
                self._redraw(int(self._slider.value))
```

Add the rendering helpers (place above `_redraw`):

```python
    def _original_frame(self, frame: int) -> np.ndarray:
        return self.original_joints[frame].astype(np.float32)

    def _draw_skeleton(self, prefix: str, pos: np.ndarray, *, ghost: bool) -> None:
        """52-joint source skeleton: body/finger bones + joints, toggle-gated."""
        body_col = skeleton.COLOR_GHOST_BODY if ghost else skeleton.COLOR_BODY
        finger_col = skeleton.COLOR_GHOST_FINGER if ghost else skeleton.COLOR_FINGER
        lw = 1.5 if ghost else 3.5

        segs, seg_cols = [], []
        if self._tog_body_bones.value:
            segs += [[pos[a], pos[b]] for a, b in skeleton.BODY_BONES]
            seg_cols += [body_col] * len(skeleton.BODY_BONES)
        if self._tog_finger_bones.value:
            segs += [[pos[a], pos[b]] for a, b in skeleton.FINGER_BONES]
            seg_cols += [finger_col] * len(skeleton.FINGER_BONES)
        if segs:
            arr = np.asarray(segs, dtype=np.float32)
            cols = np.repeat(np.asarray(seg_cols, np.uint8)[:, None, :], 2, axis=1)
            h = self.server.scene.add_line_segments(
                f"{prefix}/bones", arr, cols, line_width=lw)
            self._dynamic_handles.append(h)

        j_idx, j_cols = [], []
        if self._tog_body_joints.value:
            j_idx += skeleton.BODY_JOINT_INDICES
            j_cols += [body_col] * len(skeleton.BODY_JOINT_INDICES)
        if self._tog_finger_joints.value:
            j_idx += skeleton.FINGER_JOINT_INDICES
            j_cols += [finger_col] * len(skeleton.FINGER_JOINT_INDICES)
        if j_idx:
            h = self.server.scene.add_point_cloud(
                f"{prefix}/joints", pos[j_idx].astype(np.float32),
                np.asarray(j_cols, np.uint8), point_size=0.025)
            self._dynamic_handles.append(h)

    def _draw_stage_points(self, prefix: str, pos: np.ndarray, *, ghost: bool) -> None:
        """Mapped/preprocessing stages: joint points only (no bone topology)."""
        if not self._tog_body_joints.value:
            return
        col = skeleton.COLOR_GHOST_STAGE if ghost else skeleton.COLOR_STAGE
        h = self.server.scene.add_point_cloud(
            f"{prefix}/joints", pos.astype(np.float32), col, point_size=0.025)
        self._dynamic_handles.append(h)
```

Replace the existing `_clear_skeleton` with a generic clear, and rework `_redraw`:

```python
    def _clear_dynamic(self) -> None:
        for h in self._dynamic_handles:
            h.remove()
        self._dynamic_handles = []

    def _redraw(self, frame: int) -> None:
        method = self._methods[self._method_dd.value]
        stage = self._stage_dd.value
        self._clear_dynamic()
        self._hide_all_robots()
        if stage == ROBOT_STAGE:
            self.robots[method.robot_key].urdf.show_visual = True
            self.draw_q(method.qpos[frame], stage=method.robot_key)
        elif stage == "Original" and self.original_joints is not None:
            self._draw_skeleton("/active", self._original_frame(frame), ghost=False)
        else:
            self._draw_stage_points("/active", method.stages[stage][frame], ghost=False)
```

Delete the old `_clear_skeleton` method and the `self._skeleton` references; the
test from Task 0 (`test_bind_methods_builds_method_and_stage`) still passes
because `_draw_stage_points` handles the `Mapped` stage and `ROBOT_STAGE` the robot.

- [ ] **Step 4: Run the viewer tests**

Run: `pytest tests/test_viewer.py -v`
Expected: PASS (all, including the pre-existing tests)

- [ ] **Step 5: Commit**

```bash
git add src/viewer.py tests/test_viewer.py
git commit -m "feat(viz): body/finger skeleton rendering + toggles for Original stage"
```

---

### Task 5: SMPL-X mesh + object mesh toggles (graceful)

**Files:**
- Modify: `src/viewer.py` (`bind_methods` Meshes folder, `_draw_smplx_mesh`, `_draw_object`, call them in `_redraw`)
- Test: `tests/test_viewer.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_viewer.py
def test_smplx_toggle_noop_without_body(robot_urdf):
    import numpy as np
    from HoloNew.src.viewer import Viewer, MethodViz
    oj = np.zeros((3, 52, 3), dtype=np.float32)
    m = MethodViz(label="GMR-SOCP v1", robot_key="gmr_socp_v1",
                  qpos=np.zeros((3, 36)), stages={"Original": oj})
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None,
               stage_keys=("gmr_socp_v1",), original_joints=oj,
               original_quats=None, human_body=None)
    v.bind_methods([m])
    v._tog_smplx.value = True      # no human_body -> must not raise
    v._stage_dd.value = "Original"; v._redraw(0)
    assert v._smplx_handle is None
    v.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_viewer.py::test_smplx_toggle_noop_without_body -v`
Expected: FAIL with `AttributeError: 'Viewer' object has no attribute '_tog_smplx'`

- [ ] **Step 3: Write minimal implementation**

In `bind_methods`, add a Meshes folder after the Skeleton folder:

```python
        with self.server.gui.add_folder("Meshes"):
            self._tog_smplx = self.server.gui.add_checkbox("SMPL-X mesh", False)
            self._tog_object = self.server.gui.add_checkbox("Object mesh", False)
        for _cb in (self._tog_smplx, self._tog_object):
            @_cb.on_update
            def _(_evt):
                self._redraw(int(self._slider.value))
```

Add the mesh helpers:

```python
    def _draw_smplx_mesh(self, frame: int) -> None:
        show = (self._tog_smplx.value and self.human_body is not None
                and self.original_quats is not None and self.original_joints is not None)
        if not show:
            if self._smplx_handle is not None:
                self._smplx_handle.visible = False
            return
        verts = self.human_body.placed_verts(
            self.original_quats[frame], self.original_joints[frame, 0],
            frame_idx=frame).astype(np.float32)
        if self._smplx_handle is None:
            self._smplx_handle = self.server.scene.add_mesh_simple(
                "/human/mesh", vertices=verts, faces=self.human_body.faces,
                color=(150, 150, 150), opacity=0.7)
        else:
            self._smplx_handle.vertices = verts
            self._smplx_handle.visible = True

    def _draw_object(self, frame: int) -> None:
        show = (self._tog_object.value and self.viser_object is not None
                and self.object_base is not None and self.object_poses is not None)
        if self.viser_object is not None:
            self.viser_object.show_visual = bool(show)
        if not show:
            return
        # object_poses layout: [x, y, z, qw, qx, qy, qz] (MuJoCo order).
        self.object_base.position = self.object_poses[frame, :3]
        self.object_base.wxyz = self.object_poses[frame, 3:7]
```

Call both at the end of `_redraw`, before returning:

```python
        self._draw_smplx_mesh(frame)
        self._draw_object(frame)
```

- [ ] **Step 4: Run the viewer tests**

Run: `pytest tests/test_viewer.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add src/viewer.py tests/test_viewer.py
git commit -m "feat(viz): SMPL-X mesh + object mesh toggles (graceful when absent)"
```

---

### Task 6: Ghost overlay (independent method + stage)

**Files:**
- Modify: `src/viewer.py` (`bind_methods` Ghost folder, ghost branch in `_redraw`)
- Test: `tests/test_viewer.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_viewer.py
def test_ghost_overlays_skeleton_stage(robot_urdf):
    import numpy as np
    from HoloNew.src.viewer import Viewer, MethodViz
    oj = np.zeros((3, 52, 3), dtype=np.float32)
    m1 = MethodViz(label="GMR-SOCP v1", robot_key="gmr_socp_v1",
                   qpos=np.zeros((3, 36)),
                   stages={"Original": oj, "Mapped": np.zeros((3, 14, 3))})
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None,
               stage_keys=("gmr_socp_v1",), original_joints=oj)
    v.bind_methods([m1])
    assert v._ghost_stage_dd.value == "Off"
    # Robot stage is not a ghost option
    assert "Robot" not in v._ghost_stage_dd.options
    v._ghost_method_dd.value = "GMR-SOCP v1"
    v._ghost_stage_dd.value = "Mapped"
    v._redraw(0)   # must not raise
    v._ghost_stage_dd.value = "Off"
    v._redraw(0)
    v.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_viewer.py::test_ghost_overlays_skeleton_stage -v`
Expected: FAIL with `AttributeError: 'Viewer' object has no attribute '_ghost_stage_dd'`

- [ ] **Step 3: Write minimal implementation**

Add the imports the ghost needs at the top of `src/viewer.py` (extend the existing
stages import):

```python
from .stages import ROBOT_STAGE, method_labels, stages_for_method
```

(`method_for_label` is not needed; skeleton stages = `stages_for_method(label)` minus `ROBOT_STAGE`.)

In `bind_methods`, after the Meshes folder add the Ghost folder. Compute the
ghost stage options for a method as its stages without the robot stage:

```python
        def _ghost_stages(label: str) -> list[str]:
            return ["Off"] + [s for s in stages_for_method(label) if s != ROBOT_STAGE]

        with self.server.gui.add_folder("Ghost"):
            first = methods[0].label
            self._ghost_method_dd = self.server.gui.add_dropdown(
                "Method", options=method_labels(), initial_value=first)
            self._ghost_stage_dd = self.server.gui.add_dropdown(
                "Stage", options=_ghost_stages(first), initial_value="Off")

        @self._ghost_method_dd.on_update
        def _(_evt):
            self._ghost_stage_dd.options = _ghost_stages(self._ghost_method_dd.value)
            self._ghost_stage_dd.value = "Off"
            self._redraw(int(self._slider.value))

        @self._ghost_stage_dd.on_update
        def _(_evt):
            self._redraw(int(self._slider.value))
```

At the end of `_redraw` (after the mesh draws), add the ghost branch:

```python
        g_stage = self._ghost_stage_dd.value
        if g_stage != "Off":
            g_method = self._methods[self._ghost_method_dd.value]
            if g_stage == "Original" and self.original_joints is not None:
                self._draw_skeleton("/ghost", self._original_frame(frame), ghost=True)
            elif g_stage in g_method.stages:
                self._draw_stage_points("/ghost", g_method.stages[g_stage][frame], ghost=True)
```

(Ghost handles are appended to `self._dynamic_handles`, so the next `_redraw`'s
`_clear_dynamic()` removes them — no separate ghost clearing needed.)

- [ ] **Step 4: Run the viewer tests**

Run: `pytest tests/test_viewer.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add src/viewer.py tests/test_viewer.py
git commit -m "feat(viz): ghost overlay with independent method + stage dropdowns"
```

---

### Task 7: Wire the original motion into `view_stages.py`

**Files:**
- Modify: `examples/view_stages.py`
- Test: manual smoke (no unit test — needs the SMPL-X model dir + viser)

- [ ] **Step 1: Implement the wiring**

In `examples/view_stages.py`, add imports:

```python
import logging

from HoloNew.config_types.task import TaskConfig  # noqa: F401  (already via cfg)
from HoloNew.src.utils import load_intermimic_quats
from HoloNew.src.correspondence.constants import SMPLX_MODEL_DIR_DEFAULT
from HoloNew.src.correspondence.human_body import HumanBody

logger = logging.getLogger(__name__)
```

After `raw_joints, _object_poses, smpl_scale = load_motion_data(...)`, build the
original-motion inputs. Quats only exist for the smplh `.pt` format; guard it:

```python
    original_quats = None
    if data_format == "smplh":
        try:
            original_quats = load_intermimic_quats(str(cfg.data_path / f"{cfg.task_name}.pt"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("No per-joint quats (%s); SMPL-X mesh disabled.", exc)

    human_body = None
    if original_quats is not None:
        try:
            human_body = HumanBody(SMPLX_MODEL_DIR_DEFAULT, None, "neutral")
        except Exception as exc:  # noqa: BLE001
            logger.warning("SMPL-X unavailable (%s); mesh disabled.", exc)
```

Make `Original` uniform across methods. After `methods = [...]` is built, before
constructing the viewer:

```python
    T = min(m.qpos.shape[0] for m in methods)
    for m in methods:
        m.stages["Original"] = raw_joints[:T, :, :]
```

Pass the new inputs to the `Viewer(...)` call (replacing the existing one):

```python
    viewer = Viewer(
        robot_model_path=cfg.robot_config.ROBOT_URDF_FILE,
        object_model_path=None,
        stage_keys=keys,
        original_joints=raw_joints[:T, :, :],
        original_quats=None if original_quats is None else original_quats[:T],
        object_poses=None,
        human_body=human_body,
    )
```

(Object mesh wiring for object_interaction tasks is deliberately left for a later
change — robot_only has no object. The object toggle is a no-op until then, which
matches the graceful-degradation design.)

- [ ] **Step 2: Syntax check**

Run: `python -c "import ast; ast.parse(open('examples/view_stages.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Smoke-run headlessly (Ctrl-C after the viewer URL prints)**

Run:
```bash
timeout 120 python examples/view_stages.py --task-type robot_only \
  --task-name sub3_largebox_003 --data_format smplh --methods gmr_socp_v1 || true
```
Expected: solves, prints `Stage viewer at http://localhost:8080`; no traceback before the prompt.

- [ ] **Step 4: Commit**

```bash
git add examples/view_stages.py
git commit -m "feat(viz): wire original SMPL-X motion + uniform Original stage into view_stages"
```

---

### Task 8: Docs + full regression

**Files:**
- Modify: `COMMAND.md` (document the new toggles + ghost)
- Test: full viewer/stage/skeleton suite

- [ ] **Step 1: Update COMMAND.md §2**

Under the `view_stages.py` section, add a short paragraph after the `--methods`
examples:

```markdown
The viewer also exposes (GUI folders, right panel):

- **Skeleton** — toggle body/finger **bones** and **joints** of the 52-joint
  source skeleton (shown on the `Original` stage).
- **Meshes** — overlay the posed **SMPL-X mesh** (needs the SMPL-X model dir) and
  the **object mesh**, on any stage.
- **Ghost** — pick a second `(Method, Stage)` to overlay faded for comparison
  (`Off` to disable). The ghost covers skeleton stages, not the robot.
```

- [ ] **Step 2: Run the full relevant suite**

Run: `pytest tests/test_skeleton.py tests/test_intermimic_quats.py tests/test_viewer.py tests/test_stages.py -v`
Expected: PASS (all)

- [ ] **Step 3: Commit**

```bash
git add COMMAND.md
git commit -m "docs: document skeleton/mesh/ghost controls in view_stages cheat sheet"
```

---

## Self-review notes

- **Spec coverage:** SMPL-X mesh (Task 5/7), object mesh toggle (Task 5), body/finger
  bones+joints (Task 1/4), `Original` uniform across methods (Task 7), ghost
  method+stage (Task 6), graceful degradation (Task 5/7), tests (Tasks 1–6, 8).
- **Backward compatibility:** all new `Viewer.__init__` params default to `None`;
  pre-existing `tests/test_viewer.py` cases construct `Viewer` without them and
  select `Mapped`/`Robot` stages, both handled by `_draw_stage_points` / the robot
  branch.
- **Names are consistent across tasks:** `_tog_body_bones`, `_tog_finger_bones`,
  `_tog_body_joints`, `_tog_finger_joints`, `_tog_smplx`, `_tog_object`,
  `_ghost_method_dd`, `_ghost_stage_dd`, `_dynamic_handles`, `_draw_skeleton`,
  `_draw_stage_points`, `_draw_smplx_mesh`, `_draw_object`, `_clear_dynamic`,
  `_original_frame`.
- **Object-pose layout:** `_draw_object` assumes `[x,y,z,qw..]` (MuJoCo order); the
  `object_poses` fed in Task 7 is `None` for now, so the assumption is only
  exercised when object wiring is added later.
