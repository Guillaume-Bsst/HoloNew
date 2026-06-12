# Interaction Visualization (test_pipe parity) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show the test_pipe contact/interaction visuals in `view_stages` — SMPL-X human contact probes, object/floor probes, contact directions, and the per-frame transport of the contact field onto the solved G1 — all coloured by signed distance, computed in the TEST-SOCP solve and exposed on its result.

**Architecture:** `TestSocpRetargeter.retarget()` already builds the per-frame SMPL-X probe `ContactField` at the Grounded pose. We make the probe use the correspondence's embedded point cache (so `human_idx` aligns), record the probe world points + an analytic floor field, FK-transport the correspondence onto the solved robot per frame, and expose all of it on `RetargetResult`. `view_stages` passes it to `Viewer`, which draws it in the existing "Test" GUI folder, gated by stage (human→Grounded, G1→Robot, object/floor/SDF→any).

**Tech Stack:** Python, numpy, mujoco, yourdfpy, viser, pytest.

**Run all commands from** `modules/01_retargeting/HoloNew/HoloNew` **with the env python:**
`~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest ...`

## File structure
- `src/test_socp/correspondence/transport.py` (new): URDF FK transport of the correspondence onto a posed robot.
- `src/test_socp/contact/smplx_field.py` (modify): accept a prebuilt cache; return probe points + field.
- `src/retarget_result.py` (modify): optional interaction fields.
- `src/test_socp/test_socp.py` (modify): build probe with correspondence cache, record per-frame interaction data + G1 transport, expose on result.
- `examples/view_stages.py` (modify): pull interaction data from the TEST-SOCP result, pass to `Viewer`.
- `src/viewer.py` (modify): "Test" folder toggles + `_draw_*` helpers + gating.
- Tests: `tests/test_transport.py` (new), `tests/test_smplx_probe.py` (new), extend `tests/test_viewer.py`.

---

### Task 1: Correspondence → robot transport helper

**Files:**
- Create: `src/test_socp/correspondence/transport.py`
- Test: `tests/test_transport.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transport.py
import numpy as np
from HoloNew.src.test_socp.correspondence.transport import transported_points


def test_transported_points_identity_and_offset():
    # Two G1 points on one link "a" whose world transform is identity.
    transforms = {"a": np.eye(4, dtype=np.float64)}
    link_idx = np.array([0, 0])
    offset_local = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64)
    out = transported_points(transforms, link_idx, offset_local, ["a"])
    np.testing.assert_allclose(out, offset_local)


def test_transported_points_applies_rotation_translation():
    # 90 deg about z, then translate by (10, 0, 0).
    T = np.eye(4)
    T[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    T[:3, 3] = [10.0, 0.0, 0.0]
    out = transported_points({"a": T}, np.array([0]), np.array([[1.0, 0.0, 0.0]]), ["a"])
    np.testing.assert_allclose(out[0], [10.0, 1.0, 0.0], atol=1e-9)
```

- [ ] **Step 2: Run it — fails with ModuleNotFoundError.**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_transport.py -v`
Expected: FAIL (no module `transport`).

- [ ] **Step 3: Implement `src/test_socp/correspondence/transport.py`**

```python
"""Place the fixed human->G1 correspondence onto a posed robot.

The ~0-compute online path: given the robot's link world transforms for a frame
and the correspondence (link index + link-local offset per G1 point), gather every
point onto the robot with one batched rigid transform. No optimal transport, no
contact recompute. Mirrors test_pipe's proximity transport.
"""
from __future__ import annotations

import numpy as np


def link_world_transforms(urdf, qpos: np.ndarray, link_names) -> dict[str, np.ndarray]:
    """World 4x4 transform per link name: base pose (qpos[:7]) composed with URDF FK.

    qpos: full robot config — [0:3] base pos, [3:7] base quat wxyz, [7:7+ndof] joints,
    in the same order the URDF's actuated joints expect (as the viewer's update_cfg).
    """
    qpos = np.asarray(qpos, dtype=np.float64)
    ndof = len(urdf.actuated_joints)
    urdf.update_cfg(qpos[7:7 + ndof])

    w, x, y, z = qpos[3:7]
    n = float(np.linalg.norm(qpos[3:7])) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    T_base = np.eye(4)
    T_base[:3, :3] = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    T_base[:3, 3] = qpos[0:3]
    return {ln: T_base @ np.asarray(urdf.get_transform(ln)) for ln in set(link_names)}


def transported_points(link_world_transforms: dict[str, np.ndarray],
                       link_idx: np.ndarray, offset_local: np.ndarray,
                       link_names) -> np.ndarray:
    """(N,3) world positions of the correspondence points placed on the posed robot."""
    link_idx = np.asarray(link_idx)
    offset_local = np.asarray(offset_local, dtype=np.float64)
    out = np.empty((len(link_idx), 3), dtype=np.float64)
    for li in np.unique(link_idx):
        T = link_world_transforms[link_names[int(li)]]
        sel = link_idx == li
        out[sel] = offset_local[sel] @ T[:3, :3].T + T[:3, 3]
    return out.astype(np.float32)
```

- [ ] **Step 4: Run — PASS (2 passed).**

- [ ] **Step 5: Commit**

```bash
git add src/test_socp/correspondence/transport.py tests/test_transport.py
git commit -m "feat(test_socp): correspondence->robot transport (FK placement)"
```

---

### Task 2: SMPL-X probe uses the correspondence cache + returns points

**Files:**
- Modify: `src/test_socp/contact/smplx_field.py`
- Test: `tests/test_smplx_probe.py`

- [ ] **Step 1: Write the failing test** (cache alignment: the probe samples exactly the cache it is given, so `human_idx` indexes its points)

```python
# tests/test_smplx_probe.py
import numpy as np
from HoloNew.src.test_socp.correspondence.human_body import PointCloudCache
from HoloNew.src.test_socp.contact.smplx_field import ProbeFrame


def test_probeframe_carries_points_and_field():
    pts = np.zeros((5, 3), np.float32)

    class _F:
        distance = np.zeros(5)
    pf = ProbeFrame(points=pts, field=_F())
    assert pf.points.shape == (5, 3)
    assert pf.field.distance.shape == (5,)
```

- [ ] **Step 2: Run — fails (no `ProbeFrame`).**

- [ ] **Step 3: Modify `smplx_field.py`**

Add a `ProbeFrame` dataclass, make `__call__` return it (points + field), and let
`build_smplx_ground_probe` accept an optional prebuilt `cache` (the correspondence
cache) so the probe points align with `human_idx`:

```python
@dataclass
class ProbeFrame:
    """One frame's probe output: the SMPL-X surface points (world) and their field."""
    points: np.ndarray          # (N, 3)
    field: "ContactField"
```

In `SmplxGroundProbe.__call__`, return the points too:

```python
    def __call__(self, t, quats_wxyz, pelvis_grounded) -> "ProbeFrame":
        world = self.human_body.placed_points(quats_wxyz, pelvis_grounded, self.cache, frame_idx=t)
        local = transform_points_world_to_local(self.obj_quat[t], self.obj_trans[t], world)
        return ProbeFrame(points=world.astype(np.float32),
                          field=self.object_sdf.query(local, self.margin))
```

In `build_smplx_ground_probe`, accept `cache=None` and use it when given:

```python
def build_smplx_ground_probe(task_name, omomo_dir, model_dir, object_sdf,
                             obj_poses, margin, density, cache=None):
    ...
    betas, gender = load_human_metadata(Path(omomo_dir), task_name)
    body = HumanBody(model_dir, betas, gender)
    cache = cache if cache is not None else body.build_point_cloud_cache(density)
    ...
```

- [ ] **Step 4: Run — PASS.**

- [ ] **Step 5: Commit**

```bash
git add src/test_socp/contact/smplx_field.py tests/test_smplx_probe.py
git commit -m "feat(test_socp): probe returns ProbeFrame(points, field); accepts a prebuilt cache"
```

---

### Task 3: RetargetResult carries interaction data

**Files:**
- Modify: `src/retarget_result.py`
- Test: `tests/test_retarget_result_interaction.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_retarget_result_interaction.py
import numpy as np
from HoloNew.src.retarget_result import RetargetResult


def test_result_holds_optional_interaction_fields():
    r = RetargetResult(qpos=np.zeros((2, 36)))
    assert r.human_probe_pts is None
    assert r.g1_transport_pts is None
    r2 = RetargetResult(qpos=np.zeros((2, 36)),
                        human_probe_pts=np.zeros((2, 5, 3)),
                        g1_transport_pts=np.zeros((2, 4, 3)))
    assert r2.human_probe_pts.shape == (2, 5, 3)
```

- [ ] **Step 2: Run — fails (TypeError: unexpected kwarg).**

- [ ] **Step 3: Add fields to `RetargetResult`**

```python
@dataclass
class RetargetResult:
    qpos: np.ndarray
    stages: dict[str, np.ndarray] = field(default_factory=dict)
    cost: float = 0.0
    # Interaction data (TEST-SOCP only; None elsewhere). T-length, frame-aligned.
    human_probe_pts: np.ndarray | None = None      # (T, N, 3) SMPL-X probe world points (Grounded)
    human_obj_dist: np.ndarray | None = None       # (T, N)    signed dist to object (SDF)
    human_flr_dist: np.ndarray | None = None       # (T, N)    signed dist to floor (analytic)
    human_witness: np.ndarray | None = None        # (T, N, 3) object-local witness for the object channel
    g1_transport_pts: np.ndarray | None = None     # (T, M, 3) correspondence points on the solved robot
    human_idx: np.ndarray | None = None            # (M,)      human point driving each G1 point
    obj_pose: np.ndarray | None = None             # (T, 7)    [qw,qx,qy,qz,x,y,z] for witness->world
    contact_fields: dict | None = None             # bundled 4-channel field (object/floor reverse)
    object_sdf: object | None = None               # boosted SDF (for the SDF Floor band reuse)
```

- [ ] **Step 4: Run — PASS.**

- [ ] **Step 5: Commit**

```bash
git add src/retarget_result.py tests/test_retarget_result_interaction.py
git commit -m "feat: RetargetResult carries optional interaction data"
```

---

### Task 4: TEST-SOCP records interaction data + G1 transport, exposes on result

**Files:**
- Modify: `src/test_socp/test_socp.py` (`from_config` probe build + cache; `retarget()` per-frame recording; result construction)
- Test: covered by the smoke run in Task 7 (needs SMPL-X + solve; no fast unit test).

- [ ] **Step 1: Use the correspondence cache when building the probe.**

In `from_config`, where `build_smplx_ground_probe(...)` is called, pass the correspondence cache so `human_idx` aligns:

```python
        if rt.object_sdf is not None:
            from HoloNew.src.test_socp.contact.constants import CONTACT_MARGIN_M, OMOMO_DIR_DEFAULT
            from HoloNew.src.test_socp.contact.smplx_field import build_smplx_ground_probe
            from HoloNew.src.test_socp.correspondence.human_body import PointCloudCache
            from HoloNew.src.utils import load_intermimic_data
            _, obj_poses = load_intermimic_data(str(pt_path))
            corr_cache = None
            if rt.correspondence is not None:
                corr_cache = PointCloudCache(tri_idx=rt.correspondence.tri_idx,
                                             bary=rt.correspondence.bary)
            rt.smplx_ground_probe = build_smplx_ground_probe(
                cfg.task_name, OMOMO_DIR_DEFAULT, SMPLX_MODEL_DIR_DEFAULT,
                rt.object_sdf, obj_poses[:T], CONTACT_MARGIN_M, HUMAN_GRID_DENSITY,
                cache=corr_cache)
            rt._obj_poses = obj_poses[:T]   # kept for the result's obj_pose passthrough
```

- [ ] **Step 2: Record per-frame interaction data in `retarget()`.**

Before the frame loop, prepare accumulators and a yourdfpy URDF for transport:

```python
        probe_pts, probe_obj, probe_flr, probe_wit, g1_pts = [], [], [], [], []
        urdf = None
        if self.smplx_ground_probe is not None and self.correspondence is not None:
            import yourdfpy
            urdf = yourdfpy.URDF.load(self.task_constants.ROBOT_URDF_FILE,
                                      load_meshes=False, build_scene_graph=True)
```

Replace the existing per-frame probe block with point-recording + floor + transport:

```python
            if self.smplx_ground_probe is not None:
                pf = self.smplx_ground_probe(t, self.human_quat[t], pelvis_grounded[t])
                self.smplx_sdf_fields.append(pf.field)
                probe_pts.append(pf.points)
                probe_obj.append(pf.field.distance.copy())
                probe_wit.append(pf.field.witness.copy())
                # Analytic floor: signed z distance, clamped to +margin past the band.
                from HoloNew.src.test_socp.contact.backends.floor import floor_field
                probe_flr.append(floor_field(pf.points, self.smplx_ground_probe.margin).distance.copy())
                if urdf is not None:
                    from HoloNew.src.test_socp.correspondence.transport import (
                        link_world_transforms, transported_points)
                    Tw = link_world_transforms(urdf, q, self.correspondence.link_names)
                    g1_pts.append(transported_points(
                        Tw, self.correspondence.link_idx,
                        self.correspondence.offset_local, self.correspondence.link_names))
```

(Note: `q` here is the solved config for frame `t`; record the transport AFTER the
two IK passes so it sits on the final solved pose. Move the transport append to just
before `out.append(np.copy(q))`.)

- [ ] **Step 3: Build the result with the interaction fields.**

Where `retarget()` returns `RetargetResult(qpos=np.array(out), ...)`, attach the data:

```python
        res = RetargetResult(qpos=np.array(out), stages={}, cost=0.0)
        if probe_pts:
            res.human_probe_pts = np.stack(probe_pts)
            res.human_obj_dist = np.stack(probe_obj)
            res.human_flr_dist = np.stack(probe_flr)
            res.human_witness = np.stack(probe_wit)
            res.obj_pose = getattr(self, "_obj_poses", None)
            res.contact_fields = self.contact_fields
            res.object_sdf = self.object_sdf
            if g1_pts:
                res.g1_transport_pts = np.stack(g1_pts)
                res.human_idx = self.correspondence.human_idx
        return res
```

- [ ] **Step 4: Syntax check**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -c "import ast; ast.parse(open('src/test_socp/test_socp.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add src/test_socp/test_socp.py
git commit -m "feat(test_socp): record probe points + floor + G1 transport, expose on result"
```

---

### Task 5: Viewer — "Test" folder interaction toggles + rendering + gating

**Files:**
- Modify: `src/viewer.py`
- Test: `tests/test_viewer.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_viewer.py
def test_interaction_toggles_and_gating(robot_urdf):
    import numpy as np
    from HoloNew.src.viewer import Viewer, MethodViz
    oj = np.zeros((3, 52, 3), dtype=np.float32)
    human = np.zeros((3, 6, 3), dtype=np.float32)
    g1 = np.zeros((3, 4, 3), dtype=np.float32)
    dist = np.zeros((3, 6), dtype=np.float32)
    m = MethodViz(label="TEST-SOCP", robot_key="test_socp",
                  qpos=np.zeros((3, 36)),
                  stages={"Original": oj, "Grounded": oj},
                  human_probe_pts=human, human_dist=dist,
                  g1_transport_pts=g1, g1_dist=dist[:, :4])
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None,
               stage_keys=("test_socp",), original_joints=oj)
    v.bind_methods([m])
    assert hasattr(v, "_tog_human") and hasattr(v, "_tog_g1_transport")
    # Human contact is gated to the Grounded stage.
    v._tog_human.value = True
    v._stage_dd.value = "Robot"; v._redraw(0)
    assert v._human_handle is None or v._human_handle.visible is False
    v._stage_dd.value = "Grounded"; v._redraw(0)
    assert v._human_handle is not None and v._human_handle.visible
    # G1 transport gated to the Robot stage.
    v._tog_g1_transport.value = True
    v._stage_dd.value = "Robot"; v._redraw(0)
    assert v._g1_transport_handle is not None and v._g1_transport_handle.visible
    v.close()
```

- [ ] **Step 2: Run — fails (no `_tog_human`).**

- [ ] **Step 3: Implement in `src/viewer.py`**

(a) `MethodViz`: add interaction fields:

```python
    human_probe_pts: np.ndarray | None = None   # (T, N, 3) Grounded-pose SMPL-X probes
    human_dist: np.ndarray | None = None        # (T, N)    signed distance (min(object, floor))
    g1_transport_pts: np.ndarray | None = None  # (T, M, 3) transported points on the robot
    g1_dist: np.ndarray | None = None           # (T, M)    each G1 point's human-source distance
```

(b) In `__init__`, add handles: `self._human_handle = None`, `self._g1_transport_handle = None`.

(c) In `bind_methods`, extend the "Test" folder (where `_tog_sdf` lives):

```python
        with self.server.gui.add_folder("Test"):
            self._tog_sdf = self.server.gui.add_checkbox("SDF Object", False)
            self._tog_sdf_floor = self.server.gui.add_checkbox("SDF Floor", False)
            self._tog_human = self.server.gui.add_checkbox("Human contact", False)
            self._tog_g1_transport = self.server.gui.add_checkbox("G1 transport", False)
            self._tog_directions = self.server.gui.add_checkbox("Directions", False)
        for _cb in (self._tog_sdf, self._tog_sdf_floor, self._tog_human,
                    self._tog_g1_transport, self._tog_directions):
            @_cb.on_update
            def _(_evt):
                self._redraw(int(self._slider.value))
```

(d) Add a generic coloured-cloud helper + the two gated draws. Import the colormap:

```python
from HoloNew.src.test_socp.contact.viz import signed_distance_colors
from HoloNew.src.test_socp.contact.constants import CONTACT_MARGIN_M
```

```python
    def _draw_signed_cloud(self, handle_attr, name, pts, dist, show):
        h = getattr(self, handle_attr)
        if not show:
            if h is not None:
                h.visible = False
            return
        cols = signed_distance_colors(dist, CONTACT_MARGIN_M)
        if h is None:
            h = self.server.scene.add_point_cloud(
                name, points=pts.astype(np.float32), colors=cols, point_size=0.01)
            setattr(self, handle_attr, h)
        else:
            h.points = pts.astype(np.float32)
            h.colors = cols
            h.visible = True

    def _draw_interaction(self, frame: int) -> None:
        method = self._methods.get(self._method_dd.value)
        stage = self._stage_dd.value
        # Human contact: Grounded stage only (the grounded SMPL-X).
        show_human = (method is not None and self._tog_human.value
                      and method.human_probe_pts is not None and stage == "Grounded")
        self._draw_signed_cloud(
            "_human_handle", "/test/human",
            None if not show_human else method.human_probe_pts[frame],
            None if not show_human else method.human_dist[frame], show_human)
        # G1 transport: Robot stage only (the solved robot).
        show_g1 = (method is not None and self._tog_g1_transport.value
                   and method.g1_transport_pts is not None and stage == ROBOT_STAGE)
        self._draw_signed_cloud(
            "_g1_transport_handle", "/test/g1_transport",
            None if not show_g1 else method.g1_transport_pts[frame],
            None if not show_g1 else method.g1_dist[frame], show_g1)
```

(e) Call it at the end of `_redraw`: `self._draw_interaction(frame)`.

- [ ] **Step 4: Run the viewer suite**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viewer.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/viewer.py tests/test_viewer.py
git commit -m "feat(viz): Test-folder human-contact + G1-transport overlays, stage-gated"
```

---

### Task 6: Object/Floor reverse channels + Directions + SDF Floor

**Files:**
- Modify: `src/viewer.py`
- Test: `tests/test_viewer.py`

- [ ] **Step 1: Write the failing test** (object/floor channels from `contact_fields`, any stage; SDF Floor band)

```python
# append to tests/test_viewer.py
def test_object_floor_channels_any_stage(robot_urdf):
    import numpy as np
    from HoloNew.src.viewer import Viewer, MethodViz
    from HoloNew.src.test_socp.contact.contact_field import ContactField
    oj = np.zeros((3, 52, 3), dtype=np.float32)

    def _ch(n):
        return ContactField(distance=np.zeros((3, n)), direction=np.zeros((3, n, 3)),
                            witness=np.zeros((3, n, 3)), active=np.zeros((3, n), bool))
    fields = {"object_human": _ch(7), "floor_human": _ch(5)}
    # object probes need anchor positions; pass them on the method.
    m = MethodViz(label="TEST-SOCP", robot_key="test_socp", qpos=np.zeros((3, 36)),
                  stages={"Original": oj}, contact_fields=fields,
                  object_probe_pts=np.zeros((7, 3)), floor_probe_pts=np.zeros((5, 3)))
    v = Viewer(robot_model_path=robot_urdf, object_model_path=None,
               stage_keys=("test_socp",), original_joints=oj)
    v.bind_methods([m])
    assert hasattr(v, "_tog_object_contact") and hasattr(v, "_tog_floor_contact")
    v._tog_object_contact.value = True
    v._stage_dd.value = "Robot"; v._redraw(0)        # any stage
    assert v._object_contact_handle is not None and v._object_contact_handle.visible
    v.close()
```

- [ ] **Step 2: Run — fails (no `_tog_object_contact`).**

- [ ] **Step 3: Implement**

(a) `MethodViz`: add `contact_fields: dict | None = None`, `object_probe_pts: np.ndarray | None = None` (M_o,3 object-local), `floor_probe_pts: np.ndarray | None = None` (M_f,3 world).

(b) In the "Test" folder add `self._tog_object_contact` and `self._tog_floor_contact` checkboxes (wired like the others), and handles `self._object_contact_handle = None`, `self._floor_contact_handle = None` in `__init__`.

(c) Object probes are object-local → lift by the active stage's object pose (reuse `_object_pose`); floor probes are world. In `_draw_interaction` add:

```python
        from HoloNew.src.holosoma.interaction_mesh import transform_points_local_to_world
        cf = None if method is None else method.contact_fields
        # Object_human probes: any stage, placed at the active object pose.
        f_obj = cf.get("object_human") if cf else None
        pose = self._object_pose(stage) if method is not None else None
        show_o = (self._tog_object_contact.value and f_obj is not None
                  and method.object_probe_pts is not None and pose is not None)
        opts = (transform_points_local_to_world(pose[frame, 3:7], pose[frame, :3],
                                                method.object_probe_pts) if show_o else None)
        self._draw_signed_cloud("_object_contact_handle", "/test/object_contact",
                                opts, f_obj.distance[frame] if show_o else None, show_o)
        # Floor_human probes: any stage, already world.
        f_flr = cf.get("floor_human") if cf else None
        show_f = (self._tog_floor_contact.value and f_flr is not None
                  and method.floor_probe_pts is not None)
        self._draw_signed_cloud("_floor_contact_handle", "/test/floor_contact",
                                method.floor_probe_pts if show_f else None,
                                f_flr.distance[frame] if show_f else None, show_f)
```

(d) **Directions** (probe→witness lines, gated to Grounded for human, Robot for G1):
add `_draw_directions(frame)` using `add_line_segments`, appended to `_dynamic_handles`
(cleared each redraw). Human witness is object-local → lift by the object pose.

```python
    def _draw_directions(self, frame: int) -> None:
        if not self._tog_directions.value:
            return
        method = self._methods.get(self._method_dd.value)
        if method is None:
            return
        from HoloNew.src.holosoma.interaction_mesh import transform_points_local_to_world
        stage = self._stage_dd.value
        if stage == "Grounded" and method.human_probe_pts is not None and method.human_witness is not None:
            pose = self._object_pose(stage)
            wit = transform_points_local_to_world(pose[frame, 3:7], pose[frame, :3],
                                                  method.human_witness[frame])
            self._draw_segments("/test/dir_human", method.human_probe_pts[frame], wit,
                                method.human_dist[frame])
        if stage == ROBOT_STAGE and method.g1_transport_pts is not None and method.g1_witness is not None:
            pose = self._object_pose(stage)
            wit = transform_points_local_to_world(pose[frame, 3:7], pose[frame, :3],
                                                  method.g1_witness[frame])
            self._draw_segments("/test/dir_g1", method.g1_transport_pts[frame], wit,
                                method.g1_dist[frame])

    def _draw_segments(self, name, a, b, dist):
        segs = np.stack([a, b], axis=1).astype(np.float32)            # (K,2,3)
        cols = np.repeat(signed_distance_colors(dist, CONTACT_MARGIN_M)[:, None, :], 2, axis=1)
        h = self.server.scene.add_line_segments(name, segs, cols, line_width=1.5)
        self._dynamic_handles.append(h)
```

Add `g1_witness: np.ndarray | None = None` to `MethodViz` (the human witness read via
`human_idx` for each G1 point — built in Task 7's wiring as
`human_witness[:, human_idx]`). Call `self._draw_directions(frame)` in `_redraw`.

(e) **SDF Floor**: reuse the analytic floor band like the object SDF shell. In
`_draw_sdf` (or a sibling), when `self._tog_sdf_floor.value` and an SDF floor band is
available, draw it. Provide the floor band points from view_stages (Task 7) as
`object_sdf_floor_pts` / `object_sdf_floor_cols`, stored on the Viewer like the SDF
shell, and draw them at world (no object pose) with a persistent handle.

- [ ] **Step 4: Run the viewer suite**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viewer.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/viewer.py tests/test_viewer.py
git commit -m "feat(viz): object/floor contact channels, directions, SDF floor band"
```

---

### Task 7: Wire the TEST-SOCP interaction data into view_stages

**Files:**
- Modify: `examples/view_stages.py`
- Test: manual smoke (needs SMPL-X + solve).

- [ ] **Step 1: In `build_test_socp` (the test_socp branch of `build_gmr`/its builder), pull the result's interaction data onto the `MethodViz`.**

After `res = rt.retarget()` for the test_socp method, set:

```python
    mv = MethodViz(label, key, res.qpos, stages, stage_bones=sb,
                   robot_skeleton=rs, stage_quats=sq, robot_quats=rq, g1_points=g1)
    if res.human_probe_pts is not None:
        import numpy as np
        mv.human_probe_pts = res.human_probe_pts
        # colour by the closer of object / floor (test_pipe).
        mv.human_dist = np.minimum(res.human_obj_dist, res.human_flr_dist)
        mv.contact_fields = res.contact_fields
        mv.object_probe_pts = None  # filled below if a bundled object channel exists
        mv.floor_probe_pts = None
        mv.human_witness = res.human_witness
    if res.g1_transport_pts is not None:
        mv.g1_transport_pts = res.g1_transport_pts
        mv.g1_dist = res.human_obj_dist[:, res.human_idx]    # each G1 reads its human's object dist
        mv.g1_witness = res.human_witness[:, res.human_idx]  # and its human's object witness
    return mv
```

(Only the test_socp builder sets these; the gmr_socp builder leaves them None.)

- [ ] **Step 2: Object/floor probe anchor points from the bundled `contact_fields`.**

The bundled object/floor channels carry their own probe positions only implicitly
(the witness of the reverse channel). For the object channel, use the object grid
samples (`make_object_grid`) and for the floor, `make_floor_grid`, both from
`test_socp.contact.probes`, matching how the bundled field was probed:

```python
    if res.contact_fields is not None:
        from HoloNew.src.test_socp.contact.probes import make_floor_grid, make_object_grid
        if "object_human" in res.contact_fields and object_mesh_verts is not None:
            mv.object_probe_pts = make_object_grid(object_mesh_verts, object_mesh_faces)
        if "floor_human" in res.contact_fields:
            cx = float(raw_joints[:, 0, 0].mean()); cy = float(raw_joints[:, 0, 1].mean())
            mv.floor_probe_pts = make_floor_grid(center_xy=(cx, cy))
```

(If a channel's probe count does not match its field length, skip that channel and
log a warning — the bundled field and the regenerated grid must share N.)

- [ ] **Step 3: SDF Floor band** — pass an analytic floor band to the Viewer, built once:

```python
    from HoloNew.src.test_socp.contact.backends.floor import floor_band_points  # add if missing
```

If `floor_band_points` does not exist, build the band inline from the floor grid at a
few z layers in [-margin, margin] and colour by z, then pass `object_sdf_floor_pts` /
`object_sdf_floor_cols` to `Viewer(...)`.

- [ ] **Step 4: Smoke run** (the object_interaction default; solves TEST-SOCP):

```bash
echo "" | timeout 400 ~/.holonew_deps/miniconda3/envs/holonew/bin/python examples/view_stages.py \
  --task-name sub3_largebox_003 --methods test_socp 2>&1 | tail -20
```
Expected: solves with the `TEST-SOCP` bar, prints the viewer URL, no traceback before it.

- [ ] **Step 5: Commit**

```bash
git add examples/view_stages.py
git commit -m "feat(viz): wire TEST-SOCP interaction data (human/object/floor/G1/directions) into the viewer"
```

---

### Task 8: Docs + full regression

**Files:**
- Modify: `COMMAND.md`
- Test: full viewer/contact suite.

- [ ] **Step 1: Document the Test folder** in COMMAND.md §2 (the GUI folders list): add a "Test" bullet describing SDF Object/Floor, Human/Object/Floor contact, G1 transport, Directions, and the stage gating (human→Grounded, G1→Robot, others→any).

- [ ] **Step 2: Run the suite**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_transport.py tests/test_smplx_probe.py tests/test_retarget_result_interaction.py tests/test_viewer.py tests/test_viewer_object.py tests/test_contact_viz.py -q`
Expected: PASS (all).

- [ ] **Step 3: Commit**

```bash
git add COMMAND.md
git commit -m "docs: document the Test (interaction) viewer folder"
```

---

## Self-review notes
- **Spec coverage:** transport (T1), cache alignment + probe points (T2, T4), result exposure (T3, T4), human/G1 overlays + gating (T5), object/floor/directions/SDF-floor (T6), wiring (T7), docs (T8).
- **Naming consistency:** `human_probe_pts`, `human_obj_dist`, `human_flr_dist`, `human_witness`, `g1_transport_pts`, `human_idx`, `obj_pose` on the result; `human_probe_pts`, `human_dist`, `human_witness`, `g1_transport_pts`, `g1_dist`, `g1_witness`, `contact_fields`, `object_probe_pts`, `floor_probe_pts` on `MethodViz`; handles `_human_handle`, `_g1_transport_handle`, `_object_contact_handle`, `_floor_contact_handle`; toggles `_tog_human/_tog_g1_transport/_tog_object_contact/_tog_floor_contact/_tog_directions/_tog_sdf_floor`.
- **Open items for the implementer (verify before coding):** (a) confirm `make_object_grid` / `make_floor_grid` signatures in `test_socp/contact/probes.py` and that they reproduce the exact N used by the bundled `contact_<task>.npz` (else object/floor channels mismatch — skip+warn); (b) confirm `floor_field` exists in `contact/backends/floor.py` and returns a `ContactField`; (c) the URDF actuated-joint order equals `qpos[7:7+ndof]` (the viewer's `update_cfg(q[7:7+dof])` relies on it — reuse the same).
