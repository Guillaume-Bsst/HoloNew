# Viz Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Poser le socle viz partagé (`core/` + view-model `model.py` + `sources.py`) et porter les 7 couches prod de `viewer.py` vers des `Layer` composables lues via le view-model, câbler `app.py` à parité de comportement avec `viewer.py`, puis **supprimer `viewer.py`**. C'est la PHASE A "Foundation" de la spec `2026-06-30-viz-architecture-design.md` (étapes de migration 1-3). Solve reste un seam non câblé (`solved=None`) : la couche `robot` et le bake solve sont PHASE B.

**Architecture:** Fil rouge **Source → VizFrame → Layers**, orchestré par le `Player`. `core/` (colors, viser_ops, layer, player) est partagé ; `model.py`/`sources.py` sont numpy-only (testables sans écran) ; les `layers/` ne lisent QUE `VizFrame`/`VizContext`/`UiState`. `app.py` instancie `BakeSource → Player → 7 couches → run()`. viser est confiné à `core/viser_ops`, aux `layers/`, au `Player` et à `app.py`.

**Tech Stack:** Python, numpy, viser, scipy (Rotation), pytest. Env python: `~/.holonew_deps/miniconda3/envs/holonew/bin/python`

## Global Constraints
- Code, comments, docstrings in ENGLISH (French only in plan prose / commit messages).
- Contracts = frozen dataclasses, numpy-only (no viser/torch import in `model.py`/`sources.py`).
- viz = pure consumer ; viser confined to `core/viser_ops` + layers + `Player` + `app.py` (imported LAZILY inside functions so the package imports headless).
- Imports **relatifs** dans `src/` ; **absolus** (`from src.…`) dans `tests/`. Le `conftest.py` à la racine `HoloV2/` met `HoloV2/` sur `sys.path` (donc `src.…` et `datapaths` résolvent).
- Tests in `HoloV2/tests/`, run from `HoloV2/` with the env python ; cap `max_frames` very low (mémoire `run-tests-low-max-frames`).
- Invariants de contrat → `raise ValueError` explicite au `__post_init__` (cf. `MultiChannelField`).
- Commits: conventionnels, français, scope `holov2`. **NE JAMAIS tagger Claude** (no `Co-Authored-By`, no Claude/Anthropic mention). Auteur : `Guillaume-Bsst`.

---

## File Structure

```
src/viz/
  __init__.py        # MODIFY : re-export run_app / main
  model.py           # CREATE : VizContext · VizFrame · SolvedFrame (numpy-only)
  sources.py         # CREATE : Source protocol + BakeSource + LiveSource stub
  core/
    __init__.py      # CREATE (empty package marker)
    colors.py        # CREATE : heat_distance · diverging · parity · active_mask · AXIS_COLORS
    viser_ops.py     # CREATE : quat_wxyz_to_R · hide · add_point_cloud/line_segments/label
    layer.py         # CREATE : Layer Protocol + UiState
    player.py        # CREATE : Player (Playback + Selectors + render loop + keep-alive)
  layers/
    __init__.py      # CREATE (empty package marker)
    ground.py        # CREATE : ground_surface_mesh (pure) + GroundLayer (reads ground SDF)
    ghost.py         # CREATE : GhostLayer (SMPL mesh, per-frame re-add)
    skeleton.py      # CREATE : SkeletonLayer
    human_cloud.py   # CREATE : HumanCloudLayer
    objects.py       # CREATE : ObjectsLayer
    fields.py        # CREATE : FieldsLayer (witness + normals)
    style.py         # CREATE : StyleLayer (points + frames + labels)
  app.py             # CREATE : run_app(spec, ...) + main()
  viewer.py          # DELETE (Task 12, after app.py reaches parity)
  _scene_args.py     # REUSE as-is (CLI glue)

tests/
  test_viz_colors.py · test_viz_viser_ops.py · test_viz_model.py · test_viz_layer.py
  test_viz_player.py · test_viz_bake_source.py · test_viz_ground.py · test_viz_layers.py
  test_viz_app.py
  test_viewer_bake.py  # DELETE (Task 12 — superseded by test_viz_bake_source.py + test_viz_app.py)

docs/VIZ.md            # MODIFY (Task 13) : rewrite to the new architecture
docs/ARCHITECTURE.md   # MODIFY (Task 13) : one-line viewer.py -> app.py reference fix
```

**View-model wiring decision (read once before Task 5):** the `Player` owns the `Playback` + `Selectors` GUI and wires the per-frame `render()` to the slider + the three selectors (channel / colour mode / point size). Each `Layer`'s own checkbox is wired LOCALLY in its `setup` to flip its persistent handle's `.visible` — toggling a layer never needs a full render. Because `render()` refreshes EVERY layer's handle on each slider/selector change (regardless of visibility), a paused toggle flips the visibility of the current-frame handle (never stale). This keeps the `Layer` protocol exactly `setup(server, gui, ctx)` / `update(frame, ui)` with no render callback threaded through.

**VizContext note:** the spec's "Contrat du view-model" lists 8 static fields; this phase adds one — `ground_sdf: SDF` — because the spec ALSO mandates the `ground` layer read the real ground channel SDF (migration: "sol box plat → couche ground lit le SDF du canal"). The SDF is the prepare contract (numpy-only), so `model.py` stays viser/torch-free. The `fields` layer derives each channel's `object_idx` from the channel ORDER convention guaranteed by `prepare.runner._validate` (`channels[0]` = ground/world ; channel `c≥1` = object `c-1`/object-local), so no extra context field is needed for it.

---

### Task 1 : `core/colors.py` — colormaps (port dedup ×3)

**Files:**
- Create: `src/viz/core/__init__.py`
- Create: `src/viz/core/colors.py`
- Test: `tests/test_viz_colors.py`

**Interfaces:**
- Produces : `heat_distance(dist, margin) -> (P,3) uint8` · `diverging(signed, vmax) -> (P,3) uint8` · `parity(err, vmax) -> (P,3) uint8` · `active_mask(active) -> (P,3) uint8` · `AXIS_COLORS: (3,3) uint8`.

- [ ] **Step 1 : Write the failing test**

```python
# tests/test_viz_colors.py
"""colors — known input -> known uint8 RGB (ports viewer.py/_heat_distance/_active_colors,
cloud.py/_heat, sdf.py/_diverging into one module)."""
import numpy as np

from src.viz.core.colors import AXIS_COLORS, active_mask, diverging, heat_distance, parity


def test_heat_distance_anchors():
    # d <= 0 -> NEAR (blue) ; d == margin -> FAR (red) ; clamped both ends.
    out = heat_distance(np.array([-1.0, 0.0, 0.05]), 0.05)
    assert out.dtype == np.uint8 and out.shape == (3, 3)
    assert np.array_equal(out[0], [40, 90, 255])     # clamped to NEAR
    assert np.array_equal(out[1], [40, 90, 255])     # d=0 -> NEAR
    assert np.array_equal(out[2], [255, 60, 50])     # d=margin -> FAR


def test_active_mask():
    out = active_mask(np.array([True, False]))
    assert out.dtype == np.uint8
    assert np.array_equal(out[0], [90, 255, 130])    # active -> bright green
    assert np.array_equal(out[1], [70, 70, 80])      # inactive -> dim grey


def test_diverging_white_blue_red():
    out = diverging(np.array([0.0, -1.0, 1.0]), 1.0)
    assert out.dtype == np.uint8
    assert np.array_equal(out[0], [255, 255, 255])   # 0 -> white
    assert np.array_equal(out[1], [51, 89, 255])     # -vmax -> blue
    assert np.array_equal(out[2], [255, 63, 51])     # +vmax -> red


def test_parity_blue_to_red():
    out = parity(np.array([0.0, 0.02]), 0.02)
    assert out.dtype == np.uint8
    assert np.array_equal(out[0], [0, 0, 255])       # err 0 -> blue
    assert np.array_equal(out[1], [255, 0, 0])       # err >= vmax -> red


def test_axis_colors():
    assert AXIS_COLORS.dtype == np.uint8 and AXIS_COLORS.shape == (3, 3)
    assert np.array_equal(AXIS_COLORS[0], [255, 80, 80])
```

- [ ] **Step 2 : Run, expect FAIL**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_colors.py -q`
Expected: FAIL (`ModuleNotFoundError: src.viz.core.colors`).

- [ ] **Step 3 : Create `src/viz/core/__init__.py`** (empty package marker)

```python
"""viz/core — the shared framework (colormaps, viser-confined ops, Layer protocol, Player)."""
```

- [ ] **Step 4 : Create `src/viz/core/colors.py`**

```python
"""Colormaps for the viz layers — the SINGLE source of the heat / diverging / parity / active mappings
(ports the math duplicated across viewer.py / cloud.py / sdf.py). Pure numpy, no viser: a known input
maps to a known uint8 RGB, so each colormap is unit-tested directly. All return (P, 3) uint8."""
from __future__ import annotations

import numpy as np

# distance-heatmap anchors (uint8 RGB), ported from viewer.py:_heat_distance.
_NEAR = np.array([40, 90, 255], np.float64)    # near / penetrating (d <= 0) -> blue
_FAR = np.array([255, 60, 50], np.float64)     # far within margin (d ~ margin) -> red

# per-axis frame colours (x, y, z), ported from viewer.py:_AXIS_COLORS.
AXIS_COLORS = np.array([[255, 80, 80], [80, 230, 80], [80, 130, 255]], np.uint8)


def heat_distance(dist: np.ndarray, margin: float) -> np.ndarray:
    """(P,) signed distance -> (P, 3) uint8. Blue near/penetrating (d<=0) -> red far (d ~ margin)."""
    t = np.clip(np.asarray(dist, np.float64) / max(float(margin), 1e-9), 0.0, 1.0)[:, None]
    return (t * _FAR + (1.0 - t) * _NEAR).astype(np.uint8)


def active_mask(active: np.ndarray) -> np.ndarray:
    """(P,) bool -> (P, 3) uint8. Bright green where active (in the contact band), dim grey elsewhere."""
    a = np.asarray(active, bool)
    col = np.tile(np.array([70, 70, 80], np.uint8), (len(a), 1))
    col[a] = (90, 255, 130)
    return col


def diverging(signed: np.ndarray, vmax: float) -> np.ndarray:
    """(P,) signed value -> (P, 3) uint8. -vmax = blue, 0 = white, +vmax = red (clamped)."""
    t = np.clip(np.asarray(signed, np.float64) / max(float(vmax), 1e-9), -1.0, 1.0)
    col = np.ones((len(t), 3), np.float64)                          # white at t = 0
    neg = t < 0
    a = (-t[neg])[:, None]
    col[neg] = (1 - a) * np.array([1, 1, 1]) + a * np.array([0.20, 0.35, 1.0])    # -> blue
    b = (t[~neg])[:, None]
    col[~neg] = (1 - b) * np.array([1, 1, 1]) + b * np.array([1.0, 0.25, 0.20])   # -> red
    return (col * 255).astype(np.uint8)


def parity(err: np.ndarray, vmax: float) -> np.ndarray:
    """(P,) non-negative error -> (P, 3) uint8. Blue (0) -> red (>= vmax). Ported from cloud.py:_heat."""
    t = np.clip(np.asarray(err, np.float64) / max(float(vmax), 1e-9), 0.0, 1.0)[:, None]
    return (np.concatenate([t, np.zeros_like(t), 1.0 - t], axis=1) * 255).astype(np.uint8)
```

- [ ] **Step 5 : Run, expect PASS (5 tests)**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_colors.py -q`
Expected: PASS.

- [ ] **Step 6 : Commit**

```bash
git add src/viz/core/__init__.py src/viz/core/colors.py tests/test_viz_colors.py
git commit -m "feat(holov2): viz/core/colors — colormaps heat_distance/diverging/parity/active_mask (port dedup ×3)"
```

---

### Task 2 : `core/viser_ops.py` — viser-confined helpers

**Files:**
- Create: `src/viz/core/viser_ops.py`
- Test: `tests/test_viz_viser_ops.py`

**Interfaces:**
- Produces : `quat_wxyz_to_R(quat) -> (L,3,3)` · `hide(handle) -> None` · `add_point_cloud(server, name, points, colors, *, point_size)` · `add_line_segments(server, name, segments, colors, *, line_width)` · `add_label(server, name, text, position)`.

- [ ] **Step 1 : Write the failing test** (pure helpers only — viser confined to the add_* wrappers)

```python
# tests/test_viz_viser_ops.py
"""viser_ops — the pure helpers (quat->R, hide) test without a viser server. hide() is verified
against a duck-typed handle so it needs no screen."""
import numpy as np

from src.viz.core.viser_ops import hide, quat_wxyz_to_R


def test_quat_identity():
    R = quat_wxyz_to_R(np.array([[1.0, 0.0, 0.0, 0.0]]))     # wxyz identity
    assert R.shape == (1, 3, 3)
    assert np.allclose(R[0], np.eye(3), atol=1e-12)


def test_quat_90deg_about_z():
    s = np.sqrt(0.5)
    R = quat_wxyz_to_R(np.array([[s, 0.0, 0.0, s]]))         # 90 deg about +z (wxyz)
    expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(R[0], expected, atol=1e-9)


def test_hide_sets_visible_false():
    class _H:
        visible = True
    h = _H()
    hide(h)
    assert h.visible is False
```

- [ ] **Step 2 : Run, expect FAIL**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_viser_ops.py -q`
Expected: FAIL (`ModuleNotFoundError: src.viz.core.viser_ops`).

- [ ] **Step 3 : Create `src/viz/core/viser_ops.py`**

```python
"""viser-confined helpers for the layers / Player / app — quaternion conversion, the ``hide`` (replaces
the legacy "degenerate triangle at opacity 0" hack), and thin add_* wrappers. The pure helpers
(``quat_wxyz_to_R``, ``hide``) import and test without a screen; viser itself is only touched through
the scene of a ``server`` the caller passes in (the wrappers never import viser)."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as _Rot


def quat_wxyz_to_R(quat: np.ndarray) -> np.ndarray:
    """(L, 4) wxyz quaternions -> (L, 3, 3) rotation matrices (scipy is xyzw). The single quat->R path
    (formerly duplicated ×3 across the viewers)."""
    q = np.asarray(quat, np.float64)
    return _Rot.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()


def hide(handle) -> None:
    """Hide a viser scene handle. Replaces the legacy "add a degenerate triangle at opacity 0" trick:
    every modern viser scene handle exposes ``.visible``, so hiding is a single assignment."""
    handle.visible = False


def add_point_cloud(server, name: str, points, colors, *, point_size: float):
    """Thin wrapper over ``server.scene.add_point_cloud`` (viser-confined). Returns the handle."""
    return server.scene.add_point_cloud(name, np.asarray(points, np.float32),
                                        np.asarray(colors, np.uint8), point_size=float(point_size))


def add_line_segments(server, name: str, segments, colors, *, line_width: float):
    """Thin wrapper over ``server.scene.add_line_segments``. ``segments`` (S, 2, 3), ``colors`` (S, 2, 3)."""
    return server.scene.add_line_segments(name, np.asarray(segments, np.float32),
                                          np.asarray(colors, np.uint8), line_width=float(line_width))


def add_label(server, name: str, text: str, position):
    """Thin wrapper over ``server.scene.add_label`` (viser-confined). Returns the handle."""
    return server.scene.add_label(name, text, position=tuple(float(v) for v in position))
```

- [ ] **Step 4 : Run, expect PASS (3 tests)**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_viser_ops.py -q`
Expected: PASS.

- [ ] **Step 5 : Commit**

```bash
git add src/viz/core/viser_ops.py tests/test_viz_viser_ops.py
git commit -m "feat(holov2): viz/core/viser_ops — quat_wxyz_to_R + hide (tue le hack triangle) + add_* confinés"
```

---

### Task 3 : `model.py` — the view-model (VizContext · VizFrame · SolvedFrame)

**Files:**
- Create: `src/viz/model.py`
- Test: `tests/test_viz_model.py`

**Interfaces (other phases rely on these exact names/fields):**
- Produces : `VizContext(channel_names, margin, style_link_names, smpl_faces, smpl_parents, n_objects, robot_urdf_path, has_solve, ground_sdf)` ; `VizFrame(pose, smpl_verts_world, human_cloud_world, object_clouds_world, human_field, targets, solved)` ; `SolvedFrame(q, object_poses, robot_points_world, link_transforms, style_achieved, contact_achieved, cost, cost_by_term, n_iters, status)`.
- Consumes : `prepare.contracts.SDF`, `targets.contracts.{FramePose, MultiChannelField, FrameTargets, StyleEval, ContactEval}`.

- [ ] **Step 1 : Write the failing test**

```python
# tests/test_viz_model.py
"""view-model contracts — shape validation + the full SolvedFrame field set (phase B fills it; phase A
leaves VizFrame.solved=None). Built from minimal real pipeline contracts (numpy-only, no viser)."""
import numpy as np
import pytest

from src.prepare.contracts import SDF
from src.targets.contracts import (FramePose, FrameTargets, MultiChannelField,
                                    RobotInteractionTargets, EnvironmentInteractionTargets, StyleTargets)
from src.viz.model import SolvedFrame, VizContext, VizFrame


def _sdf() -> SDF:
    return SDF(grid=np.zeros((2, 2, 2)), witness=np.zeros((2, 2, 2, 3)),
               origin=np.zeros(3), spacing=0.1, name="ground")


def _ctx(n_objects: int = 0) -> VizContext:
    return VizContext(
        channel_names=tuple(["ground"] + [f"obj{i}" for i in range(n_objects)]),
        margin=0.05, style_link_names=("a", "b"),
        smpl_faces=np.zeros((4, 3), np.int64), smpl_parents=np.array([-1, 0, 1]),
        n_objects=n_objects, robot_urdf_path=__import__("pathlib").Path("/tmp/g1.urdf"),
        has_solve=False, ground_sdf=_sdf())


def _field(C: int = 1, P: int = 3) -> MultiChannelField:
    return MultiChannelField(
        distance=np.zeros((C, P)), direction=np.zeros((C, P, 3)),
        witness=np.zeros((C, P, 3)), active=np.zeros((C, P), bool),
        channels=tuple(f"c{i}" for i in range(C)))


def _targets() -> FrameTargets:
    style = StyleTargets(link_names=("a", "b"), position=np.zeros((2, 3)),
                         orientation=np.tile([1.0, 0, 0, 0], (2, 1)))
    return FrameTargets(style=style,
                        robot_interaction=RobotInteractionTargets(field=_field()),
                        env_interaction=EnvironmentInteractionTargets(per_object=()),
                        object_rot=np.zeros((0, 3, 3)), object_pos=np.zeros((0, 3)))


def _pose() -> FramePose:
    return FramePose(bone_rot=np.zeros((3, 3, 3)), bone_pos=np.zeros((3, 3)),
                     object_rot=np.zeros((0, 3, 3)), object_pos=np.zeros((0, 3)))


def test_context_ok_and_channel_count():
    ctx = _ctx(n_objects=2)
    assert ctx.channel_names == ("ground", "obj0", "obj1")
    assert ctx.has_solve is False


def test_context_bad_channel_count_raises():
    with pytest.raises(ValueError):
        VizContext(channel_names=("ground",), margin=0.05, style_link_names=(),
                   smpl_faces=np.zeros((4, 3), np.int64), smpl_parents=np.array([-1]),
                   n_objects=2,  # 2 objects but only 1 channel name -> mismatch
                   robot_urdf_path=__import__("pathlib").Path("/x"), has_solve=False, ground_sdf=_sdf())


def test_context_bad_faces_raises():
    with pytest.raises(ValueError):
        VizContext(channel_names=("ground",), margin=0.05, style_link_names=(),
                   smpl_faces=np.zeros((4, 4), np.int64), smpl_parents=np.array([-1]), n_objects=0,
                   robot_urdf_path=__import__("pathlib").Path("/x"), has_solve=False, ground_sdf=_sdf())


def test_vizframe_solved_none_ok():
    fr = VizFrame(pose=_pose(), smpl_verts_world=np.zeros((5, 3), np.float32),
                  human_cloud_world=np.zeros((3, 3), np.float32), object_clouds_world=(),
                  human_field=_field(), targets=_targets(), solved=None)
    assert fr.solved is None and fr.smpl_verts_world.shape == (5, 3)


def test_vizframe_bad_cloud_raises():
    with pytest.raises(ValueError):
        VizFrame(pose=_pose(), smpl_verts_world=None,
                 human_cloud_world=np.zeros((3, 2), np.float32),  # not (N, 3)
                 object_clouds_world=(), human_field=_field(), targets=_targets(), solved=None)


def test_solvedframe_full_field_set():
    sf = SolvedFrame(q=np.zeros(35), object_poses=np.zeros((1, 7)),
                     robot_points_world=np.zeros((10, 3)), link_transforms=np.zeros((4, 4, 4)),
                     style_achieved=None, contact_achieved=None,
                     cost=1.0, cost_by_term={"S-pos": 0.5}, n_iters=3, status="optimal")
    assert sf.q.shape == (35,) and sf.object_poses.shape == (1, 7)
    assert sf.cost_by_term["S-pos"] == 0.5 and sf.status == "optimal"
```

- [ ] **Step 2 : Run, expect FAIL**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_model.py -q`
Expected: FAIL (`ModuleNotFoundError: src.viz.model`).

- [ ] **Step 3 : Create `src/viz/model.py`**

```python
"""View-model contracts OWNED by viz — the seam between the Source and the layers. Frozen, numpy-only
(no viser, no torch import): importable/testable without a screen. ``VizContext`` = static per-scene
assets (handed to each layer at setup); ``VizFrame`` = one shown frame (Source.get); ``SolvedFrame`` =
the post-solve bundle (None until solved). Layers read ONLY these — never the pipeline contracts.

``ground_sdf`` extends the design's 8-field VizContext so the ``ground`` layer renders the REAL ground
channel (plane/terrain) instead of a hardcoded flat box (design migration: "sol box plat -> couche
ground lit le SDF"). ``SDF`` is a numpy-only prepare contract, so this module stays viser/torch-free."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..prepare.contracts import SDF
from ..targets.contracts import (ContactEval, FramePose, FrameTargets, MultiChannelField, StyleEval)


@dataclass(frozen=True)
class VizContext:
    """Static per-scene assets handed to every layer at ``setup``."""

    channel_names: tuple[str, ...]       # ground + N objects, in eval order
    margin: float                        # field activation band (m) — distance-heatmap scale
    style_link_names: tuple[str, ...]    # the L tracked style links (StyleTargets order)
    smpl_faces: np.ndarray               # (F, 3) int — SMPL mesh topology (ghost layer)
    smpl_parents: np.ndarray             # (J,) int — SMPL bone parents (skeleton layer)
    n_objects: int
    robot_urdf_path: Path                # G1 URDF (robot layer, phase B)
    has_solve: bool                      # True once the Source bakes solve -> SolvedFrame
    ground_sdf: SDF                      # ground channel SDF (channels[0]) — ground layer surface

    def __post_init__(self) -> None:
        if self.smpl_faces.ndim != 2 or self.smpl_faces.shape[1] != 3:
            raise ValueError(f"smpl_faces must be (F, 3), got {self.smpl_faces.shape}")
        if self.smpl_parents.ndim != 1:
            raise ValueError(f"smpl_parents must be (J,), got {self.smpl_parents.shape}")
        if len(self.channel_names) != self.n_objects + 1:
            raise ValueError(
                f"channel_names ({len(self.channel_names)}) must be n_objects+1 ({self.n_objects + 1})")


@dataclass(frozen=True)
class SolvedFrame:
    """Post-solve bundle for ONE frame (built by ``BakeSource`` from ``SolveTrajectory`` +
    ``targets.Evaluator``). Full field set defined now; PHASE A leaves ``VizFrame.solved = None``
    (BakeSource fills this in phase B). It READS ``SolveTrajectory`` and REUSES the Evaluator's
    'achieved' — no new retargeting logic."""

    q: np.ndarray                    # (nq,)     solved robot config (SolveTrajectory.qpos[f])
    object_poses: np.ndarray         # (N, 7)    solved object poses (SolveTrajectory.object_poses[f])
    robot_points_world: np.ndarray   # (M, 3)    correspondence points placed by robot FK @ q
    link_transforms: np.ndarray      # (L, 4, 4) link placements (FK) — correspondence/contact layers
    style_achieved: "StyleEval | None"      # ev.style(q)
    contact_achieved: "ContactEval | None"  # ev.contacts(q, object_rot, object_pos)
    cost: float
    cost_by_term: dict               # {term_name: squared residual} (FrameInfo)
    n_iters: int
    status: str


@dataclass(frozen=True)
class VizFrame:
    """One shown frame (``Source.get(i)``) — frozen, numpy-only. The geometric space is fixed ONCE by
    the Source (scene-scaled when solve is present); layers read these world arrays as-is."""

    pose: FramePose                              # bone + object (R, t)        (from FrameTrace)
    smpl_verts_world: np.ndarray | None          # (V, 3) f32 posed SMPL mesh (None if non-parametric)
    human_cloud_world: np.ndarray                # (N, 3) f32 posed human cloud
    object_clouds_world: tuple[np.ndarray, ...]  # per object, (P_i, 3) f32
    human_field: MultiChannelField               # field on the human cloud (PRE-transport)
    targets: FrameTargets                        # style + robot + env references
    solved: "SolvedFrame | None"                 # None tant que non résolu -> solve layers no-op

    def __post_init__(self) -> None:
        if self.human_cloud_world.ndim != 2 or self.human_cloud_world.shape[1] != 3:
            raise ValueError(f"human_cloud_world must be (N, 3), got {self.human_cloud_world.shape}")
        if self.smpl_verts_world is not None and (
                self.smpl_verts_world.ndim != 2 or self.smpl_verts_world.shape[1] != 3):
            raise ValueError(f"smpl_verts_world must be (V, 3)|None, got {self.smpl_verts_world.shape}")
```

- [ ] **Step 4 : Run, expect PASS (7 tests) + numpy-only import**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_model.py -q`
Expected: PASS.

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -c "import sys; import src.viz.model; assert 'viser' not in sys.modules and 'torch' not in sys.modules; print('model numpy-only ok')"`
Expected: `model numpy-only ok`

- [ ] **Step 5 : Commit**

```bash
git add src/viz/model.py tests/test_viz_model.py
git commit -m "feat(holov2): viz/model — VizContext/VizFrame/SolvedFrame (view-model numpy-only, solved=None seam)"
```

---

### Task 4 : `core/layer.py` — Layer Protocol + UiState

**Files:**
- Create: `src/viz/core/layer.py`
- Test: `tests/test_viz_layer.py`

**Interfaces:**
- Produces : `UiState(channel: str, color_mode: str, point_size: float)` (frozen) ; `Layer` Protocol (`folder: str`, `setup(server, gui, ctx: VizContext)`, `update(frame: VizFrame, ui: UiState)`), `@runtime_checkable`.
- Consumes : `model.VizContext`, `model.VizFrame`.

- [ ] **Step 1 : Write the failing test**

```python
# tests/test_viz_layer.py
"""Layer protocol + UiState — frozen selectors, and runtime_checkable structural conformance."""
import dataclasses

import numpy as np
import pytest

from src.viz.core.layer import Layer, UiState


def test_uistate_fields_and_frozen():
    ui = UiState(channel="ground", color_mode="distance", point_size=0.012)
    assert ui.channel == "ground" and ui.color_mode == "distance" and ui.point_size == 0.012
    with pytest.raises(dataclasses.FrozenInstanceError):
        ui.channel = "obj0"


def test_layer_isinstance_structural():
    class Good:
        folder = "X"
        def setup(self, server, gui, ctx): ...
        def update(self, frame, ui): ...

    class Bad:
        folder = "X"
        def setup(self, server, gui, ctx): ...
        # no update

    assert isinstance(Good(), Layer)
    assert not isinstance(Bad(), Layer)
```

- [ ] **Step 2 : Run, expect FAIL**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_layer.py -q`
Expected: FAIL (`ModuleNotFoundError: src.viz.core.layer`).

- [ ] **Step 3 : Create `src/viz/core/layer.py`**

```python
"""Layer protocol + the shared read-only UI selectors. A Layer owns its GUI folder + persistent scene
handles (created once at ``setup``) and, on each ``update``, only refreshes those handles from the
view-model — it never touches another layer. ``@runtime_checkable`` so the app/tests can assert a class
is a Layer (members present)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..model import VizContext, VizFrame


@dataclass(frozen=True)
class UiState:
    """Cross-layer selectors, read-only, assembled by the Player and passed to every ``update``."""

    channel: str        # selected channel name (ground / obj0 / …)
    color_mode: str     # 'uniform' | 'distance' | 'active'
    point_size: float   # cloud point size (m)


@runtime_checkable
class Layer(Protocol):
    folder: str
    def setup(self, server, gui, ctx: VizContext) -> None: ...
    def update(self, frame: VizFrame, ui: UiState) -> None: ...
```

- [ ] **Step 4 : Run, expect PASS (2 tests)**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_layer.py -q`
Expected: PASS.

- [ ] **Step 5 : Commit**

```bash
git add src/viz/core/layer.py tests/test_viz_layer.py
git commit -m "feat(holov2): viz/core/layer — Layer protocol (runtime_checkable) + UiState frozen"
```

---

### Task 5 : `core/player.py` — Player (Playback + Selectors + render loop)

**Files:**
- Create: `src/viz/core/player.py`
- Test: `tests/test_viz_player.py`

**Interfaces:**
- Produces : `Player(source, layers: list[Layer], *, port: int = 8080)` with `.n_frames` property, `.run()`, and the testable `._dispatch(frame, ui)`.
- Consumes : `Source` (duck-typed: `.context`, `.n_frames`, `.get(i)`), `Layer`, `UiState`, viser (lazy).

- [ ] **Step 1 : Write the failing test** (the dispatch fan-out is testable without viser)

```python
# tests/test_viz_player.py
"""Player — the pure dispatch (frame -> each layer.update) is unit-testable without a viser server;
the full run() (folders, daemon loop, keep-alive) is verified by the manual parity check (Task 12)."""
from src.viz.core.layer import UiState
from src.viz.core.player import Player


class _RecLayer:
    folder = "X"
    def __init__(self): self.calls = []
    def setup(self, server, gui, ctx): ...
    def update(self, frame, ui): self.calls.append((frame, ui))


class _FakeSource:
    context = None
    n_frames = 3
    def get(self, i): return f"frame{i}"


def test_n_frames_passthrough():
    assert Player(_FakeSource(), []).n_frames == 3


def test_dispatch_fans_out_to_every_layer():
    l1, l2 = _RecLayer(), _RecLayer()
    p = Player(_FakeSource(), [l1, l2])
    ui = UiState(channel="ground", color_mode="distance", point_size=0.01)
    p._dispatch("FRAME", ui)
    assert l1.calls == [("FRAME", ui)]
    assert l2.calls == [("FRAME", ui)]
```

- [ ] **Step 2 : Run, expect FAIL**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_player.py -q`
Expected: FAIL (`ModuleNotFoundError: src.viz.core.player`).

- [ ] **Step 3 : Create `src/viz/core/player.py`**

```python
"""Player — owns the Playback + Selectors GUI, the per-frame render that dispatches the view-model to
every layer, the play/fps daemon loop and the keep-alive. Factors the boilerplate formerly duplicated
across the four viewers. viser is imported LAZILY in ``run`` so the dispatch logic stays testable.

Wiring contract: ``render()`` is wired to the slider + the three selectors (channel / colour mode /
point size); each layer wires its OWN checkbox locally (in its setup) to flip its handle visibility.
Because render() refreshes EVERY layer's handle on each slider/selector change, a paused layer toggle
flips the visibility of the current-frame handle (never stale)."""
from __future__ import annotations

import threading
import time

from ..model import VizFrame
from .layer import Layer, UiState


class Player:
    """Drives a list of ``Layer`` over a ``Source``. ``run()`` builds the viser server + GUI and serves;
    the pure ``_dispatch`` (frame -> each layer.update) is unit-testable without viser."""

    def __init__(self, source, layers: list[Layer], *, port: int = 8080) -> None:
        self.source = source
        self.layers = list(layers)
        self.port = port

    @property
    def n_frames(self) -> int:
        return self.source.n_frames

    def _dispatch(self, frame: VizFrame, ui: UiState) -> None:
        """Push one (frame, ui) to every layer. No viser, no state — pure fan-out (testable)."""
        for layer in self.layers:
            layer.update(frame, ui)

    def run(self) -> None:
        import viser

        ctx = self.source.context
        srv = viser.ViserServer(port=self.port)
        srv.scene.add_grid("/grid", width=4.0, height=4.0)

        with srv.gui.add_folder("Playback"):
            sld = srv.gui.add_slider("frame", 0, max(self.n_frames - 1, 1), 1, 0)
            play = srv.gui.add_checkbox("play", False)
            fps = srv.gui.add_number("fps", 20, min=1, max=120, step=1)
        with srv.gui.add_folder("Selectors"):
            channel = srv.gui.add_dropdown("channel", ctx.channel_names,
                                           initial_value=ctx.channel_names[0])
            color_mode = srv.gui.add_dropdown("colour mode", ("uniform", "distance", "active"),
                                              initial_value="distance")
            size = srv.gui.add_number("point size", 0.012, min=0.002, max=0.05, step=0.002)
        info = srv.gui.add_markdown("")

        for layer in self.layers:
            layer.setup(srv, srv.gui, ctx)

        def render(_=None):
            i = int(sld.value)
            frame = self.source.get(i)
            ui = UiState(channel=channel.value, color_mode=color_mode.value,
                         point_size=float(size.value))
            self._dispatch(frame, ui)
            info.content = (f"**frame {i + 1}/{self.n_frames}** · channel **{channel.value}** · "
                            f"colour **{color_mode.value}** · margin {ctx.margin:.3f} m")

        for h in (sld, channel, color_mode, size):
            h.on_update(render)
        render()

        def loop():
            while True:
                if play.value:
                    sld.value = (int(sld.value) + 1) % self.n_frames
                    render()
                time.sleep(1.0 / float(fps.value))
        threading.Thread(target=loop, daemon=True).start()
        print(f"viser ready -> http://localhost:{self.port}")
        while True:
            time.sleep(1)
```

> Note: the status markdown drops the old "active probes N/total" line (a per-frame field diagnostic) to keep the Player view-model-agnostic; that diagnostic moves to the `cost_dashboard` panel in phase B.

- [ ] **Step 4 : Run, expect PASS (2 tests)**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_player.py -q`
Expected: PASS.

- [ ] **Step 5 : Commit**

```bash
git add src/viz/core/player.py tests/test_viz_player.py
git commit -m "feat(holov2): viz/core/player — Player (Playback+Selectors+render loop+keep-alive), tue dup ×4"
```

---

### Task 6 : `sources.py` — Source protocol + BakeSource (+ LiveSource stub)

**Files:**
- Create: `src/viz/sources.py`
- Test: `tests/test_viz_bake_source.py`

**Interfaces:**
- Produces : `Source` Protocol (`context: VizContext`, `n_frames: int`, `get(i) -> VizFrame`), `@runtime_checkable` ; `BakeSource(spec, config, *, solve=False, frame_step=2, max_frames=200)` ; `LiveSource` (deferred stub).
- Consumes : `prepare.runner.prepare`, `prepare.config.PrepareConfig`, `prepare.contracts.SceneSpec`, `targets.pipeline.trace_frame`, `model.{VizContext, VizFrame}`.

- [ ] **Step 1 : Write the failing test** (real HODome data, skip-guarded — mirrors `test_viewer_bake.py`)

```python
# tests/test_viz_bake_source.py
"""BakeSource — the HEADLESS view-model bake on REAL HODome data (no viser server). Determinism
(build ×2 identical), VizFrame shapes/dtypes (float32 world clouds, solved=None), and the solve=True
seam (NotImplementedError until phase B). Same skip-guard / spec as test_viewer_bake.py."""
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.prepare.config import PrepareConfig
from src.prepare.contracts import RobotSpec, SceneSpec
from src.viz.model import VizContext, VizFrame
from src.viz.sources import BakeSource, Source
from datapaths import HODOME as _HODOME, SMPLX_MODELS as _SMPLX
_CORR = Path(__file__).resolve().parent.parent / "cache" / "correspondence" / "corr_neutral.npz"
_URDF = Path(__file__).resolve().parent.parent / "models" / "g1" / "g1_29dof.urdf"


def _pick() -> Path | None:
    sm, ob = _HODOME / "smplx", _HODOME / "object"
    if not (sm.is_dir() and ob.is_dir() and _SMPLX.is_dir() and _CORR.exists()):
        return None
    shared = {p.stem for p in sm.glob("*.npz")} & {p.stem for p in ob.glob("*.npz")}
    return sm / f"{sorted(shared)[0]}.npz" if shared else None


_SEQ = _pick()
_SKIP = pytest.mark.skipif(_SEQ is None, reason="HODome data / SMPL-X model / corr_neutral.npz absent")


def _robot() -> RobotSpec:
    return RobotSpec(name="g1", urdf_path=_URDF, link_names=("pelvis",), dof=29, height=1.3)


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    c = tmp_path_factory.mktemp("viz_bake_cache")
    (c / "correspondence").mkdir(parents=True, exist_ok=True)
    shutil.copy(_CORR, c / "correspondence" / "corr_neutral.npz")
    return c


def _spec(cache) -> SceneSpec:
    return SceneSpec(dataset="hodome", motion_path=_SEQ, robot=_robot(),
                     smpl_model_dir=_SMPLX, cache_dir=cache)


@_SKIP
def test_source_protocol_and_context(cache):
    src = BakeSource(_spec(cache), PrepareConfig(), solve=False, frame_step=8, max_frames=3)
    assert isinstance(src, Source)
    assert isinstance(src.context, VizContext)
    assert src.context.has_solve is False
    assert len(src.context.channel_names) == src.context.n_objects + 1
    assert src.context.channel_names[0] == "ground"
    assert 1 <= src.n_frames <= 3


@_SKIP
def test_vizframe_shapes_dtypes_solved_none(cache):
    src = BakeSource(_spec(cache), PrepareConfig(), solve=False, frame_step=8, max_frames=3)
    ctx = src.context
    for i in range(src.n_frames):
        fr = src.get(i)
        assert isinstance(fr, VizFrame)
        assert fr.solved is None
        assert fr.human_cloud_world.dtype == np.float32 and fr.human_cloud_world.shape[1] == 3
        assert fr.smpl_verts_world is not None and fr.smpl_verts_world.dtype == np.float32
        assert len(fr.object_clouds_world) == ctx.n_objects
        for oc in fr.object_clouds_world:
            assert oc.dtype == np.float32 and oc.shape[1] == 3
        # field is channel-first (C, N), aligned with the context channels.
        assert fr.human_field.distance.shape[0] == len(ctx.channel_names)
        assert np.isfinite(fr.human_cloud_world).all()


@_SKIP
def test_determinism_build_twice_identical(cache):
    a = BakeSource(_spec(cache), PrepareConfig(), solve=False, frame_step=8, max_frames=3)
    b = BakeSource(_spec(cache), PrepareConfig(), solve=False, frame_step=8, max_frames=3)
    assert a.n_frames == b.n_frames
    for i in range(a.n_frames):
        fa, fb = a.get(i), b.get(i)
        assert np.array_equal(fa.human_cloud_world, fb.human_cloud_world)
        assert np.array_equal(fa.smpl_verts_world, fb.smpl_verts_world)
        assert np.array_equal(fa.human_field.distance, fb.human_field.distance)
        for oa, ob in zip(fa.object_clouds_world, fb.object_clouds_world):
            assert np.array_equal(oa, ob)


@_SKIP
def test_solve_true_not_implemented(cache):
    with pytest.raises(NotImplementedError):
        BakeSource(_spec(cache), PrepareConfig(), solve=True, frame_step=8, max_frames=2)
```

- [ ] **Step 2 : Run, expect FAIL**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_bake_source.py -q`
Expected: FAIL (`ModuleNotFoundError: src.viz.sources`) — or all-skip if HODome data is absent on this machine (then verify the import path with Step 4's numpy-only check instead).

- [ ] **Step 3 : Create `src/viz/sources.py`**

```python
"""Sources — the view-model providers. ``Source`` protocol (``get`` / ``n_frames`` / ``context``);
``BakeSource`` runs ``prepare`` once and bakes a ``VizFrame`` per shown frame (smooth offline playback);
``LiveSource`` (on-the-fly) is deferred by the design. numpy-only (no viser): drives the public pipeline
surface (``prepare.runner`` + ``targets.pipeline.trace_frame``) and bundles the result into the
view-model. The geometric space is fixed HERE in one place (the design's seam decision)."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..prepare.config import PrepareConfig
from ..prepare.contracts import SceneSpec
from ..prepare.runner import prepare
from ..targets.pipeline import trace_frame
from .model import VizContext, VizFrame


@runtime_checkable
class Source(Protocol):
    context: VizContext
    n_frames: int
    def get(self, i: int) -> VizFrame: ...


class BakeSource:
    """Bakes every shown frame's ``VizFrame`` up front (offline) for fluid playback. ``solve=False``
    leaves ``VizFrame.solved = None`` (pre-solve works with no solver); ``solve=True`` (the
    ``SolveTrajectory`` + ``Evaluator`` -> ``SolvedFrame`` bake) arrives in phase B."""

    def __init__(self, spec: SceneSpec, config: PrepareConfig, *, solve: bool = False,
                 frame_step: int = 2, max_frames: int = 200) -> None:
        if solve:
            raise NotImplementedError("solve baking (SolvedFrame) arrives in phase B; use solve=False")
        grounded, ctx = prepare(spec, config)
        body = grounded.body
        if body is None:
            raise ValueError("the bake source needs a parametric body (SMPL params)")

        shown = list(range(0, grounded.n_frames, frame_step))[:max_frames]
        traces = [trace_frame(grounded, ctx, spec.robot, f) for f in shown]
        verts = [body.posed_vertices(grounded.smpl_params, f).astype(np.float32) for f in shown]

        self._context = VizContext(
            channel_names=ctx.channel_names,
            margin=float(ctx.margin),
            style_link_names=traces[0].targets.style.link_names,
            smpl_faces=np.asarray(body.faces),
            smpl_parents=np.asarray(body.parents),
            n_objects=grounded.n_objects,
            robot_urdf_path=spec.robot.urdf_path,
            has_solve=False,
            ground_sdf=ctx.channels[0].sdf,
        )
        self._frames = [
            VizFrame(
                pose=tr.pose,
                smpl_verts_world=v,
                human_cloud_world=np.asarray(tr.human_cloud_world, np.float32),
                object_clouds_world=tuple(np.asarray(oc, np.float32)
                                          for oc in tr.object_clouds_world),
                human_field=tr.human_field,
                targets=tr.targets,
                solved=None,
            )
            for tr, v in zip(traces, verts)
        ]

    @property
    def context(self) -> VizContext:
        return self._context

    @property
    def n_frames(self) -> int:
        return len(self._frames)

    def get(self, i: int) -> VizFrame:
        return self._frames[int(i)]


class LiveSource:
    """Deferred by the redesign ("LiveSource … plus tard"): would run ``trace_frame`` (+ solve) on the
    fly behind the SAME ``Source`` interface for live teleop. Not built in this phase."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("LiveSource is deferred by the viz redesign; use BakeSource")
```

- [ ] **Step 4 : Run, expect PASS (4 tests, or skip if no data) + numpy-only import**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_bake_source.py -q`
Expected: PASS (4) if HODome data present; otherwise 4 skipped — both are green.

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -c "import sys; import src.viz.sources; assert 'viser' not in sys.modules; print('sources viser-free ok')"`
Expected: `sources viser-free ok`

- [ ] **Step 5 : Commit**

```bash
git add src/viz/sources.py tests/test_viz_bake_source.py
git commit -m "feat(holov2): viz/sources — Source protocol + BakeSource (solve=False, solved=None) + LiveSource stub"
```

---

### Task 7 : `layers/ground.py` — ground surface from the SDF (tested helper + layer)

**Files:**
- Create: `src/viz/layers/__init__.py`
- Create: `src/viz/layers/ground.py`
- Test: `tests/test_viz_ground.py`

**Interfaces:**
- Produces : `ground_surface_mesh(sdf) -> (verts (Nx*Ny,3) f32, faces (M,3) int64)` (pure) ; `GroundLayer` (`folder = "Static"`).
- Consumes : `VizContext.ground_sdf`, `model.{VizContext, VizFrame}`, `core.layer.UiState`.

- [ ] **Step 1 : Write the failing test** (the pure heightfield helper)

```python
# tests/test_viz_ground.py
"""ground_surface_mesh — SDF heightfield extraction (pure). A flat-plane SDF -> a flat quad at the
plane height over the grid extent (the design's "ground reads the SDF, not a hardcoded box")."""
import numpy as np

from src.prepare.contracts import SDF
from src.viz.core.layer import Layer
from src.viz.layers.ground import GroundLayer, ground_surface_mesh


def _plane_sdf(plane_z: float) -> SDF:
    """A 3x3xNz grid whose signed distance is (z - plane_z): zero-crossing at z = plane_z."""
    nx = ny = 3
    nz = 5
    spacing = 0.1
    origin = np.array([0.0, 0.0, -0.2])
    zs = origin[2] + spacing * np.arange(nz)                     # -0.2 .. 0.2
    grid = np.zeros((nx, ny, nz))
    for k in range(nz):
        grid[:, :, k] = zs[k] - plane_z                          # distance to the plane along z
    witness = np.zeros((nx, ny, nz, 3))
    return SDF(grid=grid, witness=witness, origin=origin, spacing=spacing, name="ground")


def test_flat_plane_heightfield():
    verts, faces = ground_surface_mesh(_plane_sdf(0.0))
    assert verts.shape == (9, 3) and verts.dtype == np.float32
    assert np.allclose(verts[:, 2], 0.0, atol=1e-6)              # all at the plane height
    assert faces.shape == (8, 3)                                 # 2 tris * (3-1)*(3-1) cells
    assert faces.dtype == np.int64 and faces.max() < 9


def test_plane_at_offset_height():
    verts, _ = ground_surface_mesh(_plane_sdf(0.1))
    assert np.allclose(verts[:, 2], 0.1, atol=1e-6)


def test_ground_layer_is_a_layer():
    assert isinstance(GroundLayer(), Layer)
    assert GroundLayer().folder == "Static"
```

- [ ] **Step 2 : Run, expect FAIL**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_ground.py -q`
Expected: FAIL (`ModuleNotFoundError: src.viz.layers.ground`).

- [ ] **Step 3 : Create `src/viz/layers/__init__.py`** (empty package marker)

```python
"""viz/layers — composable prod layers (one file, one toggle each), reading only the view-model."""
```

- [ ] **Step 4 : Create `src/viz/layers/ground.py`**

```python
"""Ground layer — renders the REAL ground channel surface from its SDF (plane OR terrain), replacing the
legacy hardcoded flat box at z=0. ``ground_surface_mesh`` is a pure helper (SDF -> heightfield) so it is
unit-tested directly; the layer only adds the mesh once at setup and toggles its visibility."""
from __future__ import annotations

import numpy as np

from ..core.layer import UiState
from ..model import VizContext, VizFrame


def ground_surface_mesh(sdf) -> tuple[np.ndarray, np.ndarray]:
    """``SDF`` -> (verts (Nx*Ny, 3) f32, faces (M, 3) int64). Heightfield: per (x, y) column, the surface
    z is the zero-crossing of the signed distance along z (linear interp between the bracketing nodes;
    fallback = the |d|-min node). A flat plane SDF -> a flat quad over its extent; a terrain SDF -> its
    surface. Reads the SDF only (pure, no viser)."""
    grid = np.asarray(sdf.grid, np.float64)                       # (Nx, Ny, Nz)
    nx, ny, nz = grid.shape
    ox, oy, oz = (float(v) for v in sdf.origin)
    h = float(sdf.spacing)
    xs = ox + h * np.arange(nx)
    ys = oy + h * np.arange(ny)
    zs = oz + h * np.arange(nz)

    zsurf = np.empty((nx, ny), np.float64)
    for i in range(nx):
        for j in range(ny):
            d = grid[i, j]                                        # (Nz,) signed distance along z
            sign = np.signbit(d)
            cross = np.where(sign[:-1] != sign[1:])[0]           # zero-crossing intervals
            if len(cross):
                k = int(cross[0])
                d0, d1 = d[k], d[k + 1]
                t = 0.0 if d1 == d0 else d0 / (d0 - d1)           # interp to d == 0
                zsurf[i, j] = zs[k] + t * h
            else:
                zsurf[i, j] = zs[int(np.argmin(np.abs(d)))]       # no crossing -> nearest-to-surface node

    gx, gy = np.meshgrid(xs, ys, indexing="ij")                 # (Nx, Ny)
    verts = np.stack([gx, gy, zsurf], axis=-1).reshape(-1, 3).astype(np.float32)

    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a = i * ny + j
            b = a + 1
            c = a + ny
            d_ = c + 1
            faces.append([a, c, b])
            faces.append([b, c, d_])
    faces_arr = np.asarray(faces, np.int64) if faces else np.zeros((0, 3), np.int64)
    return verts, faces_arr


class GroundLayer:
    """Static ground surface read from the ground channel SDF (added once; checkbox toggles it)."""

    folder = "Static"

    def setup(self, server, gui, ctx: VizContext) -> None:
        verts, faces = ground_surface_mesh(ctx.ground_sdf)
        self._handle = server.scene.add_mesh_simple(
            "/ground", verts, faces, color=(170, 170, 178), side="double")
        self._cb = gui.add_checkbox("ground", True)
        self._cb.on_update(lambda _: setattr(self._handle, "visible", self._cb.value))

    def update(self, frame: VizFrame, ui: UiState) -> None:
        self._handle.visible = self._cb.value
```

- [ ] **Step 5 : Run, expect PASS (3 tests)**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_ground.py -q`
Expected: PASS.

- [ ] **Step 6 : Commit**

```bash
git add src/viz/layers/__init__.py src/viz/layers/ground.py tests/test_viz_ground.py
git commit -m "feat(holov2): viz/layers/ground — surface depuis le SDF du canal ground (plan/terrain, plus de box plat)"
```

---

### Task 8 : `layers/ghost.py` + `layers/skeleton.py` (thin ports)

**Files:**
- Create: `src/viz/layers/ghost.py`
- Create: `src/viz/layers/skeleton.py`
- Test: `tests/test_viz_layers.py` (created here; extended by Tasks 9-10)

**Interfaces:**
- Produces : `GhostLayer` (`folder = "Static"`) ; `SkeletonLayer` (`folder = "Skeleton"`).
- Consumes : `VizFrame.smpl_verts_world`, `VizContext.smpl_faces` (ghost) ; `VizFrame.pose.bone_pos`, `VizContext.smpl_parents` (skeleton) ; `core.viser_ops`.

These are THIN layers (per the spec's testing section, `update()` is almost pure handle assignment — nothing to unit-test at the layer level). Deliverable: the classes + a STRUCTURAL test (`isinstance(layer, Layer)` + `folder`). Full visual parity is the consolidated manual check in Task 12.

- [ ] **Step 1 : Write the failing test**

```python
# tests/test_viz_layers.py
"""Thin layers — structural conformance (each is a Layer with the right folder). The per-pixel render
is verified by the manual parity check (Task 12); update() is near-pure handle assignment so there is
no meaningful per-layer unit test (design's testing section)."""
import pytest

from src.viz.core.layer import Layer
from src.viz.layers.ghost import GhostLayer
from src.viz.layers.skeleton import SkeletonLayer


@pytest.mark.parametrize("cls, folder", [
    (GhostLayer, "Static"),
    (SkeletonLayer, "Skeleton"),
])
def test_layer_structural(cls, folder):
    layer = cls()
    assert isinstance(layer, Layer)
    assert layer.folder == folder
```

- [ ] **Step 2 : Run, expect FAIL**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_layers.py -q`
Expected: FAIL (`ModuleNotFoundError: src.viz.layers.ghost`).

- [ ] **Step 3 : Create `src/viz/layers/ghost.py`**

```python
"""Ghost layer — the SMPL mesh as a translucent backdrop. Per-frame (the verts change every frame), so
it re-adds the mesh in ``update`` (the design's justified re-add exception); the checkbox toggles it."""
from __future__ import annotations

import numpy as np

from ..core.layer import UiState
from ..model import VizContext, VizFrame


class GhostLayer:
    folder = "Static"

    def setup(self, server, gui, ctx: VizContext) -> None:
        self._server = server
        self._faces = np.asarray(ctx.smpl_faces)
        self._handle = None
        self._cb = gui.add_checkbox("SMPL ghost", True)
        self._cb.on_update(lambda _: self._set_visible())

    def _set_visible(self) -> None:
        if self._handle is not None:
            self._handle.visible = self._cb.value

    def update(self, frame: VizFrame, ui: UiState) -> None:
        if frame.smpl_verts_world is None:                       # non-parametric source -> no mesh
            return
        self._handle = self._server.scene.add_mesh_simple(
            "/ghost", np.asarray(frame.smpl_verts_world, np.float32), self._faces,
            color=(200, 200, 210), opacity=0.45, side="double")
        self._handle.visible = self._cb.value
```

- [ ] **Step 4 : Create `src/viz/layers/skeleton.py`**

```python
"""Skeleton layer — SMPL bones as parent->child line segments from ``pose.bone_pos`` (persistent
handle, refreshed per frame — no re-add)."""
from __future__ import annotations

import numpy as np

from ..core import viser_ops
from ..core.layer import UiState
from ..model import VizContext, VizFrame


class SkeletonLayer:
    folder = "Skeleton"

    def setup(self, server, gui, ctx: VizContext) -> None:
        parents = np.asarray(ctx.smpl_parents)
        self._pairs = [(int(parents[j]), j) for j in range(len(parents)) if parents[j] >= 0]
        n = max(len(self._pairs), 1)
        seg0 = np.zeros((n, 2, 3), np.float32)
        col = np.tile([[[0, 120, 255]]], (n, 2, 1)).astype(np.uint8)
        self._handle = viser_ops.add_line_segments(server, "/skeleton", seg0, col, line_width=3.0)
        self._cb = gui.add_checkbox("skeleton", True)
        self._cb.on_update(lambda _: setattr(self._handle, "visible", self._cb.value))

    def update(self, frame: VizFrame, ui: UiState) -> None:
        if self._pairs:
            bp = np.asarray(frame.pose.bone_pos, np.float32)
            seg = np.stack([np.stack([bp[a], bp[b]]) for a, b in self._pairs]).astype(np.float32)
            self._handle.points = seg                            # viser LineSegmentsHandle.points
        self._handle.visible = self._cb.value
```

> Note for the implementer: viser's `LineSegmentsHandle` exposes writable `.points` / `.colors`. If the installed viser version rejects the setter, fall back to re-`viser_ops.add_line_segments` per frame (viewer.py's proven behaviour) — the manual parity check (Task 12) is the gate.

- [ ] **Step 5 : Run, expect PASS (2 tests)**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_layers.py -q`
Expected: PASS.

- [ ] **Step 6 : Commit**

```bash
git add src/viz/layers/ghost.py src/viz/layers/skeleton.py tests/test_viz_layers.py
git commit -m "feat(holov2): viz/layers/ghost+skeleton — port depuis viewer.py (lecture VizFrame seule)"
```

---

### Task 9 : `layers/human_cloud.py` + `layers/objects.py` (thin ports)

**Files:**
- Create: `src/viz/layers/human_cloud.py`
- Create: `src/viz/layers/objects.py`
- Test: `tests/test_viz_layers.py` (append cases)

**Interfaces:**
- Produces : `HumanCloudLayer` (`folder = "Interaction - human"`) ; `ObjectsLayer` (`folder = "Static"`).
- Consumes : `VizFrame.human_cloud_world` + `human_field` (human) ; `VizFrame.object_clouds_world` + `targets.env_interaction.per_object` (objects) ; `UiState`, `core.colors`, `core.viser_ops`, `VizContext.{channel_names, margin, n_objects}`.

- [ ] **Step 1 : Extend the test** (append to `tests/test_viz_layers.py`)

```python
# tests/test_viz_layers.py  (append imports + add to the parametrize list)
from src.viz.layers.human_cloud import HumanCloudLayer
from src.viz.layers.objects import ObjectsLayer

#   (GhostLayer, "Static"),
#   (SkeletonLayer, "Skeleton"),
    (HumanCloudLayer, "Interaction - human"),
    (ObjectsLayer, "Static"),
```

- [ ] **Step 2 : Run, expect FAIL**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_layers.py -q`
Expected: FAIL (`ModuleNotFoundError: src.viz.layers.human_cloud`).

- [ ] **Step 3 : Create `src/viz/layers/human_cloud.py`**

```python
"""Human cloud layer — the posed human cloud coloured by the SELECTED channel of ``human_field``
(uniform / distance heatmap / active mask). Persistent point-cloud handle, refreshed per frame."""
from __future__ import annotations

import numpy as np

from ..core import colors, viser_ops
from ..core.layer import UiState
from ..model import VizContext, VizFrame


class HumanCloudLayer:
    folder = "Interaction - human"

    def setup(self, server, gui, ctx: VizContext) -> None:
        self._channel_names = ctx.channel_names
        self._margin = ctx.margin
        self._handle = viser_ops.add_point_cloud(
            server, "/human", np.zeros((1, 3), np.float32), np.zeros((1, 3), np.uint8),
            point_size=0.012)
        self._cb = gui.add_checkbox("human cloud", True)
        self._cb.on_update(lambda _: setattr(self._handle, "visible", self._cb.value))

    def update(self, frame: VizFrame, ui: UiState) -> None:
        c = self._channel_names.index(ui.channel)
        field = frame.human_field
        if ui.color_mode == "distance":
            col = colors.heat_distance(field.distance[c], self._margin)
        elif ui.color_mode == "active":
            col = colors.active_mask(field.active[c])
        else:                                                    # uniform
            col = np.tile(np.array([185, 185, 195], np.uint8),
                          (frame.human_cloud_world.shape[0], 1))
        self._handle.points = np.asarray(frame.human_cloud_world, np.float32)
        self._handle.colors = col
        self._handle.point_size = float(ui.point_size)
        self._handle.visible = self._cb.value
```

- [ ] **Step 4 : Create `src/viz/layers/objects.py`**

```python
"""Objects layer — the posed object clouds, coloured by their OWN env field
(``targets.env_interaction``) on the selected channel (pick ``ground`` to see an object rest on the
floor). One persistent handle per object."""
from __future__ import annotations

import numpy as np

from ..core import colors, viser_ops
from ..core.layer import UiState
from ..model import VizContext, VizFrame


class ObjectsLayer:
    folder = "Static"

    def setup(self, server, gui, ctx: VizContext) -> None:
        self._channel_names = ctx.channel_names
        self._margin = ctx.margin
        self._handles = [
            viser_ops.add_point_cloud(server, f"/obj{k}", np.zeros((1, 3), np.float32),
                                      np.zeros((1, 3), np.uint8), point_size=0.012)
            for k in range(ctx.n_objects)]
        self._cb = gui.add_checkbox("object clouds", True)
        self._cb.on_update(lambda _: [setattr(h, "visible", self._cb.value) for h in self._handles])

    def update(self, frame: VizFrame, ui: UiState) -> None:
        c = self._channel_names.index(ui.channel)
        env = frame.targets.env_interaction.per_object
        for k, h in enumerate(self._handles):
            pts = np.asarray(frame.object_clouds_world[k], np.float32)
            if ui.color_mode == "distance":
                col = colors.heat_distance(env[k].distance[c], self._margin)
            elif ui.color_mode == "active":
                col = colors.active_mask(env[k].active[c])
            else:                                                # uniform orange
                col = np.tile(np.array([255, 140, 0], np.uint8), (pts.shape[0], 1))
            h.points = pts
            h.colors = col
            h.point_size = float(ui.point_size)
            h.visible = self._cb.value
```

- [ ] **Step 5 : Run, expect PASS (4 tests)**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_layers.py -q`
Expected: PASS.

- [ ] **Step 6 : Commit**

```bash
git add src/viz/layers/human_cloud.py src/viz/layers/objects.py tests/test_viz_layers.py
git commit -m "feat(holov2): viz/layers/human_cloud+objects — nuages colorés par champ (port viewer.py)"
```

---

### Task 10 : `layers/fields.py` + `layers/style.py` (thin ports)

**Files:**
- Create: `src/viz/layers/fields.py`
- Create: `src/viz/layers/style.py`
- Test: `tests/test_viz_layers.py` (append cases)

**Interfaces:**
- Produces : `FieldsLayer` (`folder = "Interaction - human"`) ; `StyleLayer` (`folder = "Style targets"`).
- Consumes : `VizFrame.human_field` + `human_cloud_world` + `pose.object_*` (fields) ; `VizFrame.targets.style` (style) ; `VizContext.{channel_names, style_link_names}`, `core.{colors, viser_ops}`.

`fields` maps object-local witness/direction to world using the channel-ORDER convention guaranteed by `prepare.runner._validate`: channel 0 = ground (world frame, no map) ; channel `c≥1` = object `c-1` (object-local → map by that object's per-frame `(R, t)`).

- [ ] **Step 1 : Extend the test** (append to `tests/test_viz_layers.py`)

```python
# tests/test_viz_layers.py  (append imports + add to the parametrize list)
from src.viz.layers.fields import FieldsLayer
from src.viz.layers.style import StyleLayer

    (FieldsLayer, "Interaction - human"),
    (StyleLayer, "Style targets"),
```

- [ ] **Step 2 : Run, expect FAIL**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_layers.py -q`
Expected: FAIL (`ModuleNotFoundError: src.viz.layers.fields`).

- [ ] **Step 3 : Create `src/viz/layers/fields.py`**

```python
"""Fields layer — witness lines (point -> nearest surface) and normals (short segment along the contact
direction) for the ACTIVE probes of the selected channel. Object channels store witness/direction in the
object-LOCAL frame, so they are mapped to world by that object's per-frame (R, t); the ground channel is
already world. Channel ordering convention (prepare.runner._validate): channel 0 = ground (world);
channel c>=1 = object c-1 (object-local)."""
from __future__ import annotations

import numpy as np

from ..core import viser_ops
from ..core.layer import UiState
from ..model import VizContext, VizFrame

_MAX_SEG = 400


class FieldsLayer:
    folder = "Interaction - human"

    def setup(self, server, gui, ctx: VizContext) -> None:
        self._channel_names = ctx.channel_names
        z, zc = np.zeros((1, 2, 3), np.float32), np.zeros((1, 2, 3), np.uint8)
        self._wit = viser_ops.add_line_segments(server, "/witness", z, zc, line_width=1.5)
        self._nrm = viser_ops.add_line_segments(server, "/normals", z, zc, line_width=2.0)
        self._cb_w = gui.add_checkbox("witness lines", False)
        self._cb_n = gui.add_checkbox("normals", False)
        self._cb_w.on_update(lambda _: setattr(self._wit, "visible", self._cb_w.value))
        self._cb_n.on_update(lambda _: setattr(self._nrm, "visible", self._cb_n.value))

    def update(self, frame: VizFrame, ui: UiState) -> None:
        c = self._channel_names.index(ui.channel)
        field = frame.human_field
        idx = np.where(np.asarray(field.active[c], bool))[0]
        want = self._cb_w.value or self._cb_n.value
        if len(idx) and want:
            if len(idx) > _MAX_SEG:
                idx = np.random.default_rng(0).choice(idx, _MAX_SEG, replace=False)
            pts = np.asarray(frame.human_cloud_world, np.float64)[idx]   # (S, 3) world
            wit = np.asarray(field.witness[c], np.float64)[idx]          # (S, 3) channel-local
            dirn = np.asarray(field.direction[c], np.float64)[idx]       # (S, 3) channel-local
            object_idx = None if c == 0 else c - 1                       # ground=world; obj=object-local
            if object_idx is not None:
                R = np.asarray(frame.pose.object_rot[object_idx], np.float64)
                t = np.asarray(frame.pose.object_pos[object_idx], np.float64)
                wit = wit @ R.T + t
                dirn = dirn @ R.T
        else:
            pts = wit = dirn = np.zeros((0, 3))

        if self._cb_w.value and len(pts):
            seg = np.stack([pts, wit], axis=1).astype(np.float32)
            self._wit.points = seg
            self._wit.colors = np.tile([[[230, 230, 60]]], (len(pts), 2, 1)).astype(np.uint8)
        self._wit.visible = self._cb_w.value and len(pts) > 0

        if self._cb_n.value and len(pts):
            seg = np.stack([pts, pts + dirn * 0.05], axis=1).astype(np.float32)
            self._nrm.points = seg
            self._nrm.colors = np.tile([[[60, 220, 200]]], (len(pts), 2, 1)).astype(np.uint8)
        self._nrm.visible = self._cb_n.value and len(pts) > 0
```

- [ ] **Step 4 : Create `src/viz/layers/style.py`**

```python
"""Style layer — the L StyleTargets link targets (the KEY validation layer): points at
``style.position`` (uniform orange) + per-link orientation frames (3 short xyz axes from
``style.orientation`` wxyz) + per-link name labels. Persistent handles; frames re-pushed per frame."""
from __future__ import annotations

import numpy as np

from ..core import colors, viser_ops
from ..core.layer import UiState
from ..model import VizContext, VizFrame

_AXIS_LEN = 0.08


class StyleLayer:
    folder = "Style targets"

    def setup(self, server, gui, ctx: VizContext) -> None:
        self._links = ctx.style_link_names
        L = len(self._links)
        self._pts = viser_ops.add_point_cloud(
            server, "/style_pts", np.zeros((max(L, 1), 3), np.float32),
            np.tile(np.array([255, 170, 0], np.uint8), (max(L, 1), 1)), point_size=0.03)
        self._frames = viser_ops.add_line_segments(
            server, "/style_frames", np.zeros((1, 2, 3), np.float32),
            np.zeros((1, 2, 3), np.uint8), line_width=2.5)
        self._labels = [viser_ops.add_label(server, f"/style_label/{name}", name, (0.0, 0.0, 0.0))
                        for name in self._links]
        self._cb_p = gui.add_checkbox("link points", True)
        self._cb_f = gui.add_checkbox("orientation frames", True)
        self._cb_l = gui.add_checkbox("link labels", False)
        self._cb_p.on_update(lambda _: setattr(self._pts, "visible", self._cb_p.value))
        self._cb_f.on_update(lambda _: setattr(self._frames, "visible", self._cb_f.value))
        self._cb_l.on_update(lambda _: [setattr(h, "visible", self._cb_l.value) for h in self._labels])

    def update(self, frame: VizFrame, ui: UiState) -> None:
        style = frame.targets.style
        pos = np.asarray(style.position, np.float32)             # (L, 3)
        self._pts.points = pos
        self._pts.colors = np.tile(np.array([255, 170, 0], np.uint8), (len(pos), 1))
        self._pts.point_size = max(float(ui.point_size) * 2.0, 0.02)
        self._pts.visible = self._cb_p.value

        if self._cb_f.value and style.orientation is not None:
            rots = viser_ops.quat_wxyz_to_R(style.orientation)  # (L, 3, 3)
            segs, cols = [], []
            for i in range(len(self._links)):
                for a in range(3):
                    d = rots[i][:, a]                            # world dir of body axis a
                    segs.append([pos[i], pos[i] + d * _AXIS_LEN])
                    cols.append([colors.AXIS_COLORS[a], colors.AXIS_COLORS[a]])
            self._frames.points = np.asarray(segs, np.float32)
            self._frames.colors = np.asarray(cols, np.uint8)
            self._frames.visible = True
        else:
            self._frames.visible = False

        for i, h in enumerate(self._labels):
            h.position = tuple(float(v) for v in pos[i])
            h.visible = self._cb_l.value
```

- [ ] **Step 5 : Run, expect PASS (6 tests)**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_layers.py -q`
Expected: PASS.

- [ ] **Step 6 : Commit**

```bash
git add src/viz/layers/fields.py src/viz/layers/style.py tests/test_viz_layers.py
git commit -m "feat(holov2): viz/layers/fields+style — witness/normales + cibles de style (port viewer.py)"
```

---

### Task 11 : `app.py` + `__init__.py` — wire the unified prod viewer

**Files:**
- Create: `src/viz/app.py`
- Modify: `src/viz/__init__.py`
- Test: `tests/test_viz_app.py`

**Interfaces:**
- Produces : `run_app(spec, *, port=8080, frame_step=2, max_frames=200, solve=False) -> None` ; `main() -> None` ; `viz.run_app` / `viz.main` re-exports.
- Consumes : `sources.BakeSource`, `core.player.Player`, the 7 layers, `_scene_args.{add_scene_args, scene_from_args}`, `prepare.config.PrepareConfig`.

- [ ] **Step 1 : Write the failing test** (API surface only — NO viser server started)

```python
# tests/test_viz_app.py
"""app — the unified prod entry surface. Imports as a pure consumer and exposes run_app/main; the 7
layers are wired (one list); NO viser server is started (that is the runnable __main__ path)."""
import src.viz as viz
from src.viz import app
from src.viz.core.layer import Layer


def test_app_exposes_entry():
    assert callable(viz.run_app) and callable(viz.main)
    assert viz.run_app is app.run_app and viz.main is app.main


def test_app_layer_set_is_the_seven_ported_layers():
    # Instantiate the same layer list app.run_app builds, and check it is the 7 ported Layers.
    from src.viz.layers.fields import FieldsLayer
    from src.viz.layers.ghost import GhostLayer
    from src.viz.layers.ground import GroundLayer
    from src.viz.layers.human_cloud import HumanCloudLayer
    from src.viz.layers.objects import ObjectsLayer
    from src.viz.layers.skeleton import SkeletonLayer
    from src.viz.layers.style import StyleLayer

    layers = [GroundLayer(), GhostLayer(), SkeletonLayer(), HumanCloudLayer(),
              ObjectsLayer(), FieldsLayer(), StyleLayer()]
    assert len(layers) == 7
    assert all(isinstance(layer, Layer) for layer in layers)
```

- [ ] **Step 2 : Run, expect FAIL**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_app.py -q`
Expected: FAIL (`ModuleNotFoundError: src.viz.app` / `run_app` not exported).

- [ ] **Step 3 : Create `src/viz/app.py`**

```python
"""Production viz app — the unified viewer. Builds a BakeSource, a Player, and the seven ported layers,
then serves. Replaces the legacy ``viewer.py`` (one ~370-line god-class) with composable layers: adding
a roadmap layer = one file in ``layers/`` + one line here. Pure consumer: viser stays in
``core/viser_ops`` + the layers + the Player.

Run:
    fuser -k 8080/tcp   # free the port FIRST (never pkill -f this script: it self-kills)
    python -m src.viz.app --motion-path <smplx.npz> --model-dir <smplx_models> \
        [--dataset hodome --port 8080 --frame-step 2 --max-frames 200]
"""
from __future__ import annotations

import argparse

from ..prepare.config import PrepareConfig
from ..prepare.contracts import SceneSpec
from ._scene_args import add_scene_args, scene_from_args
from .core.player import Player
from .layers.fields import FieldsLayer
from .layers.ghost import GhostLayer
from .layers.ground import GroundLayer
from .layers.human_cloud import HumanCloudLayer
from .layers.objects import ObjectsLayer
from .layers.skeleton import SkeletonLayer
from .layers.style import StyleLayer
from .sources import BakeSource


def run_app(spec: SceneSpec, *, port: int = 8080, frame_step: int = 2, max_frames: int = 200,
            solve: bool = False) -> None:
    """Build BakeSource -> Player -> the 7 ported layers -> serve. ``solve`` (phase B) bakes the robot
    side; left False here (the SolvedFrame seam is wired but not filled)."""
    source = BakeSource(spec, PrepareConfig(), solve=solve, frame_step=frame_step,
                        max_frames=max_frames)
    layers = [GroundLayer(), GhostLayer(), SkeletonLayer(), HumanCloudLayer(),
              ObjectsLayer(), FieldsLayer(), StyleLayer()]
    Player(source, layers, port=port).run()


def main() -> None:
    ap = argparse.ArgumentParser()
    add_scene_args(ap)
    ap.add_argument("--solve", action="store_true", help="bake the solved robot side (phase B)")
    a = ap.parse_args()
    spec = scene_from_args(a)
    run_app(spec, port=a.port, frame_step=a.frame_step, max_frames=a.max_frames, solve=a.solve)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4 : Modify `src/viz/__init__.py`** (re-export the entry)

```python
"""viz — pure-consumer visualisation (golden rule 6). Public entry: ``run_app`` / ``main`` (the unified
prod viewer over the Source -> VizFrame -> Layers framework). See docs/VIZ.md."""
from .app import main, run_app

__all__ = ["run_app", "main"]
```

- [ ] **Step 5 : Run, expect PASS (2 tests) + headless import (viser not eagerly loaded)**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_app.py -q`
Expected: PASS.

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -c "import sys; import src.viz; assert 'viser' not in sys.modules, 'viser leaked at import!'; print('viz import viser-free ok')"`
Expected: `viz import viser-free ok`

- [ ] **Step 6 : Commit**

```bash
git add src/viz/app.py src/viz/__init__.py tests/test_viz_app.py
git commit -m "feat(holov2): viz/app — run_app/main (BakeSource+Player+7 couches), entrée prod unifiée"
```

---

### Task 12 : Manual parity check → DELETE `viewer.py` + retire its test

**Files:**
- Delete: `src/viz/viewer.py`
- Delete: `tests/test_viewer_bake.py`  (superseded by `test_viz_bake_source.py` + `test_viz_app.py`)

**Interfaces:** none new — this task removes the legacy viewer once `app.py` reaches parity.

- [ ] **Step 1 : Manual viser parity — run BOTH viewers side by side (DO THIS BEFORE deleting)**

Pick a real demo sequence (relative path resolves via `paths.toml`; substitute a real one). Run from `HoloV2/`:

```bash
fuser -k 8080/tcp 8081/tcp
PY=~/.holonew_deps/miniconda3/envs/holonew/bin/python
# legacy viewer on 8080
$PY -m src.viz.viewer --dataset hodome --motion-path smplx/<seq>.npz --max-frames 20 --port 8080 &
# new app on 8081
$PY -m src.viz.app    --dataset hodome --motion-path smplx/<seq>.npz --max-frames 20 --port 8081 &
```

Confirm in the browser (http://localhost:8080 vs http://localhost:8081) that the new app matches on:
ground (now the real SDF surface, not a flat box — EXPECTED improvement), SMPL ghost, skeleton, object
clouds, human cloud colours under each colour mode + channel, witness/normals toggles, style points +
orientation frames + labels, the Playback slider/play/fps. Then `fuser -k 8080/tcp 8081/tcp`.

- [ ] **Step 2 : Delete the legacy files**

```bash
git rm src/viz/viewer.py tests/test_viewer_bake.py
```

- [ ] **Step 3 : Confirm no dangling imports of the old viewer**

Run: `grep -rn "viz.viewer\|view_trace\|import viewer" src tests | grep -v ".pyc"`
Expected: NO matches in `src/` or `tests/` (only `docs/` history may mention it — left as-is; the VIZ.md/ARCHITECTURE.md rewrite is Task 13).

- [ ] **Step 4 : Full viz suite green**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_viz_colors.py tests/test_viz_viser_ops.py tests/test_viz_model.py tests/test_viz_layer.py tests/test_viz_player.py tests/test_viz_bake_source.py tests/test_viz_ground.py tests/test_viz_layers.py tests/test_viz_app.py tests/test_scene_args.py -q`
Expected: PASS (data-gated bake tests skip if HODome data absent).

- [ ] **Step 5 : Commit**

```bash
git add -A
git commit -m "refactor(holov2): viz — supprime viewer.py (god-class) remplacé par app.py à parité ; retire test_viewer_bake"
```

---

### Task 13 : Rewrite `docs/VIZ.md` for the new architecture (+ ARCHITECTURE.md one-liner)

**Files:**
- Modify: `docs/VIZ.md` (full architecture rewrite — this phase OWNS it)
- Modify: `docs/ARCHITECTURE.md` (single stale `viz/viewer.py` reference → `viz/app.py`)

**Interfaces:** none (docs only).

- [ ] **Step 1 : Rewrite `docs/VIZ.md`** (replace the whole file)

```markdown
# HoloV2 — `viz/` (visualiseur)

Module **top-level indépendant**, **consommateur pur** (règle d'or 6) : il LIT des artefacts typés et
les affiche. **Zéro hook** dans le calcul, jamais d'`if visualize` dans `prepare`/`targets`/`solve`.
viser est confiné à `core/viser_ops`, aux `layers/`, au `Player` et à `app.py`.

## Fil rouge : Source → VizFrame → Layers (le Player orchestre)

`viz/` possède SON view-model par frame et le construit depuis la sortie publique du pipeline
(`prepare` + `targets` (+ `solve`)). Les couches ne lisent QUE le view-model, jamais les contrats
pipeline.

- **`model.py`** (numpy-only, sans viser/torch) :
  - `VizContext` — assets STATIQUES par scène, fournis à chaque couche au `setup` (noms de canaux,
    `margin`, links de style, faces/parents SMPL, `n_objects`, URDF robot, `has_solve`, SDF du sol).
  - `VizFrame` — UNE frame (`Source.get(i)`) : `pose`, `smpl_verts_world`, `human_cloud_world`,
    `object_clouds_world`, `human_field`, `targets`, `solved | None`.
  - `SolvedFrame` — bundle post-solve (`q`, `object_poses`, points robot FK, transforms de liens,
    `style_achieved`/`contact_achieved`, `cost`/`cost_by_term`/`n_iters`/`status`). `None` tant que
    non résolu → les couches solve se masquent. **Optionnel de bout en bout.**

- **`sources.py`** (numpy-only) — protocole `Source` (`get(i)->VizFrame`, `n_frames`, `context`) :
  - `BakeSource(spec, config, *, solve=False, frame_step, max_frames)` : `prepare` 1×, puis bake
    `trace_frame` + `smpl_verts_world` par frame montrée (playback fluide offline). `solve=False`
    laisse `solved=None` ; `solve=True` (bake `SolveTrajectory` + `Evaluator`) = **phase B**.
  - `LiveSource` : `trace_frame` (+ solve) à la volée — **différé** (même interface plus tard).

- **`core/`** (socle partagé prod + debug) :
  - `colors.py` — `heat_distance` · `diverging` · `parity` · `active_mask` · `AXIS_COLORS` (colormaps
    uniques, ex-dup ×3).
  - `viser_ops.py` — `quat_wxyz_to_R` · `hide` (tue le hack du triangle dégénéré) · wrappers add_*.
  - `layer.py` — protocole `Layer` (`folder`, `setup(server, gui, ctx)`, `update(frame, ui)`) +
    `UiState` (sélecteurs partagés : canal · mode couleur · taille de point).
  - `player.py` — `Player` : folder Playback (slider/play/fps) + Selectors, le `render()` qui itère
    `update` sur les couches, la boucle play/fps + le keep-alive (ex-dup ×4).

## Couches (`layers/`, 1 fichier = 1 toggle)

Portées de l'ancien `viewer.py` : `ground` (lit le **SDF du canal sol** — plan/terrain réel, plus de
box plat), `ghost` (mesh SMPL), `skeleton`, `human_cloud` (coloré par champ), `objects` (nuages +
champ env), `fields` (witness + normales), `style` (points + frames + labels). Chaque couche garde
ses handles persistants (créés au `setup`) et n'affecte que `.points/.colors/.visible` au `update` ;
sa checkbox bascule la visibilité localement.

**Roadmap (phase B+)** : `robot` (G1 résolu, ViserUrdf, lit `solved.q`), `cost_dashboard` (panel 2D),
`contacts`, `correspondence`, `sdf_iso`, `geodesic`. Ajouter une couche = 1 fichier `layers/x.py` +
1 ligne dans la liste de `app.py`.

## Entrée prod : `app.py`

`run_app(spec, *, port, frame_step, max_frames, solve=False)` câble `BakeSource → Player → 7 couches →
run()`. CLI : `python -m src.viz.app --motion-path … --model-dir … [--dataset … --max-frames …]`
(flags partagés via `_scene_args`). `viz.run_app` / `viz.main` sont re-exportés.

## Viewers de debug par étape

`viz/scene.py` · `viz/cloud.py` · `viz/sdf.py` · `viz/hoim3_multiperson.py` — consommateurs purs
focalisés par étape (load / point_cloud / sdf / multi-personne). Leur réécriture sur `core/` est la
**phase 6** de la migration (indépendante des couches prod).

## Tests

`viz` = effets aux extrémités ; on ne teste pas le rendu viser. Helpers purs (`core/colors`,
`viser_ops.quat_wxyz_to_R`/`hide`, `ground_surface_mesh`) → entrée connue/uint8 connu. `BakeSource` →
déterminisme + formes/dtypes + chemin `solved=None` (sur données démo, `max_frames` très bas). Couches
minces → conformité structurelle (`isinstance(layer, Layer)` + `folder`) ; parité visuelle = check
manuel. Tests dans `HoloV2/tests/`, python de l'env `holonew`.

## Anti-spaghetti
- Source → VizFrame → Layers, acyclique : `viz/` importe la sortie publique de prepare/targets/solve ;
  rien n'importe `viz/`.
- viser confiné (core/viser_ops + layers + Player + app) ; `model.py`/`sources.py` numpy-only.
- `targets`/`solve` jamais modifiés (consommateur pur) ; `solved` optionnel partout.
```

- [ ] **Step 2 : Fix the stale `viewer.py` reference in `docs/ARCHITECTURE.md`**

Edit the line that reads `viewer de PROD (\`viz/viewer.py\`, le \`FrameTrace\`) respecte la seam …` to point at `viz/app.py` (the unified Source → VizFrame → Layers viewer) instead of the deleted `viz/viewer.py`. Keep the rest of the sentence intact.

- [ ] **Step 3 : Sanity — docs reference the real entry**

Run: `grep -rn "viz/app.py\|viz/viewer.py" docs/VIZ.md docs/ARCHITECTURE.md`
Expected: references point at `viz/app.py`; no live `viz/viewer.py` mention in VIZ.md/ARCHITECTURE.md.

- [ ] **Step 4 : Commit**

```bash
git add docs/VIZ.md docs/ARCHITECTURE.md
git commit -m "docs(holov2): VIZ.md réécrit (archi Source→VizFrame→Layers) + ARCHITECTURE.md pointe app.py"
```

---

## Self-Review

**1. Spec coverage** (every PHASE-A item from `2026-06-30-viz-architecture-design.md`, migration 1-3):
- `core/colors` (heat_distance/diverging/parity/active_mask/AXIS) → Task 1. ✅
- `core/viser_ops` (quat wxyz→R, hide() replaces the triangle hack, add_* helpers) → Task 2. ✅
- `core/layer` (Layer Protocol + UiState) → Task 4. ✅
- `core/player` (Playback folder + render loop + play/fps + keep-alive, tue dup ×4) → Task 5. ✅
- `model.py` (VizContext · VizFrame · SolvedFrame full field set; solved=None) → Task 3. ✅
- `sources.py` (Source Protocol + BakeSource(solve=False) + LiveSource deferred stub) → Task 6. ✅
- 7 ported layers (ground reads the ground SDF; ghost/skeleton/human_cloud/objects/fields/style) → Tasks 7-10. ✅
- `app.py` run_app/main + `__init__` re-export → Task 11. ✅
- DELETE viewer.py after parity → Task 12 (manual parity FIRST, then delete + retire its test). ✅
- VIZ.md architecture rewrite (this phase owns it) + ARCHITECTURE.md one-liner → Task 13. ✅
- Invariants: viz importe que les sorties publiques (prepare.runner/targets.pipeline) ; viser confiné ; model/sources numpy-only (asserted in Tasks 3, 6, 11) ; targets/solve inchangés. ✅
- Deferred & noted: LiveSource (stub), solve bake + robot layer + panels (phase B), debug viewers rewrite (migration phase 6), the "active probes" status line (→ cost_dashboard panel).

**2. Placeholder scan**: no `TBD`/`TODO`/`...similar to Task N`. Every task ships full code + exact run/expected/commit. The only `...` are Python `Protocol` method bodies (idiomatic) and one explicit fallback NOTE (viser line-handle setter) — not a placeholder. ✅

**3. Type-name consistency** with the canonical API: `VizContext`/`VizFrame`/`SolvedFrame`, `Source`/`BakeSource`/`LiveSource`, `Layer`/`UiState`, `Player`, `heat_distance`/`diverging`/`parity`/`active_mask`/`AXIS_COLORS`, `quat_wxyz_to_R`/`hide`, `run_app`/`main`, layer class names (`GroundLayer`/`GhostLayer`/`SkeletonLayer`/`HumanCloudLayer`/`ObjectsLayer`/`FieldsLayer`/`StyleLayer`) and their `folder` strings — all identical across model/sources/layers/app/tests. `ground_surface_mesh` is the only added helper. ✅
```
