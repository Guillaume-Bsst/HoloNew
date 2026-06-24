# Object Grounding in the `ground` Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `ground` stage carry the grounded object pose as the single source every interaction consumer reads, and align the MuJoCo-driven object (`_obj_poses_mj`) onto it so the HODome object stops floating.

**Architecture:** One `ground_object_pose()` call (in `movable.py`) produces the scaled+grounded object pose. The builder runs it right after `compute_stages`, stores the result in `gmr_stages["ground"]["object_pose"]` and binds it to `rt._obj_poses_raw` (same array, one computation), still sets `rt._obj_ground_shift`, and derives `_obj_poses_mj` and the SDF probe from it. The viewer is unchanged (it already grounds the Floor stage via `rt._obj_ground_shift`). Windowing slices the new stage key in lockstep.

**Tech Stack:** Python, numpy, scipy (`Rotation`), pinocchio, cvxpy; pytest.

**Spec:** `HoloNew/docs/specs/2026-06-24-object-grounding-grounded-stage-design.md`

## Global Constraints

- **Commits:** NEVER add a `Co-Authored-By` / `Claude` trailer or any Claude attribution to commit messages.
- **Test env:** run pytest with the conda python `/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python` from the inner package dir `…/HoloNew/HoloNew/` (MuJoCo model paths are relative). The repo's default `python` lacks cvxpy.
- **Keep runs fast:** prefer targeted test selection; never run the whole `pytest tests/` suite for iteration (30+ min). HODome integration tests are gated on local assets and skip when absent.
- **Gating:** the object ground shift is HODome-only (`dataset == "hodome"`); every other dataset returns shift `0.0`, keeping OMOMO byte-identical.
- **Single source:** the grounded object pose is computed exactly once (`ground_object_pose`); no consumer recomputes the shift from raw poses.

---

### Task 1: `ground_object_pose` helper

**Files:**
- Modify: `HoloNew/src/test_socp/movable.py` (add function after `sample_object_surface`, ~line 95)
- Test: `HoloNew/tests/test_object_grounding.py` (create)

**Interfaces:**
- Consumes: nothing (pure function; numpy already imported in `movable.py`).
- Produces: `ground_object_pose(obj_poses_scaled: np.ndarray, object_surface_local: np.ndarray | None, dataset: str | None) -> tuple[np.ndarray, float]` returning `(grounded_poses (T,7) [qw,qx,qy,qz,x,y,z], shift)`.

- [ ] **Step 1: Write the failing test**

Create `HoloNew/tests/test_object_grounding.py`:

```python
"""ground_object_pose: a single constant per-clip z-shift drops the object so its lowest
surface point over the clip rests on z=0 (HODome only). Non-HODome / no-surface = no-op."""
import numpy as np

from HoloNew.src.test_socp.movable import ground_object_pose


def _identity_poses(T, z):
    p = np.zeros((T, 7))
    p[:, 0] = 1.0          # qw (identity rotation)
    p[:, 6] = z            # translate z
    return p


def test_hodome_grounds_lowest_surface_to_zero():
    surface = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    poses = _identity_poses(3, 0.5)        # lowest surface world z = 0.5
    grounded, shift = ground_object_pose(poses, surface, "hodome")
    assert np.isclose(shift, -0.5)
    # lowest surface point over the clip now rests on z=0 (identity rot => world = local + t)
    zmin = min((surface + grounded[t, 4:7])[:, 2].min() for t in range(3))
    assert np.isclose(zmin, 0.0, atol=1e-9)
    # only z is shifted; rotation + xy untouched
    np.testing.assert_array_equal(grounded[:, :6], poses[:, :6])


def test_non_hodome_is_noop():
    surface = np.array([[0, 0, 0], [0, 0, 1]], float)
    poses = _identity_poses(2, 0.5)
    grounded, shift = ground_object_pose(poses, surface, "omomo")
    assert shift == 0.0
    np.testing.assert_array_equal(grounded, poses)


def test_no_surface_is_noop():
    poses = _identity_poses(2, 0.5)
    grounded, shift = ground_object_pose(poses, None, "hodome")
    assert shift == 0.0
    np.testing.assert_array_equal(grounded, poses)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_object_grounding.py -q`
Expected: FAIL with `ImportError: cannot import name 'ground_object_pose'`.

- [ ] **Step 3: Write minimal implementation**

In `HoloNew/src/test_socp/movable.py`, add after `sample_object_surface` (before the next `def`):

```python
def ground_object_pose(obj_poses_scaled, object_surface_local, dataset):
    """Ground the object onto z=0 with one constant per-clip z-shift (HODome only).

    Mirrors the human floor correction on the object side: lift the (already XY/Z
    scaled) object so its lowest surface point over the whole clip rests on z=0 — the
    nominal floor the robot stands on. HODome reconstructs the human (SMPL-X/RGB) and the
    object (optitrack) in frames whose floors disagree by a few cm, leaving the object
    below z=0; OMOMO objects are already floor-consistent (and golden-locked), so the
    shift is gated to HODome.

    Args:
        obj_poses_scaled: (T, 7) [qw, qx, qy, qz, x, y, z], XY/Z scaling already applied.
        object_surface_local: (M, 3) object-local surface samples, or None.
        dataset: dataset key; the shift is applied only when it equals "hodome".

    Returns:
        (grounded (T, 7), shift float). shift == 0.0 and poses are returned unchanged
        when dataset != "hodome" or object_surface_local is None.
    """
    poses = np.asarray(obj_poses_scaled, dtype=float).copy()
    if dataset != "hodome" or object_surface_local is None:
        return poses, 0.0
    from scipy.spatial.transform import Rotation as _Rot
    osl = np.asarray(object_surface_local, dtype=float)
    fsamp = np.unique(np.linspace(0, len(poses) - 1, min(len(poses), 60)).astype(int))
    obj_low = min(
        float((osl @ _Rot.from_quat(poses[f, [1, 2, 3, 0]]).as_matrix().T
               + poses[f, 4:7])[:, 2].min())
        for f in fsamp
    )
    shift = float(-obj_low)
    poses[:, 6] += shift
    return poses, shift
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_object_grounding.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add HoloNew/src/test_socp/movable.py HoloNew/tests/test_object_grounding.py
git commit -m "feat(movable): add ground_object_pose (object floor correction, HODome-gated)"
```

---

### Task 2: Route the builder through the grounded stage

**Files:**
- Modify: `HoloNew/src/test_socp/builder.py` (new ground block after `:338`; remove `rt._obj_poses_raw = None` at `:342`; rewrite `_obj_poses_mj` block `:367-376`; remove surface-sampling block `:426-432`; simplify SDF-probe object block `:447-474`)
- Test: `HoloNew/tests/test_object_grounding.py` (append the gated integration test)

**Interfaces:**
- Consumes: `ground_object_pose` (Task 1); `sample_object_surface` (`movable.py`); `convert_object_poses_to_mujoco_order` (`examples/robot_retarget.py`).
- Produces: after `from_config`, `rt.gmr_stages["ground"]["object_pose"]` is `(T,7)` and `is rt._obj_poses_raw` (when an object mesh is present), `rt._obj_ground_shift` is set, `rt._obj_poses_mj` (when built) equals `convert_object_poses_to_mujoco_order(rt._obj_poses_raw)`.

- [ ] **Step 1: Write the failing test**

Append to `HoloNew/tests/test_object_grounding.py`:

```python
import pytest
from pathlib import Path

from HoloNew.src.paths import get_path

_HODOME = get_path("hodome") / "smplx" / "subject01_baseball.npz"
_SMPLX = get_path("smplx_models") / "smplx"
_HAVE = (_HODOME.exists() and _SMPLX.is_dir()
         and (get_path("hodome") / "object" / "subject01_baseball.npz").exists())


@pytest.mark.skipif(not _HAVE, reason="HODome + SMPL-X assets not present")
def test_object_grounded_in_ground_stage_single_source():
    from HoloNew.examples.robot_retarget import (
        RetargetingConfig, convert_object_poses_to_mujoco_order)
    from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
    from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
    from HoloNew.src.data_loaders.facade import normalize_dataset_cfg
    from scipy.spatial.transform import Rotation as Rot

    cfg = RetargetingConfig(dataset="hodome", motion_name="subject01_baseball",
                            task_type="object_interaction",
                            retargeter=TestSocpRetargeterConfig(floor_contact_margin=0.01))
    normalize_dataset_cfg(cfg)
    rt = TestSocpRetargeter.from_config(cfg)

    # The grounded object lives in the ground stage AND is the single source object pose.
    obj = rt.gmr_stages["ground"]["object_pose"]
    assert obj is rt._obj_poses_raw
    assert rt._obj_ground_shift > 0.0          # a real downward correction was applied

    # The lowest object surface point over the clip rests on z ~ 0.
    osl = np.asarray(rt.object_surface_local, float)
    fsamp = np.unique(np.linspace(0, len(obj) - 1, min(len(obj), 60)).astype(int))
    zmin = min(float((osl @ Rot.from_quat(obj[f, [1, 2, 3, 0]]).as_matrix().T
                      + obj[f, 4:7])[:, 2].min()) for f in fsamp)
    assert abs(zmin) < 1e-6, f"object lowest surface z={zmin:.4f} not grounded"

    # If the MuJoCo drive was built, it derives from the same grounded source.
    if rt._obj_poses_mj is not None:
        np.testing.assert_allclose(
            rt._obj_poses_mj, convert_object_poses_to_mujoco_order(rt._obj_poses_raw))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_object_grounding.py::test_object_grounded_in_ground_stage_single_source -q`
Expected: FAIL — `gmr_stages["ground"]` has no `"object_pose"` key (`KeyError`), because the current builder grounds inside the SDF block and never writes the stage. (If assets are absent the test SKIPs — in that case this task's behaviour cannot be verified locally; note the skip and rely on review.)

- [ ] **Step 3a: Add the grounded-object block after `compute_stages`**

In `HoloNew/src/test_socp/builder.py`, immediately after the q_init base placement
(after the line `rt.q_init_full[3:7] = ground["quat"][0, _pelvis_bi]`, ~`:338`), insert:

```python
    # Object scene grounding (single source). Scale the raw object poses, sample the
    # object surface (mesh only — no SDF dependency), and ground the object onto z=0 with
    # one constant per-clip z-shift (HODome only). The grounded pose lands in the 'ground'
    # stage and is bound to rt._obj_poses_raw: it is THE object pose every interaction
    # consumer reads (SDF probe, per-frame obj_pose_ref, movable reference, object<->floor
    # inertia) and the MuJoCo drive (_obj_poses_mj) derives from below.
    from pathlib import Path
    rt._obj_ground_shift = 0.0
    rt.object_surface_local = None
    rt._obj_poses_raw = None
    if (_obj_poses_all is not None and _mesh_file is not None
            and Path(_mesh_file).exists()):
        from HoloNew.src.test_socp.movable import (
            ground_object_pose, sample_object_surface)
        _obj_scaled = _obj_poses_all[:T].copy()        # (T, 7) [qw,qx,qy,qz,x,y,z]
        _obj_scaled[:, 4:6] *= _obj_xy   # XY
        _obj_scaled[:, 6] *= _obj_z      # Z
        rt.object_surface_local = sample_object_surface(_mesh_file)
        _grounded_obj, rt._obj_ground_shift = ground_object_pose(
            _obj_scaled, rt.object_surface_local, getattr(cfg, "dataset", None))
        rt.gmr_stages["ground"]["object_pose"] = _grounded_obj
        rt._obj_poses_raw = _grounded_obj
```

- [ ] **Step 3b: Remove the now-stale `_obj_poses_raw` default**

Delete the existing assignment near `:340-342` so it does not clobber Step 3a's binding.
Remove these lines:

```python
    # Raw object poses [qw, qx, qy, qz, x, y, z] used by the smplx_ground_probe
    # and D/X interaction terms. None until the object SDF block loads them.
    rt._obj_poses_raw = None
```

- [ ] **Step 3c: Derive `_obj_poses_mj` from the grounded source**

Replace the `_obj_poses_mj` block (currently `:367-376`):

```python
    rt._obj_poses_mj = None
    if (sc.activate_obj_non_penetration and sc.load_object_scene
            and rt.object_name not in (None, "ground") and _obj_poses_all is not None):
        from HoloNew.examples.robot_retarget import convert_object_poses_to_mujoco_order
        obj_poses = _obj_poses_all[:T].copy()           # (T, 7) [qw,qx,qy,qz,x,y,z]
        # Place the object independently of the robot (no-op at the TEST defaults 1.0).
        obj_poses[:, 4:6] *= _obj_xy   # XY
        obj_poses[:, 6] *= _obj_z      # Z
        # Convert from [qw,qx,qy,qz,x,y,z] to MuJoCo order [x,y,z,qw,qx,qy,qz]
        rt._obj_poses_mj = convert_object_poses_to_mujoco_order(obj_poses)
```

with:

```python
    rt._obj_poses_mj = None
    if (sc.activate_obj_non_penetration and sc.load_object_scene
            and rt.object_name not in (None, "ground")
            and rt._obj_poses_raw is not None):
        from HoloNew.examples.robot_retarget import convert_object_poses_to_mujoco_order
        # Single source: the grounded stage object (already scaled + HODome-grounded).
        # Converting [qw,qx,qy,qz,x,y,z] -> MuJoCo [x,y,z,qw,qx,qy,qz]. This aligns the
        # MuJoCo-driven object's z with the interaction object (it previously omitted the
        # HODome ground shift, so the driven object floated).
        rt._obj_poses_mj = convert_object_poses_to_mujoco_order(rt._obj_poses_raw[:T])
```

- [ ] **Step 3d: Remove the relocated surface-sampling block**

Delete the surface-sampling block (currently `:426-432`), since the surface is now
sampled in Step 3a:

```python
    # Object surface control points (object-local) for the object<->floor
    # inertia term. Sampled once from the object mesh; only needed when the
    # object pose is a variable (movable) on an object task. (_mesh_file computed above.)
    if (rt.object_sdf is not None and _mesh_file is not None
            and Path(_mesh_file).exists()):
        from HoloNew.src.test_socp.movable import sample_object_surface
        rt.object_surface_local = sample_object_surface(_mesh_file)
```

- [ ] **Step 3e: Simplify the SDF-probe object block to read the single source**

Replace the object-pose preparation inside the probe block (currently `:447-474`):

```python
        if rt.object_sdf is not None and _obj_poses_all is not None:
            obj_poses = _obj_poses_all[:T].copy()           # (T, 7) [qw,qx,qy,qz,x,y,z]
            # Place the object independently (no-op at the TEST defaults 1.0); the D/X
            # interaction and movable terms read these poses.
            obj_poses[:, 4:6] *= _obj_xy   # XY
            obj_poses[:, 6] *= _obj_z      # Z
            # Ground the object to ITS OWN floor (HODome only): lift so its lowest surface
            # point over the clip rests on z=0 (the nominal floor the robot stands on). HODome
            # reconstructs the human (SMPL-X/RGB) and the object (optitrack) in frames whose
            # floors disagree by a few cm, leaving the object below z=0; OMOMO objects are
            # already floor-consistent (and golden-locked), so this is gated to HODome. A
            # floor-resting object (chair, table) then sits on z=0; a held object rests on the
            # floor at some point too, so its min is the floor. The solve's object-contact keeps
            # the robot on the object regardless of the few-cm reference offset.
            rt._obj_ground_shift = 0.0
            if getattr(cfg, "dataset", None) == "hodome":
                from scipy.spatial.transform import Rotation as _Rot
                _osl = np.asarray(rt.object_surface_local, dtype=float)
                _fsamp = np.unique(np.linspace(0, len(obj_poses) - 1, min(len(obj_poses), 60)).astype(int))
                _obj_low = min(float((_osl @ _Rot.from_quat(obj_poses[f, [1, 2, 3, 0]]).as_matrix().T
                                      + obj_poses[f, 4:7])[:, 2].min()) for f in _fsamp)
                rt._obj_ground_shift = float(-_obj_low)
                obj_poses[:, 6] += rt._obj_ground_shift
            rt._obj_poses_raw = obj_poses
            _obj_poses_arg = obj_poses
        else:
            rt._obj_poses_raw = None
            _obj_poses_arg = None
```

with:

```python
        # The grounded stage object (built above, bound to rt._obj_poses_raw) is the
        # single source the probe queries for the object-local SDF transform. Pass it
        # only when an object SDF exists (floor-only mode ignores the object channel).
        _obj_poses_arg = rt._obj_poses_raw if rt.object_sdf is not None else None
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_object_grounding.py -q`
Expected: PASS (unit tests pass; the integration test passes if HODome assets are present, otherwise SKIP). Also run the existing object tests as a regression check:
`cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_object_warmstart.py tests/test_floor_offset.py -q`
Expected: PASS or SKIP (no new failures).

- [ ] **Step 5: Commit**

```bash
git add HoloNew/src/test_socp/builder.py HoloNew/tests/test_object_grounding.py
git commit -m "feat(builder): ground the object in the ground stage as the single interaction source"
```

---

### Task 3: Window the grounded stage object pose

**Files:**
- Modify: `HoloNew/examples/view_stages.py` (`_slice_stage`, ~`:136`)
- Test: `HoloNew/tests/test_view_stages_windowing.py` (append a test)

**Interfaces:**
- Consumes: `_window_solve_frames` (`examples/view_stages.py`); the `gmr_stages["ground"]["object_pose"]` key produced by Task 2.
- Produces: `_window_solve_frames` slices `gmr_stages[*]["object_pose"]` to `[start:]` in lockstep with `_obj_poses_raw`.

- [ ] **Step 1: Write the failing test**

Append to `HoloNew/tests/test_view_stages_windowing.py`:

```python
def test_window_slices_stage_object_pose():
    # The grounded object lives in gmr_stages["ground"]["object_pose"] and must be
    # windowed in lockstep with _obj_poses_raw (same data the builder binds them to).
    T, start = 60, 20
    op = np.arange(T * 7).reshape(T, 7).astype(float)
    ground = {"pos": np.zeros((T, 5, 3)), "quat": np.zeros((T, 5, 4)), "object_pose": op}
    rt = types.SimpleNamespace(
        gmr_stages={"ground": ground}, gmr_ground=ground,
        gmr_grounded=np.zeros((T, 52, 3)),
        _obj_poses_raw=op.copy(), _obj_poses_mj=None,
        _smplx_orientations=np.zeros((T, 22, 4)), human_quat=np.zeros((T, 52, 4)),
        foot_sticking_sequences=None,
        smplx_ground_probe=types.SimpleNamespace(obj_quat=None, obj_trans=None))
    _window_solve_frames(rt, start)
    assert rt.gmr_stages["ground"]["object_pose"].shape[0] == T - start
    # local frame 0 maps to global frame `start`
    np.testing.assert_array_equal(
        rt.gmr_stages["ground"]["object_pose"][0], np.arange(start * 7, start * 7 + 7))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_view_stages_windowing.py::test_window_slices_stage_object_pose -q`
Expected: FAIL — `object_pose` still has `T` rows (slicer only handles `pos`/`quat`).

- [ ] **Step 3: Add `object_pose` to the slicer**

In `HoloNew/examples/view_stages.py`, in `_slice_stage`, change:

```python
        for _k in ("pos", "quat"):
            if d.get(_k) is not None:
                d[_k] = d[_k][start:]
```

to:

```python
        for _k in ("pos", "quat", "object_pose"):
            if d.get(_k) is not None:
                d[_k] = d[_k][start:]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_view_stages_windowing.py -q`
Expected: PASS (all windowing tests).

- [ ] **Step 5: Commit**

```bash
git add HoloNew/examples/view_stages.py HoloNew/tests/test_view_stages_windowing.py
git commit -m "fix(view_stages): window the grounded stage object_pose in lockstep"
```

---

## Notes for the implementer

- `rt.gmr_ground` is the SAME dict object as `rt.gmr_stages["ground"]` (the builder
  aliases them), so writing `gmr_stages["ground"]["object_pose"]` is visible via both.
- `_obj_poses_raw` is intentionally a normal attribute (not a property): tests assign it
  directly (`test_object_warmstart.py:74`, `test_view_stages_windowing.py`). After
  windowing it and the stage `object_pose` hold equal-but-separate sliced arrays — that is
  fine; nothing relies on them being the same object post-window.
- Behaviour: OMOMO and other non-HODome datasets get shift `0.0`, so `_obj_poses_raw`,
  `_obj_poses_mj`, and the stage object are numerically identical to before. Only HODome
  changes (the MuJoCo-driven object gains the ground shift). The viewer's Floor stage was
  already grounded via `rt._obj_ground_shift`, which is still set.
```
