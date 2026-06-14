# Inertia Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the robot's placement emerge from contacts (feet pinned to a permanent floor entity) plus a weak centroidal W^c that fills the residual/flight, removing the anti-paper pelvis scaffold and CoM position anchor — all behind an opt-in `inertia_mode` flag that leaves parity mode bit-exact.

**Architecture:** A new `inertia_mode` config flag bundles: floor-as-permanent-entity (feet↔floor D/X + persistence loaded even without an object), `pelvis_anchor_weight=0`, `lambda_c_pos=0`, and weak `activate_centroidal` (W^c/W^L). The floor channel is decoupled from object presence by giving `query_entities` and `SmplxGroundProbe` a floor-only path (object_sdf=None → inactive object field). Default (`inertia_mode=False`) is untouched.

**Tech Stack:** Python, cvxpy/CLARABEL, pinocchio, numpy. Env: `/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python`. Run from package dir `/home/gbesset/Documents/wbt_rl/modules/01_retargeting/HoloNew/HoloNew`. Reference spec: `docs/specs/2026-06-14-inertia-mode-design.md`.

**Run all commands from the package dir.** Tests: `<env-python> -m pytest tests/<file> -q`.

---

### Task 1: Inactive object field + `query_entities` floor-only guard

**Files:**
- Modify: `src/test_socp/contact/contact_field.py` (add `inactive_field`)
- Modify: `src/test_socp/interaction.py` (`query_entities`)
- Test: `tests/test_interaction_floor_only.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_interaction_floor_only.py
"""Floor-only interaction path: query_entities works when object_sdf is None."""
import numpy as np
from HoloNew.src.test_socp.contact.contact_field import ContactField, inactive_field


def test_inactive_field_shapes_and_flags():
    f = inactive_field(5, margin=0.1)
    assert isinstance(f, ContactField)
    assert f.distance.shape == (5,) and np.all(f.distance >= 0.1)
    assert f.direction.shape == (5, 3) and np.all(f.direction == 0)
    assert f.witness.shape == (5, 3) and np.all(f.witness == 0)
    assert f.active.shape == (5,) and not f.active.any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `<env-python> -m pytest tests/test_interaction_floor_only.py::test_inactive_field_shapes_and_flags -q`
Expected: FAIL (`ImportError: cannot import name 'inactive_field'`).

- [ ] **Step 3: Add `inactive_field` to contact_field.py**

After the `ContactField` dataclass definition add:

```python
def inactive_field(n: int, margin: float) -> "ContactField":
    """An all-inactive ContactField of length n (distance=+margin, others zero).

    Used for the object channel when no object SDF is present (floor-only mode):
    activation alpha(d>=margin)=0, so these points contribute nothing to D/X/P.
    """
    return ContactField(
        distance=np.full(n, margin, dtype=np.float64),
        direction=np.zeros((n, 3), dtype=np.float64),
        witness=np.zeros((n, 3), dtype=np.float64),
        active=np.zeros(n, dtype=bool),
    )
```

(Ensure `import numpy as np` is present at the top of the file.)

- [ ] **Step 4: Run test to verify it passes**

Run: `<env-python> -m pytest tests/test_interaction_floor_only.py::test_inactive_field_shapes_and_flags -q`
Expected: PASS.

- [ ] **Step 5: Guard `query_entities` for `object_sdf=None`**

In `src/test_socp/interaction.py`, `query_entities`, replace the object-SDF block:

```python
    # Object SDF: query in the object-local frame (mirrors smplx_field exactly).
    pts_local = _world_to_object_local(pts_world, obj_pose)
    fobj = rt.object_sdf.query(pts_local, margin)
```

with:

```python
    # Object SDF: query in the object-local frame (mirrors smplx_field exactly).
    # Floor-only mode (inertia_mode without an object): no object SDF -> an
    # all-inactive object field, so only the floor channel contributes.
    if getattr(rt, "object_sdf", None) is None:
        from HoloNew.src.test_socp.contact.contact_field import inactive_field
        fobj = inactive_field(pts_world.shape[0], margin)
    else:
        pts_local = _world_to_object_local(pts_world, obj_pose)
        fobj = rt.object_sdf.query(pts_local, margin)
```

- [ ] **Step 6: Add the floor-only query test**

```python
# append to tests/test_interaction_floor_only.py
import pytest
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.src.test_socp.interaction import query_entities, robot_control_points


def test_query_entities_floor_only():
    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="object_interaction", task_name="sub3_largebox_003", data_format="smplh"))
    if rt.correspondence is None:
        pytest.skip("assets not present")
    # Force floor-only by dropping the object SDF.
    rt.object_sdf = None
    q_pin = rt.pin.qpos_mj_to_q_pin(rt.q_init_full[:36])
    P = robot_control_points(rt, q_pin)
    fobj, fflr = query_entities(rt, P, rt._obj_poses_raw[0], margin=0.1)
    assert not np.asarray(fobj.active).any(), "object channel must be inactive with no SDF"
    assert np.all(np.isfinite(np.asarray(fflr.distance))), "floor field must be finite"
```

- [ ] **Step 7: Run both tests**

Run: `<env-python> -m pytest tests/test_interaction_floor_only.py -q`
Expected: PASS (2 passed).

- [ ] **Step 8: Commit**

```bash
git add src/test_socp/contact/contact_field.py src/test_socp/interaction.py tests/test_interaction_floor_only.py
git commit -m "feat(test_socp): floor-only interaction path (inactive object field when object_sdf is None)"
```

---

### Task 2: `SmplxGroundProbe` floor-only path

**Files:**
- Modify: `src/test_socp/contact/smplx_field.py` (`SmplxGroundProbe`, `build_smplx_ground_probe`)
- Test: `tests/test_interaction_floor_only.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_interaction_floor_only.py
def test_ground_probe_floor_only_runs():
    from HoloNew.src.test_socp.contact.smplx_field import build_smplx_ground_probe
    from HoloNew.src.test_socp.contact.constants import (
        CONTACT_MARGIN_M, OMOMO_DIR_DEFAULT)
    from HoloNew.src.test_socp.correspondence.build_correspondence import (
        SMPLX_MODEL_DIR_DEFAULT, HUMAN_GRID_DENSITY)
    import os
    if not os.path.isdir(OMOMO_DIR_DEFAULT):
        pytest.skip("OMOMO data not present")
    probe = build_smplx_ground_probe(
        "sub3_largebox_003", OMOMO_DIR_DEFAULT, SMPLX_MODEL_DIR_DEFAULT,
        object_sdf=None, obj_poses=None, margin=CONTACT_MARGIN_M,
        density=HUMAN_GRID_DENSITY)
    pf = probe(0, np.zeros((52, 4)), np.zeros(3))
    assert pf.points.ndim == 2 and pf.points.shape[1] == 3
    assert np.all(np.isfinite(pf.points))
    assert not np.asarray(pf.field.active).any()
```

(If the exact import path of `SMPLX_MODEL_DIR_DEFAULT`/`HUMAN_GRID_DENSITY` differs, mirror the imports used in `from_config` around `test_socp.py:1545-1580`.)

- [ ] **Step 2: Run to verify it fails**

Run: `<env-python> -m pytest tests/test_interaction_floor_only.py::test_ground_probe_floor_only_runs -q`
Expected: FAIL (object_sdf required / AttributeError on None).

- [ ] **Step 3: Make `object_sdf`/poses optional in `SmplxGroundProbe`**

In `src/test_socp/contact/smplx_field.py`, change the dataclass fields to allow None and the `__call__` to branch:

```python
    object_sdf: "ObjectSDF | None"
    obj_quat: "np.ndarray | None"
    obj_trans: "np.ndarray | None"
    margin: float

    def __call__(self, t: int, quats_wxyz: np.ndarray, pelvis_grounded: np.ndarray) -> "ProbeFrame":
        """ProbeFrame (probe world points + their ContactField) at frame t. Reads only t."""
        world = self.human_body.placed_points(quats_wxyz, pelvis_grounded, self.cache, frame_idx=t)
        if self.object_sdf is None:
            from HoloNew.src.test_socp.contact.contact_field import inactive_field
            field = inactive_field(world.shape[0], self.margin)
        else:
            local = transform_points_world_to_local(self.obj_quat[t], self.obj_trans[t], world)
            field = self.object_sdf.query(local, self.margin)
        return ProbeFrame(points=world.astype(np.float32), field=field)
```

- [ ] **Step 4: Make `build_smplx_ground_probe` accept `obj_poses=None`**

```python
def build_smplx_ground_probe(task_name, omomo_dir, model_dir, object_sdf,
                             obj_poses, margin, density, cache=None):
    ...  # body construction unchanged
    if obj_poses is None:
        return SmplxGroundProbe(body, cache, object_sdf, None, None, margin)
    obj_poses = np.asarray(obj_poses, dtype=np.float64)
    return SmplxGroundProbe(body, cache, object_sdf, obj_poses[:, :4], obj_poses[:, 4:7], margin)
```

- [ ] **Step 5: Run to verify it passes**

Run: `<env-python> -m pytest tests/test_interaction_floor_only.py::test_ground_probe_floor_only_runs -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/test_socp/contact/smplx_field.py tests/test_interaction_floor_only.py
git commit -m "feat(test_socp): floor-only SmplxGroundProbe (no object SDF)"
```

---

### Task 3: Config flags, asset loading, and `from_config` bundle

**Files:**
- Modify: `src/test_socp/config.py` (`inertia_mode`, `floor_as_entity`)
- Modify: `src/test_socp/test_socp.py` (`__init__` param `floor_as_entity`; `from_config` loading + bundle + guards)
- Test: `tests/test_inertia_mode.py` (new)

- [ ] **Step 1: Write the failing parity + smoke test**

```python
# tests/test_inertia_mode.py
"""Inertia mode: parity preserved when off; floor channel + W^c active when on."""
import numpy as np
import pytest
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter


def _robot_only(**kw):
    return TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh",
        retargeter=TestSocpRetargeterConfig(**kw)))


def test_parity_mode_default_unchanged():
    rt = _robot_only()
    assert rt.activate_centroidal is False
    assert rt.pelvis_anchor_weight == 10.0
    assert getattr(rt, "floor_as_entity", False) is False


def test_inertia_mode_applies_bundle():
    rt = _robot_only(inertia_mode=True)
    assert rt.floor_as_entity is True
    assert rt.pelvis_anchor_weight == 0.0
    assert rt.lambda_c_pos == 0.0
    assert rt.activate_centroidal is True
    assert rt.lambda_c > 0 and rt.lambda_L > 0
    # Floor channel must be loaded even without an object.
    assert rt.correspondence is not None
    assert rt.smplx_ground_probe is not None


def test_inertia_mode_robot_only_runs_finite():
    rt = _robot_only(inertia_mode=True)
    if rt.correspondence is None:
        pytest.skip("assets not present")
    res = rt.retarget(max_frames=5)
    assert np.all(np.isfinite(res.qpos))
```

- [ ] **Step 2: Run to verify it fails**

Run: `<env-python> -m pytest tests/test_inertia_mode.py -q`
Expected: FAIL (`inertia_mode` not a config field).

- [ ] **Step 3: Add config flags**

In `src/test_socp/config.py`, add after the centroidal block:

```python
    # Inertia mode (paper-faithful body placement). When True, from_config applies
    # a bundle: floor_as_entity=True, pelvis_anchor_weight=0, lambda_c_pos=0,
    # activate_centroidal=True with weak lambda_c/lambda_L. The body is then placed
    # by contacts (feet pinned to the permanent floor entity) and a weak W^c that
    # fills the residual/flight, with NO positional pelvis/CoM target. Default off:
    # parity mode (golden/parity tests) is bit-exact. See
    # docs/specs/2026-06-14-inertia-mode-design.md.
    inertia_mode: bool = False
    # floor_as_entity: load the floor interaction channel (correspondence + ground
    # probe + floor field) for ANY task, not only object tasks, so the feet track
    # the floor field even without an object. Turned on by inertia_mode; separable
    # for testing.
    floor_as_entity: bool = False
```

- [ ] **Step 4: Thread `floor_as_entity` through the solver `__init__`**

In `src/test_socp/test_socp.py` `__init__` signature add `floor_as_entity: bool = False,` (next to `load_object_scene`), and in the body add `self.floor_as_entity = floor_as_entity`.

- [ ] **Step 5: Apply the bundle + asset loading + guards in `from_config`**

In `from_config`, BEFORE constructing `rt` (near the `_obj_name` block ~line 1445), insert:

```python
        # Inertia mode bundle: paper-faithful placement (see design doc).
        if sc.inertia_mode:
            kwargs["floor_as_entity"] = True
            kwargs["pelvis_anchor_weight"] = 0.0
            kwargs["lambda_c_pos"] = 0.0
            kwargs["activate_centroidal"] = True
            # Weak W^c / W^L (tuned in Task 4). Below the interaction weights so
            # contacts place the body and W^c only fills the residual/flight.
            kwargs["lambda_c"] = sc.lambda_c if sc.lambda_c > 0 else 1e-5
            kwargs["lambda_L"] = sc.lambda_L if sc.lambda_L > 0 else 1e-4
        else:
            kwargs["floor_as_entity"] = sc.floor_as_entity
```

Then change the robot_only / object weight gating (~line 1446) so floor_as_entity
keeps the interaction live:

```python
        _obj_name = getattr(constants, "OBJECT_NAME", "ground")
        _floor_entity = kwargs.get("floor_as_entity", False)
        if _obj_name in (None, "ground") and not _floor_entity:
            kwargs["lambda_D"] = 0.0
            kwargs["lambda_X"] = 0.0
            kwargs["lambda_P"] = 0.0
            kwargs["activate_persistence"] = False
        elif sc.lambda_D > 0 or sc.lambda_X > 0 or sc.lambda_P > 0 or sc.activate_persistence or _floor_entity:
            kwargs["activate_obj_non_penetration"] = True
            kwargs["load_object_scene"] = False  # ground non-pen only; plain model
```

- [ ] **Step 6: Load the floor channel under `floor_as_entity` even without an object**

In `from_config`, the asset-loading block (`if rt.object_sdf is not None:` ~line 1564) builds correspondence/probe only for object tasks. Wrap it so floor_as_entity gets a floor-only probe. Replace that block with:

```python
        _floor_entity = getattr(rt, "floor_as_entity", False)
        if rt.object_sdf is not None or _floor_entity:
            from HoloNew.src.test_socp.contact.constants import CONTACT_MARGIN_M, OMOMO_DIR_DEFAULT
            from HoloNew.src.test_socp.contact.smplx_field import build_smplx_ground_probe
            from HoloNew.src.test_socp.correspondence.human_body import PointCloudCache
            from HoloNew.src.utils import load_intermimic_data
            if rt.object_sdf is not None:
                _, obj_poses = load_intermimic_data(str(pt_path))
                rt._obj_poses_raw = obj_poses[:T]
                _obj_poses_arg = obj_poses[:T]
            else:
                # Floor-only: no object poses.
                rt._obj_poses_raw = None
                _obj_poses_arg = None
            corr_cache = None
            if rt.correspondence is not None:
                corr_cache = PointCloudCache(tri_idx=rt.correspondence.tri_idx,
                                             bary=rt.correspondence.bary)
            rt.smplx_ground_probe = build_smplx_ground_probe(
                cfg.task_name, OMOMO_DIR_DEFAULT, SMPLX_MODEL_DIR_DEFAULT,
                rt.object_sdf, _obj_poses_arg, CONTACT_MARGIN_M, HUMAN_GRID_DENSITY,
                cache=corr_cache)
```

Note: the correspondence build block (just above, ~line 1545) must also run when
`floor_as_entity`; if it is currently gated, ensure it executes whenever
`rt.object_sdf is not None or floor_as_entity`.

- [ ] **Step 7: Relax the interaction guards in `solve_single_iteration` for floor-only**

In `src/test_socp/test_socp.py`, the D/X, soft-P, and persistence-constraint guards
each require `getattr(self, "object_sdf", None) is not None`. Replace that single
condition in all three guards (~lines 925-961) with:

```python
                and getattr(self, "correspondence", None) is not None \
                and (getattr(self, "object_sdf", None) is not None
                     or getattr(self, "floor_as_entity", False)) \
```

Also the cross-frame `_p_state` init (~line 1222) and the per-frame probe block
(`if self.smplx_ground_probe is not None`) already key off the probe/correspondence,
which are now present in floor-only mode — verify they trigger. The object-pose
threading (`_obj_pose = _obj_poses_raw[t]`) must tolerate `_obj_poses_raw is None`:
guard `_obj_pose = (self._obj_poses_raw[t] if getattr(self, "_obj_poses_raw", None) is not None else None)` (already the case at ~line 1248).

- [ ] **Step 8: Run the inertia-mode tests**

Run: `<env-python> -m pytest tests/test_inertia_mode.py -q`
Expected: PASS (3 passed). If `test_inertia_mode_robot_only_runs_finite` errors in
the floor-only probe/threading, fix the None-guards it surfaces (do not weaken the
assertion).

- [ ] **Step 9: Run the parity regression (must stay bit-exact)**

Run: `<env-python> -m pytest tests/test_retarget_golden.py tests/test_test_socp_parity.py -q`
Expected: PASS (inertia_mode defaults off → unchanged solve).

- [ ] **Step 10: Commit**

```bash
git add src/test_socp/config.py src/test_socp/test_socp.py tests/test_inertia_mode.py
git commit -m "feat(test_socp): inertia_mode flag — floor-as-entity + centroidal bundle, parity preserved"
```

---

### Task 4: Tune weak W^c / W^L and fix W^L reference

**Files:**
- Modify: `src/test_socp/config.py` (`lambda_c`, `lambda_L` inertia defaults via the bundle)
- Modify: `src/test_socp/centroidal.py` (W^L comment: weak spin regularizer, L_ref deferred)
- Test: `tests/test_inertia_mode_metric.py` (new)

- [ ] **Step 1: Write the placement metric test (initially xfail-tolerant)**

```python
# tests/test_inertia_mode_metric.py
"""Inertia mode places the body by contacts alone (no positional target)."""
import numpy as np
import pytest
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.src.test_socp.tables import IK_MATCH_TABLE1, ROBOT_ROOT_NAME
from HoloNew.src.test_socp.targets import ground_frame_targets

N = 30


def _base_drift(rt, qpos):
    gpos, gquat = rt.gmr_ground["pos"], rt.gmr_ground["quat"]
    n = min(N, qpos.shape[0]); ref = []
    for t in range(n):
        tg = ground_frame_targets(gpos[t], gquat[t], IK_MATCH_TABLE1)
        for frame, (p_t, _, _, _) in tg.items():
            if rt.robot_link_names[frame] == ROBOT_ROOT_NAME:
                ref.append(p_t[:2]); break
    ref = np.array(ref)
    d = np.linalg.norm(qpos[:len(ref), :2] - ref, axis=1)
    return float(d.mean()), float(d.max())


def test_inertia_mode_largebox_placement():
    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="object_interaction", task_name="sub3_largebox_003", data_format="smplh",
        retargeter=TestSocpRetargeterConfig(inertia_mode=True)))
    if rt.correspondence is None:
        pytest.skip("assets not present")
    res = rt.retarget(max_frames=N)
    assert np.all(np.isfinite(res.qpos)), "non-finite qpos in inertia mode"
    z = res.qpos[:N, 2]
    assert z.min() >= 0.3 and z.max() <= 1.0, f"pelvis z out of range [{z.min():.3f},{z.max():.3f}]"
    mean_d, max_d = _base_drift(rt, res.qpos)
    # With NO positional target, contacts must keep the base near the reference path.
    # Threshold set from the Task-4 sweep; record the chosen value here.
    assert mean_d < 0.20, f"inertia-mode base drift too large: mean={mean_d:.3f} m (contacts not anchoring)"
```

- [ ] **Step 2: Run the lambda_c sweep to pick the weight**

Create `/tmp/inertia_sweep.py` (mirror `/tmp/lopos_sweep.py` structure: chdir to the
package dir, build `TestSocpRetargeterConfig(inertia_mode=True, lambda_c=X, lambda_L=X*10)`
for `X in [1e-6, 1e-5, 1e-4, 1e-3]`, retarget 30 frames on `object_interaction`
`sub3_largebox_003`, print base drift mean/max + pelvis z range + finite). Run:

`<env-python> /tmp/inertia_sweep.py`

Pick the largest `lambda_c` that keeps base drift bounded (contacts still dominate)
AND does not inflate drift (W^c fighting contacts). Record the table in the config
comment. Expected sweet spot ~1e-5.

- [ ] **Step 3: Set the tuned weights in the `from_config` bundle**

Update the bundle defaults in `from_config` (Task 3, Step 5) to the chosen values,
e.g. `kwargs["lambda_c"] = sc.lambda_c if sc.lambda_c > 0 else 1e-5` and
`lambda_L = ... else 1e-4`, and document the sweep table in a comment above them.

- [ ] **Step 4: Fix the W^L docstring/comment in centroidal.py**

In `src/test_socp/centroidal.py`, update the `lambda_L` docstring line and the
`--- W^L ---` comment to state it is a **weak spin regularizer toward zero** and
that paper-faithful `L_ref` tracking is deferred (needs a reference-velocity
pipeline + a free-flight clip; see design doc). Do NOT change the math yet.

- [ ] **Step 5: Run the metric test**

Run: `<env-python> -m pytest tests/test_inertia_mode_metric.py -q`
Expected: PASS. If base drift exceeds the threshold at every weight, lower the
threshold to the best achieved value and document WHY (contacts-only placement has
a floor); do not silently loosen — record the number.

- [ ] **Step 6: Commit**

```bash
git add src/test_socp/config.py src/test_socp/test_socp.py src/test_socp/centroidal.py tests/test_inertia_mode_metric.py
git commit -m "tune(test_socp): weak W^c/W^L for inertia mode; W^L documented as spin regularizer"
```

---

### Task 5: Inertia-mode regression snapshot

**Files:**
- Create: `tests/golden/inertia_mode_qpos.npz`
- Test: `tests/test_inertia_mode_golden.py` (new)

- [ ] **Step 1: Generate the snapshot**

Run a 30-frame `object_interaction` `sub3_largebox_003` retarget with
`inertia_mode=True` and save `res.qpos` to `tests/golden/inertia_mode_qpos.npz`
under key `qpos` (mirror how `baseline_qpos.npz` is structured). Use a short script
in `/tmp`.

- [ ] **Step 2: Write the golden test**

```python
# tests/test_inertia_mode_golden.py
"""Pin the inertia-mode output so future changes are intentional."""
from pathlib import Path
import numpy as np
import numpy.testing as npt
import pytest
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter

_GOLD = Path(__file__).parent / "golden" / "inertia_mode_qpos.npz"


@pytest.mark.skipif(not _GOLD.exists(), reason="inertia-mode golden not present")
def test_inertia_mode_matches_golden():
    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="object_interaction", task_name="sub3_largebox_003", data_format="smplh",
        retargeter=TestSocpRetargeterConfig(inertia_mode=True)))
    if rt.correspondence is None:
        pytest.skip("assets not present")
    res = rt.retarget(max_frames=30)
    gold = np.load(_GOLD)["qpos"]
    npt.assert_allclose(res.qpos, gold, atol=1e-6)
```

- [ ] **Step 3: Run it**

Run: `<env-python> -m pytest tests/test_inertia_mode_golden.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/golden/inertia_mode_qpos.npz tests/test_inertia_mode_golden.py
git commit -m "test(test_socp): inertia-mode regression snapshot"
```

---

### Task 6: Full regression + docs + memory

**Files:**
- Modify: `src/test_socp/centroidal.py` (flight-branch comment)
- Modify: `docs/specs/2026-06-14-inertia-mode-design.md` (status -> implemented)

- [ ] **Step 1: Run the object-path + parity suites**

Run: `<env-python> -m pytest tests/test_inertia_mode.py tests/test_inertia_mode_metric.py tests/test_inertia_mode_golden.py tests/test_interaction_floor_only.py tests/test_retarget_golden.py tests/test_test_socp_parity.py tests/test_movable_metric.py tests/test_interaction_metric.py -q`
Expected: all PASS. Parity tests confirm default behaviour is bit-exact.

- [ ] **Step 2: Document the flight branch**

In `src/test_socp/centroidal.py`, add a comment near W^c noting that in inertia
mode the ballistic-flight branch (contacts deactivate → W^c sole CoM term →
`cddot_ref` carries `-g z`) is implemented but unvalidated (no flight clip in
demo_data).

- [ ] **Step 3: Flip the spec status**

Edit `docs/specs/2026-06-14-inertia-mode-design.md` status line to
`implemented (validated on largebox; flight branch unvalidated)`.

- [ ] **Step 4: Commit + push**

```bash
git add -A && git commit -m "docs(test_socp): inertia mode implemented — flight branch documented as unvalidated"
git push origin HEAD
```

- [ ] **Step 5: Update memory**

Update `project_paper_formulation_autonomy.md` (and the MEMORY.md hook) to record
that inertia_mode replaces the positional crutches with contacts+weak-W^c, default
off (parity preserved), W^L deferred, flight unvalidated.
```
