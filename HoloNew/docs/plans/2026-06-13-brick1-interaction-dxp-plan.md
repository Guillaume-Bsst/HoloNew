# Brick 1 — Interaction D / X / P costs (object + floor) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the paper's uniform interaction costs (normal proximity D, tangential placement X, contact persistence P) into the TEST-SOCP objective for the object SDF and the floor, using the already-computed fields + correspondence, linearized against the pinocchio point Jacobians.

**Architecture:** A new `src/test_socp/interaction.py` owns the per-frame interaction data: robot control-point field queries (object + floor), reference extraction (source probe indexed by correspondence), and the D/X/P residual assembly. `solve_single_iteration` adds the D/X (and P) cvxpy terms next to the existing tracking terms; weights live in `TestSocpRetargeterConfig`. Brick 0's pinocchio backend supplies `point_translational_jacobian` and `v_a_indices`. Objects are still driven per frame (W^o is brick 5).

**Tech Stack:** Python, pinocchio, numpy, cvxpy, pytest.

**Run from** `modules/01_retargeting/HoloNew/HoloNew` with `~/.holonew_deps/miniconda3/envs/holonew/bin/python` (alias `PY`).

**Design:** `docs/specs/2026-06-13-brick1-interaction-dxp-design.md`.

## Confirmed interfaces (from the codebase — use these exactly)
- `rt.correspondence` is a `CorrespondenceTable` (correspondence/build_correspondence.py): `link_idx (M,)`, `offset_local (M,3)`, `link_names (L,)`, `human_idx (M,)` (each G1 control point i is on link `link_names[link_idx[i]]` at `offset_local[i]`, driven by human point `human_idx[i]`).
- `rt.object_sdf.query(pts_local (N,3), margin) -> ContactField` with fields `distance (N,)` signed, `direction (N,3)` unit normal (surface→probe), `witness (N,3)` closest surface point, `active (N,)` bool — in the OBJECT-LOCAL frame. (`contact/backends/sdf.py`, `contact/contact_field.py`.)
- `from HoloNew.src.test_socp.contact.backends.floor import floor_field`; `floor_field(pts_world (N,3), margin) -> ContactField` (analytic z, world frame).
- `rt.smplx_ground_probe(t, human_quat[t], pelvis_grounded[t]) -> ProbeFrame` with `.field` (object `ContactField` at the human correspondence points, object-local) and `.points` (world). The probe samples exactly the correspondence human points, so `field.<x>[human_idx[i]]` is the reference for robot control point i. Object pose per frame is `smplx_ground_probe.obj_quat[t]` (wxyz) / `smplx_ground_probe.obj_trans[t]` (the probe stores these; confirm attribute names in Task 1).
- pinocchio point Jacobian: `rt.pin.point_translational_jacobian(q_pin, link_name, offset_local) -> (3, nv)`; slice `[:, rt.v_a_indices]`. World link transforms: `rt.pin.body_position/body_rotation(q_pin, link_name)`. Convert qpos: `rt.pin.qpos_mj_to_q_pin(q[:36])`.
- The solve decision variable in `solve_single_iteration` is `dqa = cp.Variable(self.nv_a)`; `self.v_a_indices` selects active tangent columns; residual terms are appended to `obj_terms` before `cp.Problem`.

## File structure
- Create `src/test_socp/interaction.py` — per-frame interaction data + residual assembly (one responsibility: turn (q, correspondence, fields, references) into cvxpy D/X/P terms).
- Modify `src/test_socp/test_socp.py` — config weights, thread per-frame interaction data into `solve_single_iteration`, append the terms.
- Modify `src/test_socp/config.py` — add `lambda_D`, `lambda_X`, `lambda_P`, `sigma_v` (default off, then turned on after validation).
- Tests: `tests/test_interaction_dxp.py`.

---

### Task 1: Robot-side field query helper (object + floor)

**Files:** Create `src/test_socp/interaction.py`; Test: `tests/test_interaction_dxp.py`

- [ ] **Step 1: Write the failing test** (querying the object SDF + floor at robot control points returns aligned per-point fields)

```python
# tests/test_interaction_dxp.py
import numpy as np
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.src.test_socp.interaction import robot_control_points, query_entities


def _rt():
    return TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="object_interaction", task_name="sub3_largebox_003", data_format="smplh"))


def test_robot_control_points_and_query_shapes():
    rt = _rt()
    if rt.correspondence is None or rt.object_sdf is None:
        import pytest; pytest.skip("correspondence/object_sdf assets not present")
    q_pin = rt.pin.qpos_mj_to_q_pin(rt.q_init_full[:36])
    P = robot_control_points(rt, q_pin)                 # (M, 3) world
    assert P.shape == (rt.correspondence.link_idx.shape[0], 3)
    obj_pose = np.array([1.0, 0, 0, 0, 0, 0, 0])        # identity for the shape test
    fobj, fflr = query_entities(rt, P, obj_pose)
    assert fobj.distance.shape == (P.shape[0],)
    assert fflr.distance.shape == (P.shape[0],)
```

- [ ] **Step 2: Run** `PY -m pytest tests/test_interaction_dxp.py -q` → FAIL (interaction module missing).

- [ ] **Step 3: Implement `robot_control_points` + `query_entities`** (mirror `smplx_field`'s world→object-local transform):

```python
# src/test_socp/interaction.py
"""Per-frame interaction data and D/X/P residual assembly for TEST-SOCP.

Queries the object SDF and floor fields at the robot control points (the G1 side
of the correspondence), extracts the source references, and builds the cvxpy
normal-proximity (D), tangential-placement (X) and persistence (P) terms. See
docs/specs/2026-06-13-brick1-interaction-dxp-design.md.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from HoloNew.src.test_socp.contact.backends.floor import floor_field


def _world_to_object(pts_world, obj_pose):
    """obj_pose = [qw,qx,qy,qz, x,y,z]; return pts in the object-local frame + R_obj (3,3)."""
    Robj = R.from_quat([obj_pose[1], obj_pose[2], obj_pose[3], obj_pose[0]]).as_matrix()
    t = obj_pose[4:7]
    return Robj.T @ (pts_world - t).T, Robj      # ((3,N) -> caller transposes), Robj


def robot_control_points(rt, q_pin):
    """(M, 3) world positions of the G1 correspondence control points at q_pin."""
    corr = rt.correspondence
    out = np.zeros((corr.link_idx.shape[0], 3))
    for i in range(corr.link_idx.shape[0]):
        link = corr.link_names[corr.link_idx[i]]
        Rw = rt.pin.body_rotation(q_pin, link)
        pw = rt.pin.body_position(q_pin, link)
        out[i] = pw + Rw @ corr.offset_local[i]
    return out


def query_entities(rt, pts_world, obj_pose, margin=None):
    """Return (object ContactField in object-local, floor ContactField in world)."""
    m = rt.smplx_ground_probe.margin if margin is None else margin
    local, _ = _world_to_object(pts_world, obj_pose)
    fobj = rt.object_sdf.query(local.T, m)               # object-local query
    fflr = floor_field(pts_world.astype(np.float32), m)  # world query
    return fobj, fflr
```

- [ ] **Step 4: Run** `PY -m pytest tests/test_interaction_dxp.py -q` → PASS.
  (If `object_interaction` for this clip doesn't load the assets, the test skips — acceptable; the deeper tasks use the same skip guard. Confirm the assets DO load for at least one task name so the suite exercises the path; if `sub3_largebox_003` has the `largebox_sdf.npz` + correspondence assets, it will run.)

- [ ] **Step 5: Commit**
```bash
git add src/test_socp/interaction.py tests/test_interaction_dxp.py
git commit -m "feat(test_socp): robot-side object/floor field query at control points"
```

---

### Task 2: Per-frame references from the source probe

**Files:** Modify `src/test_socp/interaction.py`; Test: append to `tests/test_interaction_dxp.py`

The reference `(d_ref, x_ref)` for robot control point i comes from the source probe field at human point `human_idx[i]`. For the object: `rt.smplx_ground_probe(t, ...)` gives the object-local field; for the floor: `floor_field(probe.points, margin)`.

- [ ] **Step 1: Write the failing test**

```python
def test_reference_extraction_aligns_with_control_points():
    rt = _rt()
    if rt.correspondence is None or rt.object_sdf is None:
        import pytest; pytest.skip("assets not present")
    from HoloNew.src.test_socp.interaction import frame_references
    d_obj_ref, x_obj_ref, d_flr_ref = frame_references(rt, t=0)
    M = rt.correspondence.link_idx.shape[0]
    assert d_obj_ref.shape == (M,) and x_obj_ref.shape == (M, 3) and d_flr_ref.shape == (M,)
```

- [ ] **Step 2: Run** → FAIL (`frame_references` missing).

- [ ] **Step 3: Implement `frame_references`** (the probe must be evaluated at frame t; reuse the same probe call `retarget` already makes, indexing by `human_idx`):

```python
def frame_references(rt, t):
    """Per-control-point references: (d_obj_ref (M,), x_obj_ref (M,3) object-local, d_flr_ref (M,))."""
    pf = rt.smplx_ground_probe(t, rt.human_quat[t], rt.gmr_grounded[:, 0][t])
    hi = rt.correspondence.human_idx
    d_obj_ref = pf.field.distance[hi]
    x_obj_ref = pf.field.witness[hi]
    d_flr_ref = floor_field(pf.points, rt.smplx_ground_probe.margin).distance[hi]
    return d_obj_ref, x_obj_ref, d_flr_ref
```
(Confirm `rt.human_quat` and `rt.gmr_grounded` are the same arrays `retarget` passes to the probe — read `retarget`'s probe call at the top of the loop and mirror its arguments exactly.)

- [ ] **Step 4: Run** → PASS.

- [ ] **Step 5: Commit**
```bash
git add src/test_socp/interaction.py tests/test_interaction_dxp.py
git commit -m "feat(test_socp): per-frame interaction references from the source probe"
```

---

### Task 3: D + X residual assembly (no cross-frame state)

**Files:** Modify `src/test_socp/interaction.py`; Test: append.

For each control point i and entity j, with frozen normal `n0` and activation `α`,
the orthogonal-projection residuals (design): D uses `n0ᵀ (J_i dqa)` toward
`(d_ref − d0)`; X uses `Π0 (J_i dqa)` toward `Π0 (x_ref − x0)`, `Π0 = I − n0 n0ᵀ`.
For the object, work in the object-local frame: the object-local point Jacobian is
`Robjᵀ J_i` (object driven ⇒ object fixed in the per-frame solve). For the floor,
work in world.

- [ ] **Step 1: Write the failing test** (the assembled term is a cvxpy expression; enabling it adds to the objective and the solve stays finite)

```python
def test_dx_terms_assemble_and_solve():
    import cvxpy as cp
    rt = _rt()
    if rt.correspondence is None or rt.object_sdf is None:
        import pytest; pytest.skip("assets not present")
    from HoloNew.src.test_socp.interaction import build_dx_terms
    q = rt.q_init_full.copy()
    q_pin = rt.pin.qpos_mj_to_q_pin(q[:36])
    dqa = cp.Variable(rt.nv_a)
    obj_pose = np.array([1.0, 0, 0, 0, 0, 0, 0])
    terms = build_dx_terms(rt, q_pin, dqa, t=0, obj_pose=obj_pose,
                           lambda_D=1.0, lambda_X=1.0)
    assert isinstance(terms, list) and len(terms) >= 1
    prob = cp.Problem(cp.Minimize(cp.sum(terms) + cp.sum_squares(dqa)),
                      [cp.SOC(0.2, dqa)])
    prob.solve(solver=cp.CLARABEL)
    assert prob.status in ("optimal", "optimal_inaccurate")
```

- [ ] **Step 2: Run** → FAIL (`build_dx_terms` missing).

- [ ] **Step 3: Implement `build_dx_terms`**

```python
import cvxpy as cp

def _activation(d_ref, L):
    a = 1.0 - d_ref / L
    return np.where(d_ref < L, np.maximum(a, 0.0) ** 2, 0.0)

def build_dx_terms(rt, q_pin, dqa, t, obj_pose, lambda_D, lambda_X):
    corr = rt.correspondence
    P = robot_control_points(rt, q_pin)                       # (M,3) world
    fobj, fflr = query_entities(rt, P, obj_pose)
    d_obj_ref, x_obj_ref, d_flr_ref = frame_references(rt, t)
    Robj = R.from_quat([obj_pose[1], obj_pose[2], obj_pose[3], obj_pose[0]]).as_matrix()
    Lobj = rt.smplx_ground_probe.margin
    Lflr = rt.smplx_ground_probe.margin
    terms = []
    M = corr.link_idx.shape[0]
    for i in range(M):
        link = corr.link_names[corr.link_idx[i]]
        Ji = rt.pin.point_translational_jacobian(q_pin, link, corr.offset_local[i])[:, rt.v_a_indices]
        # ----- object entity (object-local frame) -----
        a = _activation(d_obj_ref[i], Lobj)
        if a > 0 and fobj.active[i]:
            n0 = fobj.direction[i]                            # object-local unit normal
            Jloc = Robj.T @ Ji                               # object-local point Jacobian
            d0 = fobj.distance[i]; x0 = fobj.witness[i]
            # D: n0^T Jloc dqa toward (d_ref - d0)
            terms.append((lambda_D * a / Lobj**2) * cp.square(n0 @ (Jloc @ dqa) - (d_obj_ref[i] - d0)))
            # X: Pi0 (x_ref - x0) - Pi0 Jloc dqa
            Pi0 = np.eye(3) - np.outer(n0, n0)
            terms.append((lambda_X * a / Lobj**2) * cp.sum_squares(Pi0 @ (Jloc @ dqa) - Pi0 @ (x_obj_ref[i] - x0)))
        # ----- floor entity (world frame) -----
        a = _activation(d_flr_ref[i], Lflr)
        if a > 0 and fflr.active[i]:
            n0 = fflr.direction[i]
            d0 = fflr.distance[i]; x0 = fflr.witness[i]
            terms.append((lambda_D * a / Lflr**2) * cp.square(n0 @ (Ji @ dqa) - (d_flr_ref[i] - d0)))
            Pi0 = np.eye(3) - np.outer(n0, n0)
            # floor x_ref is world; use the floor witness of the probe point's world pos
            terms.append((lambda_X * a / Lflr**2) * cp.sum_squares(Pi0 @ (Ji @ dqa) - Pi0 @ (0.0)))
    return terms
```
(NOTE for the implementer: the floor X reference `x_ref` is the world witness of the SOURCE point on the floor; `frame_references` returns only `d_flr_ref`. Extend `frame_references` to also return the floor witness `x_flr_ref` (from `floor_field(pf.points, m).witness[hi]`) and use it in the floor X term exactly like the object. Fix the placeholder `0.0` above. Keep D/X orthogonal.)

- [ ] **Step 4: Run** → PASS (solve is feasible).

- [ ] **Step 5: Commit**
```bash
git add src/test_socp/interaction.py tests/test_interaction_dxp.py
git commit -m "feat(test_socp): D+X interaction residual assembly (orthogonal projection)"
```

---

### Task 4: Wire D + X into the solver (config weights + thread object pose)

**Files:** Modify `src/test_socp/config.py`, `src/test_socp/test_socp.py`; Test: append.

- [ ] **Step 1: Add config weights.** In `TestSocpRetargeterConfig` (`src/test_socp/config.py`) add: `lambda_D: float = 0.0`, `lambda_X: float = 0.0`, `lambda_P: float = 0.0`, `sigma_v: float = 0.05`. (Default 0.0 so the solve is unchanged until Task 6 turns them on after validation.) Thread them through `from_config` into the constructor and store on `self`.

- [ ] **Step 2: Thread + append in `solve_single_iteration`.** Add params `frame_idx` (already present) and `obj_pose=None`. After the existing tracking `obj_terms` are built and before `cp.Problem`, append:
```python
        if (self.lambda_D > 0 or self.lambda_X > 0) and getattr(self, "correspondence", None) is not None \
                and getattr(self, "object_sdf", None) is not None and obj_pose is not None:
            from HoloNew.src.test_socp.interaction import build_dx_terms
            q_pin = self.pin.qpos_mj_to_q_pin(q[:36])
            obj_terms += build_dx_terms(self, q_pin, dqa, frame_idx, obj_pose,
                                        self.lambda_D, self.lambda_X)
```
In `retarget`, pass `obj_pose=self._obj_poses_raw[t]` (the raw `[qw,qx,qy,qz,x,y,z]` pose the probe uses) into the `iterate` calls; thread `obj_pose` through `iterate` → `solve_single_iteration`. Load `self._obj_poses_raw` in `from_config` (the `obj_poses` already loaded for the probe — store it on `rt`).

- [ ] **Step 3: Test** (with weights on, the solve completes on the object clip):
```python
def test_solver_with_dx_weights_runs():
    import pytest, numpy as np
    rt = _rt()
    if rt.correspondence is None or rt.object_sdf is None:
        pytest.skip("assets not present")
    rt.lambda_D = 1.0; rt.lambda_X = 1.0
    res = rt.retarget()
    assert np.all(np.isfinite(res.qpos))
```

- [ ] **Step 4: Run** `PY -m pytest tests/test_interaction_dxp.py tests/test_test_socp_parity.py -q` → PASS (parity holds because weights default 0.0; the new test sets them on and just checks finiteness).

- [ ] **Step 5: Commit**
```bash
git add src/test_socp/config.py src/test_socp/test_socp.py tests/test_interaction_dxp.py
git commit -m "feat(test_socp): wire D+X interaction terms into the solve (config weights, default off)"
```

---

### Task 5: P (contact persistence) with cross-frame state

**Files:** Modify `src/test_socp/interaction.py`, `src/test_socp/test_socp.py`; Test: append.

P needs the previous-frame robot control-point world positions `p_{i,t-1}`, the
previous source activation `α^{t-1}`, the previous **solved** robot-side distance
for `α̂^{t-1}`, and the reference tangential displacement `Δp_i^ref`.

- [ ] **Step 1: Write the failing test**

```python
def test_p_term_assembles():
    import cvxpy as cp, numpy as np, pytest
    rt = _rt()
    if rt.correspondence is None or rt.object_sdf is None:
        pytest.skip("assets not present")
    from HoloNew.src.test_socp.interaction import build_p_terms
    q = rt.q_init_full.copy(); q_pin = rt.pin.qpos_mj_to_q_pin(q[:36])
    dqa = cp.Variable(rt.nv_a)
    M = rt.correspondence.link_idx.shape[0]
    state = {"p_prev": np.zeros((M, 3)), "d_prev": np.full(M, 1e3), "a_prev": np.zeros(M)}
    terms = build_p_terms(rt, q_pin, dqa, t=1, obj_pose=np.array([1.,0,0,0,0,0,0]),
                          state=state, lambda_P=1.0, sigma_v=0.05, dt=1/30)
    assert isinstance(terms, list)
```

- [ ] **Step 2: Run** → FAIL (`build_p_terms` missing).

- [ ] **Step 3: Implement `build_p_terms`** (object channel shown; floor analogous):
```python
def build_p_terms(rt, q_pin, dqa, t, obj_pose, state, lambda_P, sigma_v, dt):
    corr = rt.correspondence
    P = robot_control_points(rt, q_pin)
    fobj, _ = query_entities(rt, P, obj_pose)
    d_obj_ref, x_obj_ref, _ = frame_references(rt, t)
    d_obj_ref_prev, x_obj_ref_prev, _ = frame_references(rt, t - 1)
    Robj = R.from_quat([obj_pose[1], obj_pose[2], obj_pose[3], obj_pose[0]]).as_matrix()
    Lobj = rt.smplx_ground_probe.margin
    terms = []
    for i in range(corr.link_idx.shape[0]):
        a_t = _activation(d_obj_ref[i], Lobj)
        a_tm1 = state["a_prev"][i]
        a_hat = (max(1.0 - state["d_prev"][i] / Lobj, 0.0)) ** 2 if state["d_prev"][i] < Lobj else 0.0
        gamma = min(a_t, a_tm1, a_hat)
        if gamma <= 0 or not fobj.active[i]:
            continue
        link = corr.link_names[corr.link_idx[i]]
        Ji = rt.pin.point_translational_jacobian(q_pin, link, corr.offset_local[i])[:, rt.v_a_indices]
        n0 = fobj.direction[i]; Pi0 = np.eye(3) - np.outer(n0, n0)
        dp = (P[i] - state["p_prev"][i]) + (Ji @ dqa)          # linearized Δp_i (world; object driven)
        dp_ref = x_obj_ref[i] - x_obj_ref_prev[i]              # reference tangential displacement
        # both projected by the (world-rotated) normal; object-driven ⇒ Robj cancels in Pi0 world
        terms.append((lambda_P / (sigma_v * dt) ** 2) * cp.sum_squares(Pi0 @ dp - Pi0 @ dp_ref))
    return terms
```
(The implementer must confirm the frame of `Δp_i` vs `Δp_i^ref`: keep both in the SAME frame — if `x_obj_ref` is object-local, rotate `dp` to object-local via `Robj.T`, or rotate `dp_ref` to world via `Robj`. Pick one consistently and document it. `dt = 1/30` unless the clip's frame rate differs — read it from the motion config.)

- [ ] **Step 4: Thread state through `retarget`.** Maintain `state` dict across frames: after each solved frame, set `p_prev` = current robot control points at the solved `q`, `d_prev` = solved robot-side object distance (`query_entities(...).distance`), `a_prev` = current source activation. Pass `state` into the iterate/solve calls when `lambda_P > 0`. Append the P terms in `solve_single_iteration` like the D/X block (guarded by `self.lambda_P > 0` and `frame_idx >= 1`).

- [ ] **Step 5: Test + run** `PY -m pytest tests/test_interaction_dxp.py tests/test_test_socp_parity.py -q` → PASS (parity holds, weights default 0).

- [ ] **Step 6: Commit**
```bash
git add src/test_socp/interaction.py src/test_socp/test_socp.py tests/test_interaction_dxp.py
git commit -m "feat(test_socp): contact persistence (P) term with cross-frame state (default off)"
```

---

### Task 6: Metric validation + turn on + re-baseline + smoke

**Files:** Modify `src/test_socp/config.py` (defaults), `tests/test_interaction_dxp.py`, `tests/test_test_socp_parity.py`.

- [ ] **Step 1: Metric test (the acceptance gate).** Solve the object clip twice — once with `lambda_D=lambda_X=0` (baseline) and once with them on — and assert the mean object-contact gap improves:
```python
def test_dx_reduces_contact_gap():
    import numpy as np, pytest
    from HoloNew.src.test_socp.interaction import robot_control_points, query_entities, frame_references
    def mean_obj_gap(rt):
        res = rt.retarget(); hi = rt.correspondence.human_idx
        gaps = []
        for t in range(res.qpos.shape[0]):
            q_pin = rt.pin.qpos_mj_to_q_pin(res.qpos[t, :36])
            P = robot_control_points(rt, q_pin)
            fobj, _ = query_entities(rt, P, rt._obj_poses_raw[t])
            d_ref, _, _ = frame_references(rt, t)
            m = (d_ref < rt.smplx_ground_probe.margin)
            if m.any():
                gaps.append(np.mean(np.abs(fobj.distance[m] - d_ref[m])))
        return float(np.mean(gaps))
    rt0 = _rt()
    if rt0.correspondence is None or rt0.object_sdf is None: pytest.skip("assets")
    base = mean_obj_gap(rt0)
    rt1 = _rt(); rt1.lambda_D = 1.0; rt1.lambda_X = 1.0
    on = mean_obj_gap(rt1)
    assert on < base, f"D/X did not reduce contact gap: {on:.4f} >= {base:.4f}"
```
Run it. If it does not improve, TUNE `lambda_D`/`lambda_X` (and verify the residual signs) until the gap genuinely shrinks — this is the brick's reason to exist; do not weaken the assertion. Record the observed base/on values in a comment.

- [ ] **Step 2: Turn the weights on by default.** Set the validated `lambda_D`, `lambda_X`, `lambda_P` defaults in `TestSocpRetargeterConfig` to the tuned values (so the new default solve includes interaction).

- [ ] **Step 3: Re-baseline parity.** With the weights on by default, the `robot_only` parity solve is unchanged ONLY IF interaction is inactive for robot_only (no object). Confirm: for `robot_only` there is no object_sdf / correspondence may still load. If interaction now changes the robot_only default solve, re-record `tests/test_test_socp_parity.py::BASELINE` deliberately (sanity-check sane root poses), keep atol=1e-6. If robot_only has no object and floor-only interaction changes it, that is the intended brick effect — re-baseline.

- [ ] **Step 4: Full regression + smoke.**
```
PY -m pytest tests/test_interaction_dxp.py tests/test_test_socp_parity.py tests/test_holosoma_constraints.py tests/test_pin_solver_swap.py -q
```
Expect PASS. Confirm the object clip solve completes with interaction on (finite, no infeasibility).

- [ ] **Step 5: Commit**
```bash
git add src/test_socp/config.py tests/test_interaction_dxp.py tests/test_test_socp_parity.py
git commit -m "feat(test_socp): enable interaction D/X/P by default after metric validation + re-baseline"
```

---

## Self-review notes
- **Spec coverage:** robot-side query (T1), references (T2), D+X assembly (T3), solver wiring + config (T4), P persistence + cross-frame state (T5), metric gate + turn-on + re-baseline (T6). Object + floor entities; objects driven (W^o deferred). Orthogonal-projection residuals per the design.
- **Naming consistency:** `interaction.py` exports `robot_control_points`, `query_entities`, `frame_references`, `build_dx_terms`, `build_p_terms`; config weights `lambda_D/lambda_X/lambda_P/sigma_v`; reuse `rt.pin.point_translational_jacobian` + `rt.v_a_indices`; `rt._obj_poses_raw` for the per-frame object pose.
- **Open items the implementer must confirm (do not guess):** (a) the exact attribute names on `smplx_ground_probe` for the per-frame object pose (`obj_quat`/`obj_trans` vs stored array) and `margin`. (b) `rt.human_quat` / `rt.gmr_grounded` arguments to the probe — mirror `retarget`'s real call. (c) floor X reference witness (extend `frame_references`). (d) the P-term frame consistency (Δp vs Δp^ref same frame). (e) the clip frame rate for `dt`. (f) whether `robot_only` loads correspondence/object assets (affects the parity re-baseline in T6).
