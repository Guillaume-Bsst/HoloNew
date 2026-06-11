# GMR-SOCP Retargeter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second, standalone GMR-style retargeter (two identical versions v1/v2) whose resolution is a copy of holosoma's SOCP solve with the objective and constraints changed to match GMR/mink (body-frame tracking cost, joint-limits + trust-region only), shown alongside the native trajectory in the same viser.

**Architecture:** A new isolated package `src/gmr_socp/` with shared support (`tables.py` = GMR IK tables ported from test_pipe; `targets.py` = per-body SE3 targets from positions + loaded quaternions + table offsets) and two standalone retargeters (`gmr_socp_v1.py`, `gmr_socp_v2.py`). Each retargeter is a focused copy of the native solve stack (model load, Jacobians, cvxpy/CLARABEL SQP) with a GMR body-tracking objective (position + linearized orientation) and a two-pass per-frame solve (table1 → table2). The native `InteractionMeshRetargeter` is never modified.

**Tech Stack:** Python, numpy, mujoco, cvxpy/CLARABEL, scipy, pytest. Solve runs in the `holonew` conda env.

**Reference spec:** `docs/specs/2026-06-11-gmr-socp-retargeter-design.md`

## Critical environment (for every task)

- Repo root: `/home/gbesset/Documents/wbt_rl/modules/01_retargeting/HoloNew`; package dir (cwd for commands): `.../HoloNew/HoloNew`.
- Package imports as `HoloNew`. Run on a feature branch off `main` (the controller creates it).
- **Always use this Python:** `/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python` (never bare `python`).
- Commits: local git identity is `Guillaume-Bsst`; **never add any Co-Authored-By/Claude mention**; comments/docs in **English**.
- The source of the GMR tables is the read-only file
  `/home/gbesset/Documents/wbt_rl/modules/third_party/test_pipe/src/test_pipe_retargeting/test_pipe_retargeting/solver/gmr/tables.py`.

## File Structure

- Create: `src/gmr_socp/__init__.py`
- Create: `src/gmr_socp/tables.py` — GMR IK match tables + human-body indices (ported). Shared.
- Create: `src/gmr_socp/targets.py` — quaternion loading + per-frame SE3 target builder. Shared.
- Create: `src/gmr_socp/gmr_socp_v1.py` — `GmrSocpRetargeterV1`, standalone solve copy + GMR objective + 2-pass.
- Create: `src/gmr_socp/gmr_socp_v2.py` — literal copy of v1 (`GmrSocpRetargeterV2`).
- Modify: `src/stages.py` — add the two GMR stages.
- Modify: `examples/view_stages.py` — run native + both GMR versions, bind all.
- Tests: `tests/test_gmr_tables.py`, `tests/test_gmr_targets.py`, `tests/test_gmr_socp.py`, `tests/test_gmr_orientation.py`.

---

## Task 1: Port the GMR IK tables

**Files:** Create `src/gmr_socp/__init__.py` (empty), `src/gmr_socp/tables.py`; Test `tests/test_gmr_tables.py`.

- [ ] **Step 1: Write the failing test**

The ported indices must line up with holosoma's `SMPLH_DEMO_JOINTS`. This test pins that contract (verified: idx 0=Pelvis, 3=L_Ankle, 11=Chest, 36=R_Wrist).

```python
# tests/test_gmr_tables.py
from HoloNew.src.gmr_socp.tables import (
    IK_MATCH_TABLE1, IK_MATCH_TABLE2, HUMAN_BODY_TO_IDX,
)
from HoloNew.config_types.data_type import SMPLH_DEMO_JOINTS

# Expected smplh joint name behind each GMR human-body name (holosoma naming).
_EXPECTED = {
    "pelvis": "Pelvis", "left_hip": "L_Hip", "left_knee": "L_Knee",
    "left_foot": "L_Ankle", "right_hip": "R_Hip", "right_knee": "R_Knee",
    "right_foot": "R_Ankle", "spine3": "Chest",
    "left_shoulder": "L_Shoulder", "left_elbow": "L_Elbow", "left_wrist": "L_Wrist",
    "right_shoulder": "R_Shoulder", "right_elbow": "R_Elbow", "right_wrist": "R_Wrist",
}

def test_human_body_indices_match_smplh_demo_joints():
    for body, idx in HUMAN_BODY_TO_IDX.items():
        assert SMPLH_DEMO_JOINTS[idx] == _EXPECTED[body], (body, idx, SMPLH_DEMO_JOINTS[idx])

def test_tables_have_same_robot_frames():
    assert set(IK_MATCH_TABLE1) == set(IK_MATCH_TABLE2)

def test_table_row_shape():
    for table in (IK_MATCH_TABLE1, IK_MATCH_TABLE2):
        for frame, (human, pos_w, rot_w, pos_off, rot_off) in table.items():
            assert human in HUMAN_BODY_TO_IDX
            assert len(pos_off) == 3 and len(rot_off) == 4
```

- [ ] **Step 2: Run it, confirm failure**

Run: `/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_gmr_tables.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'HoloNew.src.gmr_socp.tables'`).

- [ ] **Step 3: Create the module**

Copy the table definitions **verbatim** from the test_pipe file named in "Critical environment" above into `src/gmr_socp/tables.py`. Copy exactly: `IK_MATCH_TABLE1`, `IK_MATCH_TABLE2`, `HUMAN_BODY_TO_IDX`, `MAPPED_BODY_NAMES`, `MAPPED_BODY_BONES`, `HUMAN_SCALE_TABLE`, `HUMAN_ROOT_NAME`, `HUMAN_HEIGHT_ASSUMPTION`, `GROUND_HEIGHT`, `ROBOT_ROOT_NAME`, and the `_H`/`_K` constants. Keep the module docstring crediting GMR (YanjieZe/GMR). Do not change any numbers.

Create `src/gmr_socp/__init__.py` as an empty file.

- [ ] **Step 4: Run the tests, confirm PASS** (3 tests).

Run: `/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_gmr_tables.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/gmr_socp/__init__.py src/gmr_socp/tables.py tests/test_gmr_tables.py
git commit -m "feat(gmr): port GMR IK match tables (credit YanjieZe/GMR)"
```

---

## Task 2: Quaternion loading + per-frame SE3 targets

**Files:** Create `src/gmr_socp/targets.py`; Test `tests/test_gmr_targets.py`.

`targets.py` provides two things: (a) loading per-body quaternions from the OMOMO `.pt`, and (b) building, for each mapped robot frame and each table, the world-frame position + rotation target with the table offsets applied.

The OMOMO `.pt` layout (verified by test_pipe's `load_pt`): a torch save whose tensors include human joint world positions `(T, J, 3)` and per-joint quaternions `(T, J, 4)` in **wxyz**. The loader returns the quaternion array.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gmr_targets.py
import numpy as np
from HoloNew.src.gmr_socp.targets import build_frame_targets
from HoloNew.src.gmr_socp.tables import IK_MATCH_TABLE1, HUMAN_BODY_TO_IDX

def test_build_frame_targets_applies_offsets_and_maps_frames():
    J = 52
    pos = np.zeros((J, 3)); pos[0] = [1.0, 2.0, 3.0]           # pelvis position
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (J, 1))    # identity wxyz
    targets = build_frame_targets(pos, quat, IK_MATCH_TABLE1)
    # one entry per robot frame in the table
    assert set(targets) == set(IK_MATCH_TABLE1)
    p, R, w_p, w_r = targets["pelvis"]
    # pelvis pos_offset is [0,0,0]; with identity orientation the target == human pos
    np.testing.assert_allclose(p, [1.0, 2.0, 3.0], atol=1e-9)
    assert R.shape == (3, 3)
    assert (w_p, w_r) == (IK_MATCH_TABLE1["pelvis"][1], IK_MATCH_TABLE1["pelvis"][2])
```

- [ ] **Step 2: Run it, confirm failure** (`ModuleNotFoundError`).

Run: `/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_gmr_targets.py -v`

- [ ] **Step 3: Implement `targets.py`**

```python
# src/gmr_socp/targets.py
"""Build per-frame SE3 body targets for the GMR-SOCP objective.

For each robot frame in an IK match table, produce the world-frame target
position and rotation (with the table's pos_offset / rot_offset applied) plus the
position and orientation weights. Quaternions are wxyz throughout.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .tables import HUMAN_BODY_TO_IDX


def _wxyz_to_R(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def build_frame_targets(joint_pos: np.ndarray, joint_quat_wxyz: np.ndarray, table: dict):
    """joint_pos: (J,3) world positions; joint_quat_wxyz: (J,4) wxyz per joint.

    Returns dict: robot_frame -> (p_target(3,), R_target(3,3), pos_weight, rot_weight).
    The pos_offset is applied in the re-oriented (rot_offset-composed) body frame;
    the rot_offset is composed onto the human orientation (both per GMR's tables).
    """
    out = {}
    for frame, (human, pos_w, rot_w, pos_off, rot_off) in table.items():
        idx = HUMAN_BODY_TO_IDX[human]
        R_h = _wxyz_to_R(joint_quat_wxyz[idx])
        R_off = _wxyz_to_R(np.asarray(rot_off, dtype=float))
        R_target = R_h @ R_off
        p_target = joint_pos[idx] + R_target @ np.asarray(pos_off, dtype=float)
        out[frame] = (np.asarray(p_target, float), R_target, float(pos_w), float(rot_w))
    return out


def load_pt_quaternions(pt_path: str | Path) -> np.ndarray:
    """Load per-joint quaternions (T, J, 4) wxyz from an OMOMO .pt file.

    Mirrors test_pipe's load_pt. The exact key/layout must be confirmed against
    the real .pt at implementation time (see Step 4); this function isolates that
    so the rest of the pipeline does not depend on the file format.
    """
    import torch
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    # Confirm the quaternion field name/layout against test_pipe's load_pt and the
    # actual file (Step 4). Return a (T, J, 4) wxyz numpy array.
    raise NotImplementedError("Fill in after inspecting the .pt in Step 4")
```

- [ ] **Step 4: Inspect the real `.pt` and finish `load_pt_quaternions`**

Read test_pipe's loader for the exact field names:
`/home/gbesset/Documents/wbt_rl/modules/third_party/test_pipe/src/test_pipe_retargeting/test_pipe_retargeting/human/motion.py` (`load_pt`).
Then inspect the demo file:
```bash
/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -c "
import torch; d=torch.load('demo_data/OMOMO_new/sub3_largebox_003.pt', map_location='cpu', weights_only=False)
print(type(d)); print(d.keys() if hasattr(d,'keys') else 'seq', )
"
```
Replace the `raise NotImplementedError` with the real extraction (return `(T, J, 4)` wxyz). Add a test that loads the demo file and asserts shape `(T, 52, 4)` and unit-norm quaternions.

- [ ] **Step 5: Run tests, confirm PASS.** Then **Commit**

```bash
git add src/gmr_socp/targets.py tests/test_gmr_targets.py
git commit -m "feat(gmr): per-frame SE3 targets and .pt quaternion loading"
```

---

## Task 3: GMR-SOCP v1 — copied solve with GMR position objective (no orientation yet)

This is the largest task. Create `gmr_socp_v1.py` as a focused copy of the native solve stack, then replace the objective with GMR **position** tracking and reduce constraints to mink-equivalent. Orientation is added in Task 4.

**Files:** Create `src/gmr_socp/gmr_socp_v1.py`; Test `tests/test_gmr_socp.py`.

- [ ] **Step 1: Copy the native solver as the starting point**

```bash
cp src/interaction_mesh_retargeter.py src/gmr_socp/gmr_socp_v1.py
```
This intentionally copies the framework (model load, `_calc_manipulator_jacobians`,
`_calc_contact_jacobian_from_point`, `_build_transform_qdot_to_qvel_fast`, `iterate`,
`retarget_motion`, `_get_robot_link_positions`). You will now strip and rewire it.

- [ ] **Step 2: Strip to the solve stack and rename the class**

In `src/gmr_socp/gmr_socp_v1.py`:
- Rename the class `InteractionMeshRetargeter` → `GmrSocpRetargeterV1`.
- Fix the `src`-relative imports that the copy inherits: the original adds `parent.parent/"src"` to `sys.path`; update it to `Path(__file__).parent.parent` so `mujoco_utils`, `utils`, `viser_utils` still import (they live in `src/`).
- **Delete** the methods not needed for the mink-equivalent solve: all `draw_*`/`visualize_*`/`_setup_visualization`/`draw_mesh_*`, the self-collision methods (`_init_self_collision`, `_compute_self_collision_constraints`, `_draw_self_collision_geoms`, `_prefilter_pairs_with_mj_collision`, `_compute_jacobian_for_contact_relative`), and the foot-lock methods (`_init_foot_lock`, `_is_foot_locked_in_window`). Remove their calls from `__init__`.
- In `__init__`, set `self.visualize = False` always and remove the viser setup. Keep the mujoco model load, `q_a_indices`, joint-limit arrays (`q_a_lb/ub`), `step_size`, and `_calc_manipulator_jacobians` / `_calc_contact_jacobian_from_point` / `_build_transform_qdot_to_qvel_fast` / `_get_robot_link_positions`.

- [ ] **Step 3: Write the failing integration test**

```python
# tests/test_gmr_socp.py
import numpy as np
import pytest

@pytest.fixture(scope="module")
def demo_inputs():
    # Build the same inputs the native example builds, but only what the GMR
    # solver needs: human joint positions (T,J,3), quaternions (T,J,4), and the
    # robot constants. Reuse robot_retarget helpers where possible (read it).
    from HoloNew.examples.robot_retarget import RetargetingConfig
    return RetargetingConfig(task_type="robot_only", task_name="sub3_largebox_003",
                             data_format="smplh")

def test_gmr_v1_runs_and_tracks_pelvis(demo_inputs):
    from HoloNew.src.gmr_socp.gmr_socp_v1 import GmrSocpRetargeterV1
    rt = GmrSocpRetargeterV1.from_config(demo_inputs)
    result = rt.retarget()                  # returns RetargetResult (qpos (T,7+dof), stages={})
    assert result.qpos.shape[1] == 7 + rt.task_constants.ROBOT_DOF
    # pelvis of the robot should track the human pelvis within a loose tolerance
    pelvis_robot = result.qpos[:, :3]
    assert np.isfinite(pelvis_robot).all()
```

- [ ] **Step 4: Run it, confirm failure** (no `from_config` / `retarget`).

Run: `/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_gmr_socp.py -v`

- [ ] **Step 5: Add a GMR position-tracking solve**

Replace the body of `solve_single_iteration` with a GMR position objective. The native method already computes per-frame position Jacobians via
`self._calc_manipulator_jacobians(q, links=<robot_frames>, obj_frame=False)`
returning `J_dict[frame] (3 x nq_a)` and `p_dict[frame] (3,)`. Build:

```python
    def solve_single_iteration(self, q_locked, q_a_n_last, q_t_last, frame_targets, init_t=False):
        import cvxpy as cp
        q = np.copy(q_locked)
        q[self.q_a_indices] = q_a_n_last
        robot_frames = {f: self.robot_link_names[f] for f in frame_targets}  # frame -> mujoco body
        J_dict, p_dict, _ = self._calc_manipulator_jacobians(q, links=robot_frames, obj_frame=False)

        dqa = cp.Variable(self.nq_a, name="dqa")
        obj_terms = []
        for frame, (p_t, _R_t, w_p, _w_r) in frame_targets.items():
            if w_p == 0:
                continue
            J = J_dict[frame]                     # (3 x nq_a)
            r = p_t - p_dict[frame]               # position residual (target - current)
            obj_terms.append(w_p * cp.sum_squares(J @ dqa - r))

        constraints = [cp.SOC(self.step_size, dqa)]          # trust region
        if self.activate_joint_limits:
            constraints += [dqa >= (self.q_a_lb - q_a_n_last), dqa <= (self.q_a_ub - q_a_n_last)]

        prob = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)
        prob.solve(solver=cp.CLARABEL)
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"GMR-SOCP solve failed: {prob.status}")
        q_star = np.copy(q)
        q_star[self.q_a_indices] = dqa.value + q_a_n_last
        q_star[3:7] /= np.linalg.norm(q_star[3:7]) + 1e-12
        return q_star, float(prob.value)
```

Add `self.robot_link_names` — the map `gmr_frame_name -> mujoco body name`. The GMR table keys ARE the G1 mujoco body names (e.g. `left_toe_link`, `pelvis`), so `robot_link_names = {f: f for f in IK_MATCH_TABLE1}`. Confirm each key exists as a mujoco body (`mujoco.mj_name2id`); if a name differs (e.g. `pelvis` vs `pelvis_link`), map it. Add an assertion in `__init__` that every frame resolves to a valid body id.

- [ ] **Step 6: Add the two-pass `iterate` + `retarget` driver**

```python
    def iterate(self, q_locked, q_n, q_t_last, frame_targets, n_iter=10):
        last = np.inf
        for _ in range(n_iter):
            q_n, cost = self.solve_single_iteration(q_locked, q_n[self.q_a_indices], q_t_last, frame_targets)
            if np.isclose(cost, last):
                break
            last = cost
        return q_n, cost

    def retarget(self):
        from HoloNew.src.retarget_result import RetargetResult
        from .tables import IK_MATCH_TABLE1, IK_MATCH_TABLE2
        from .targets import build_frame_targets
        T = self.human_pos.shape[0]
        q = np.copy(self.q_init_full)
        out = []
        for t in range(T):
            tg1 = build_frame_targets(self.human_pos[t], self.human_quat[t], IK_MATCH_TABLE1)
            tg2 = build_frame_targets(self.human_pos[t], self.human_quat[t], IK_MATCH_TABLE2)
            q, _ = self.iterate(q, q, q, tg1, n_iter=(50 if t == 0 else 10))   # pass 1
            q, _ = self.iterate(q, q, q, tg2, n_iter=(50 if t == 0 else 10))   # pass 2
            out.append(np.copy(q))
        return RetargetResult(qpos=np.array(out), stages={}, cost=0.0)
```

Add a `from_config(cls, cfg)` classmethod that builds the retargeter and loads `self.human_pos (T,J,3)` and `self.human_quat (T,J,4)` and `self.q_init_full` (7+dof, pelvis-initialised). Reuse the input-prep helpers from `examples/robot_retarget.py` (preprocess + `initialize_robot_pose`) read-only to obtain `human_pos` and `q_init`; load quaternions via `targets.load_pt_quaternions`.

- [ ] **Step 7: Run the integration test, confirm PASS.**

Run: `/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_gmr_socp.py -v`
Expected: PASS (runs the full sequence; may take ~1-2 min). If the solve diverges or a body name is unresolved, fix the `robot_link_names` mapping.

- [ ] **Step 8: Commit**

```bash
git add src/gmr_socp/gmr_socp_v1.py tests/test_gmr_socp.py
git commit -m "feat(gmr): GMR-SOCP v1 with position tracking and two-pass solve"
```

---

## Task 4: Add orientation tracking to v1

The native code has **no rotational Jacobian**. Add one (`mujoco.mj_jacBody` gives `jacr`, the world-frame angular-velocity Jacobian) and an orientation residual. Correctness (sign/frame) is pinned by an empirical convergence test rather than by inspection.

**Files:** Modify `src/gmr_socp/gmr_socp_v1.py`; Test `tests/test_gmr_orientation.py`.

- [ ] **Step 1: Write the empirical convergence test**

This test does NOT assert a formula; it asserts the orientation error to a fixed target **decreases** across iterations on a single tracked frame — which fails loudly if the Jacobian sign/frame is wrong.

```python
# tests/test_gmr_orientation.py
import numpy as np
from scipy.spatial.transform import Rotation

def test_orientation_error_decreases(tmp_path):
    from HoloNew.examples.robot_retarget import RetargetingConfig
    from HoloNew.src.gmr_socp.gmr_socp_v1 import GmrSocpRetargeterV1
    rt = GmrSocpRetargeterV1.from_config(
        RetargetingConfig(task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    frame = "left_elbow_link"
    body = rt.robot_link_names[frame]
    q = np.copy(rt.q_init_full)
    # target = current orientation rotated 30deg about z
    R_cur0 = rt.body_rotation(q, body)
    R_tgt = R_cur0 @ Rotation.from_euler("z", 30, degrees=True).as_matrix()
    p_tgt = rt.body_position(q, body)
    targets = {frame: (p_tgt, R_tgt, 0.0, 10.0)}   # orientation-only
    errs = []
    for _ in range(15):
        q, _ = rt.solve_single_iteration(q, q[rt.q_a_indices], q, targets)
        R_cur = rt.body_rotation(q, body)
        errs.append(np.linalg.norm(Rotation.from_matrix(R_cur.T @ R_tgt).as_rotvec()))
    assert errs[-1] < errs[0] * 0.5, errs   # error at least halved
```

- [ ] **Step 2: Run it, confirm failure** (no `body_rotation`/`body_position`; orientation term not in the objective).

- [ ] **Step 3: Add rotational Jacobian + helpers + orientation term**

Add to `GmrSocpRetargeterV1`:
```python
    def _body_jac(self, q, body_name):
        """World-frame (jacp, jacr) for a body, reduced to actuated columns."""
        import mujoco
        self.robot_data.qpos[:] = q
        mujoco.mj_forward(self.robot_model, self.robot_data)
        bid = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        jacp = np.zeros((3, self.robot_model.nv)); jacr = np.zeros((3, self.robot_model.nv))
        mujoco.mj_jacBody(self.robot_model, self.robot_data, jacp, jacr, bid)
        # nv columns are velocity-space; map to q_a via the existing qdot->qvel transform
        T = self._build_transform_qdot_to_qvel_fast()        # (nv x nq) already used by native
        return (jacp @ T)[:, self.q_a_indices], (jacr @ T)[:, self.q_a_indices]

    def body_position(self, q, body_name):
        import mujoco
        self.robot_data.qpos[:] = q; mujoco.mj_forward(self.robot_model, self.robot_data)
        return self.robot_data.xpos[mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_name)].copy()

    def body_rotation(self, q, body_name):
        import mujoco
        self.robot_data.qpos[:] = q; mujoco.mj_forward(self.robot_model, self.robot_data)
        bid = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.robot_data.xmat[bid].reshape(3, 3).copy()
```
In `solve_single_iteration`, add the orientation residual for frames with `w_r > 0`. World-frame convention: `Jr @ dqa ≈ ω` (small world-frame rotation vector); the target rotation error in the world frame is `e = log(R_target @ R_current.T)` (a rotvec):
```python
        for frame, (p_t, R_t, w_p, w_r) in frame_targets.items():
            body = self.robot_link_names[frame]
            Jp, Jr = self._body_jac(q, body)
            if w_p > 0:
                obj_terms.append(w_p * cp.sum_squares(Jp @ dqa - (p_t - self.body_position(q, body))))
            if w_r > 0:
                R_c = self.body_rotation(q, body)
                e = Rotation.from_matrix(R_t @ R_c.T).as_rotvec()    # world-frame error
                obj_terms.append(w_r * cp.sum_squares(Jr @ dqa - e))
```
(Replace the position-only loop from Task 3 with this combined loop. Keep `_calc_manipulator_jacobians` available but the orientation path uses `_body_jac`; you may use `_body_jac` for both for consistency.)

- [ ] **Step 4: Run the convergence test.** If it FAILS (error grows), flip the world-frame convention to `e = log(R_c.T @ R_t)` with `Jr` interpreted in body frame, i.e. try `e = Rotation.from_matrix(R_c.T @ R_t).as_rotvec()` and/or use the body-frame jacr. Iterate sign/frame until the error at least halves. **This empirical test is the spec of correctness for the orientation term.**

Run: `/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_gmr_orientation.py -v`

- [ ] **Step 5: Re-run the v1 integration test** (`tests/test_gmr_socp.py`) — still PASS with the combined objective.

- [ ] **Step 6: Commit**

```bash
git add src/gmr_socp/gmr_socp_v1.py tests/test_gmr_orientation.py
git commit -m "feat(gmr): add orientation tracking (rotational Jacobian) to GMR-SOCP v1"
```

---

## Task 5: Stage registry + viewer integration

**Files:** Modify `src/stages.py`, `examples/view_stages.py`; Test `tests/test_stages.py` (append).

- [ ] **Step 1: Append the failing registry test**

```python
# tests/test_stages.py (append)
def test_gmr_stages_present_and_drive_robots():
    from HoloNew.src.stages import STAGE_SPECS, produces_qpos, key_for_label
    labels = [s.label for s in STAGE_SPECS]
    assert "GMR-SOCP v1" in labels and "GMR-SOCP v2" in labels
    assert produces_qpos("GMR-SOCP v1") and key_for_label("GMR-SOCP v1") == "gmr_socp_v1"
    assert produces_qpos("GMR-SOCP v2") and key_for_label("GMR-SOCP v2") == "gmr_socp_v2"
```

- [ ] **Step 2: Run it, confirm failure.**

- [ ] **Step 3: Add the two stages to `STAGE_SPECS` in `src/stages.py`**

```python
    StageSpec("GMR-SOCP v1", "gmr_socp_v1", True),
    StageSpec("GMR-SOCP v2", "gmr_socp_v2", True),
```
(Append after the existing `SOCP` entry.)

- [ ] **Step 4: Wire `examples/view_stages.py` to run native + both GMR versions**

After the native `run_headless(cfg)` result, also run the GMR retargeters and add their robots + qpos. Concretely: build `GmrSocpRetargeterV1.from_config(cfg)` and `V2`, call `.retarget()`, collect `{"gmr_socp_v1": res_v1.qpos, "gmr_socp_v2": res_v2.qpos}`. Pass all `produces_qpos` keys to the `Viewer` (already derived from `STAGE_SPECS`), and extend `Viewer.bind`/`_redraw` so a stage whose key is in an extra `qpos_by_stage` dict draws that trajectory's qpos (small change: `bind(result, extra_qpos=...)`; `_redraw` picks `self._extra_qpos[key]` when `spec.key` is a GMR key). Keep it minimal.

- [ ] **Step 5: Run the registry test + a bounded headless smoke of `view_stages`**

```bash
/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_stages.py -v
echo "" | timeout 280 /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python examples/view_stages.py --data_path demo_data/OMOMO_new --task-type robot_only --task-name sub3_largebox_003 --data_format smplh 2>&1 | tail -20
```
Expected: registry test PASS; smoke runs native + v1 + v2, opens viser, prints the viewer URL, exits cleanly, no Traceback.

- [ ] **Step 6: Commit**

```bash
git add src/stages.py examples/view_stages.py tests/test_stages.py
git commit -m "feat(gmr): show native + GMR-SOCP v1/v2 trajectories in the viewer"
```

---

## Task 6: Create v2 as a copy of v1

**Files:** Create `src/gmr_socp/gmr_socp_v2.py`; Test `tests/test_gmr_socp.py` (parameterize).

- [ ] **Step 1: Copy v1 to v2 and rename the class**

```bash
cp src/gmr_socp/gmr_socp_v1.py src/gmr_socp/gmr_socp_v2.py
```
Rename the class `GmrSocpRetargeterV1` → `GmrSocpRetargeterV2` in the new file (only the class name; everything else identical — v2 will diverge in a later increment).

- [ ] **Step 2: Parameterize the integration test over both classes**

Change `tests/test_gmr_socp.py` so `test_gmr_v1_runs_and_tracks_pelvis` is parameterized:
```python
import pytest
from HoloNew.src.gmr_socp.gmr_socp_v1 import GmrSocpRetargeterV1
from HoloNew.src.gmr_socp.gmr_socp_v2 import GmrSocpRetargeterV2

@pytest.mark.parametrize("cls", [GmrSocpRetargeterV1, GmrSocpRetargeterV2])
def test_gmr_runs_and_tracks_pelvis(demo_inputs, cls):
    rt = cls.from_config(demo_inputs)
    result = rt.retarget()
    assert result.qpos.shape[1] == 7 + rt.task_constants.ROBOT_DOF
    assert np.isfinite(result.qpos[:, :3]).all()
```

- [ ] **Step 3: Run the full suite**

Run: `/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/ -q`
Expected: all pass (native golden + stages + viewer + gmr tables/targets/socp/orientation, with the gmr integration parameterized over v1 and v2).

- [ ] **Step 4: Commit**

```bash
git add src/gmr_socp/gmr_socp_v2.py tests/test_gmr_socp.py
git commit -m "feat(gmr): add GMR-SOCP v2 as an identical copy of v1"
```

---

## Self-Review notes

- **Spec coverage:** tables port → Task 1; quat loading + targets → Task 2; copied solve + GMR cost + mink constraints (joint limits + trust-region, others disabled) + two-pass → Task 3; orientation tracking → Task 4; viewer stages (native + v1 + v2) → Task 5; v2 copy → Task 6. No step1/step2 stages (per spec). Native retargeter untouched (GMR is a separate copy) — the native golden test in `tests/test_retarget_golden.py` continues to guard it.
- **Empirically-pinned uncertainty:** the rotational-Jacobian sign/frame is verified by the convergence test in Task 4, not by inspection — the most likely source of a bug is isolated and self-checking.
- **Open items for the implementer:** the `.pt` quaternion field name (Task 2 Step 4); the exact `gmr_frame -> mujoco body` name resolution (Task 3 Step 5); whether `_build_transform_qdot_to_qvel_fast` returns `(nv x nq)` as assumed by `_body_jac` (verify and adapt); the `Viewer.bind` extra-qpos wiring (Task 5 Step 4). Each is localized and has a test that exercises it.
```
