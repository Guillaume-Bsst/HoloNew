# Brick 0 — MuJoCo → pinocchio kinematics migration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `TestSocpRetargeter`'s rigid-body kinematics/Jacobian/CoM layer with pinocchio, with `dqa` living in the pinocchio tangent space, keeping MuJoCo + coal only for collision/SDF.

**Architecture:** A new focused module `src/test_socp/pin_model.py` owns the pinocchio g1 model, the `qpos_mj ↔ q_pin` seam, and all kinematics (FK, frame Jacobians in the tangent space, point Jacobians, CoM + CoM Jacobian). `TestSocpRetargeter` delegates to it; the decision variable becomes the `nv`-sized tangent step integrated with `pin.integrate`, eliminating the `_build_transform_qdot_to_qvel_fast` quaternion bridge. Every kinematics helper is validated against MuJoCo before the solver swap; the end-to-end solve is re-baselined.

**Tech Stack:** Python, pinocchio (new dep), numpy, mujoco (collision/SDF only), cvxpy, pytest.

**Run from** `modules/01_retargeting/HoloNew/HoloNew` with `~/.holonew_deps/miniconda3/envs/holonew/bin/python` (alias `PY`).

**Design:** `docs/specs/2026-06-13-brick0-mujoco-to-pinocchio-design.md`.

## Read before coding (real code this plan migrates)
- `src/test_socp/test_socp.py`: `__init__` (`self.robot_model`/`robot_data` at 102-104; `q_a_init_idx`/`q_a_indices`/`nq_a` at 112-114; `q_a_lb/ub` at 129-137), the kinematics block `_build_transform_qdot_to_qvel_fast` (440), `_calc_contact_jacobian_from_point` (503), `_calc_manipulator_jacobians` (525), `_get_robot_link_positions` (580), `_body_jac` (594), `body_position` (622), `body_rotation` (637), and `solve_single_iteration` (656: `q[self.q_a_indices]=q_a_n_last` at 695, `dqa=cp.Variable(self.nq_a)` at 697, joint-limit rows 724-728, `q_star[self.q_a_indices]=dqa.value+q_a_n_last` + quat renorm at 804-806).
- For g1: `ROBOT_DOF = 29`, so today `nq_a = 7 + 29 = 36` (base 3 pos + 4 quat + 29 joints). Target tangent `nv_a = 6 + 29 = 35`.

---

### Task 1: Install and verify pinocchio in the env

**Files:** Test: `tests/test_pin_available.py`

- [ ] **Step 1: Install pinocchio into the holonew env**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/pip install pin`
Expected: installs `pin` (pinocchio) and its deps; exits 0.

- [ ] **Step 2: Write the smoke test**

```python
# tests/test_pin_available.py
def test_pinocchio_imports_and_has_freeflyer():
    import pinocchio as pin
    assert hasattr(pin, "JointModelFreeFlyer")
    assert hasattr(pin, "integrate")
    assert hasattr(pin, "jacobianCenterOfMass")
```

- [ ] **Step 3: Run it**

Run: `PY -m pytest tests/test_pin_available.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_pin_available.py
git commit -m "build(test_socp): add pinocchio dependency for the kinematics migration"
```

---

### Task 2: pinocchio g1 model loader

**Files:** Create `src/test_socp/pin_model.py`; Test: `tests/test_pin_model_build.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pin_model_build.py
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.src.test_socp.pin_model import PinModel


def test_pin_model_builds_freeflyer_g1():
    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    pm = PinModel(rt.task_constants.ROBOT_URDF_FILE)
    # free-flyer base: nq = 7 (3 pos + 4 quat) + 29 joints = 36; nv = 6 + 29 = 35
    assert pm.model.nq == 36
    assert pm.model.nv == 35
    # all g1 actuated joint names are present
    assert "left_hip_pitch_joint" in pm.joint_names
```

- [ ] **Step 2: Run it** — `PY -m pytest tests/test_pin_model_build.py -q` → FAIL (`pin_model` missing).

- [ ] **Step 3: Implement the loader**

```python
# src/test_socp/pin_model.py
"""pinocchio g1 model + MuJoCo<->pinocchio seam and kinematics.

Single rigid-body backend for TEST-SOCP: FK, frame Jacobians (tangent space),
point Jacobians, CoM and CoM Jacobian. MuJoCo/coal remain only for
collision/SDF. See docs/specs/2026-06-13-brick0-mujoco-to-pinocchio-design.md.
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin


class PinModel:
    def __init__(self, urdf_path: str):
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data = self.model.createData()
        # joint_names[0] is the universe; [1] is the free-flyer; rest are actuated.
        self.joint_names = [n for n in self.model.names]

    def neutral(self) -> np.ndarray:
        return pin.neutral(self.model)
```

- [ ] **Step 4: Run it** — `PY -m pytest tests/test_pin_model_build.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/test_socp/pin_model.py tests/test_pin_model_build.py
git commit -m "feat(test_socp): pinocchio g1 free-flyer model loader"
```

---

### Task 3: qpos_mj ↔ q_pin conversion seam

**Files:** Modify `src/test_socp/pin_model.py`; Test: `tests/test_pin_conversion.py`

The MuJoCo qpos for g1 is `[pos(3), quat wxyz(4), joints(29)]`; pinocchio's
free-flyer q is `[pos(3), quat xyzw(4), joints(29)]`. Joints share names (both
come from the same robot), but the per-joint **order** in MuJoCo (`jnt_qposadr`
order) may differ from pinocchio's; map by name.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pin_conversion.py
import numpy as np
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.src.test_socp.pin_model import PinModel


def _rt_pm():
    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    pm = PinModel(rt.task_constants.ROBOT_URDF_FILE)
    pm.bind_mujoco_order(rt.robot_model)  # establish the joint-name mapping
    return rt, pm


def test_qpos_roundtrip():
    rt, pm = _rt_pm()
    q_mj = rt.q_init_full[:36].copy()      # robot-only: 36 qpos
    q_pin = pm.qpos_mj_to_q_pin(q_mj)
    q_mj2 = pm.q_pin_to_qpos_mj(q_pin)
    np.testing.assert_allclose(q_mj2, q_mj, atol=1e-12)


def test_quaternion_reordered():
    rt, pm = _rt_pm()
    q_mj = rt.q_init_full[:36].copy()
    q_pin = pm.qpos_mj_to_q_pin(q_mj)
    # MuJoCo quat is wxyz at [3:7]; pinocchio is xyzw at [3:7]
    np.testing.assert_allclose(q_pin[3:7], q_mj[[4, 5, 6, 3]], atol=1e-12)
```

- [ ] **Step 2: Run it** — `PY -m pytest tests/test_pin_conversion.py -q` → FAIL (`bind_mujoco_order` missing).

- [ ] **Step 3: Implement the seam**

```python
# add to PinModel in src/test_socp/pin_model.py
import mujoco

def bind_mujoco_order(self, mj_model) -> None:
    """Map each actuated joint between MuJoCo qpos order and pinocchio q order."""
    # pinocchio actuated-joint qpos addresses (skip universe[0] + freeflyer[1]).
    self._pin_joint_qadr = {}
    for jid in range(2, self.model.njoints):
        name = self.model.names[jid]
        self._pin_joint_qadr[name] = self.model.joints[jid].idx_q
    # MuJoCo actuated-joint qpos addresses by name (skip the free base joint).
    self._mj_joint_qadr = {}
    for j in range(mj_model.njnt):
        if mj_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)
        self._mj_joint_qadr[name] = int(mj_model.jnt_qposadr[j])
    assert set(self._pin_joint_qadr) == set(self._mj_joint_qadr), \
        "MuJoCo and pinocchio joint name sets differ"

def qpos_mj_to_q_pin(self, q_mj: np.ndarray) -> np.ndarray:
    q = np.zeros(self.model.nq)
    q[0:3] = q_mj[0:3]                       # base position
    q[3:7] = q_mj[[4, 5, 6, 3]]              # wxyz -> xyzw
    for name, pin_adr in self._pin_joint_qadr.items():
        q[pin_adr] = q_mj[self._mj_joint_qadr[name]]
    return q

def q_pin_to_qpos_mj(self, q_pin: np.ndarray) -> np.ndarray:
    q = np.zeros(7 + len(self._mj_joint_qadr))
    q[0:3] = q_pin[0:3]
    q[3:7] = q_pin[[6, 3, 4, 5]]             # xyzw -> wxyz
    for name, pin_adr in self._pin_joint_qadr.items():
        q[self._mj_joint_qadr[name]] = q_pin[pin_adr]
    return q
```

- [ ] **Step 4: Run it** — `PY -m pytest tests/test_pin_conversion.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/test_socp/pin_model.py tests/test_pin_conversion.py
git commit -m "feat(test_socp): qpos_mj<->q_pin conversion seam (quat reorder + joint-name map)"
```

---

### Task 4: FK parity (body position + rotation) vs MuJoCo

**Files:** Modify `src/test_socp/pin_model.py`; Test: `tests/test_pin_fk_parity.py`

- [ ] **Step 1: Write the failing test** (pinocchio FK must match MuJoCo at sampled configs)

```python
# tests/test_pin_fk_parity.py
import numpy as np
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.src.test_socp.pin_model import PinModel


def _setup():
    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    pm = PinModel(rt.task_constants.ROBOT_URDF_FILE)
    pm.bind_mujoco_order(rt.robot_model)
    return rt, pm


def test_fk_position_rotation_match_mujoco():
    rt, pm = _setup()
    rng = np.random.default_rng(0)
    body = "left_ankle_roll_link"
    for _ in range(5):
        q_mj = rt.q_init_full[:36].copy()
        q_mj[7:] += 0.1 * rng.standard_normal(29)  # perturb joints
        p_mj = rt.body_position(q_mj, body)
        R_mj = rt.body_rotation(q_mj, body)
        q_pin = pm.qpos_mj_to_q_pin(q_mj)
        p_pin = pm.body_position(q_pin, body)
        R_pin = pm.body_rotation(q_pin, body)
        np.testing.assert_allclose(p_pin, p_mj, atol=1e-6)
        np.testing.assert_allclose(R_pin, R_mj, atol=1e-6)
```

- [ ] **Step 2: Run it** — FAIL (`PinModel.body_position` missing).

- [ ] **Step 3: Implement FK helpers**

```python
# add to PinModel
def _fk(self, q_pin: np.ndarray) -> None:
    pin.forwardKinematics(self.model, self.data, q_pin)
    pin.updateFramePlacements(self.model, self.data)

def _frame_id(self, body_name: str) -> int:
    return self.model.getFrameId(body_name)

def body_position(self, q_pin: np.ndarray, body_name: str) -> np.ndarray:
    self._fk(q_pin)
    return np.array(self.data.oMf[self._frame_id(body_name)].translation)

def body_rotation(self, q_pin: np.ndarray, body_name: str) -> np.ndarray:
    self._fk(q_pin)
    return np.array(self.data.oMf[self._frame_id(body_name)].rotation)
```

- [ ] **Step 4: Run it** — PASS. (If a body name is a link in the URDF but a frame lookup fails, the MuJoCo body name equals the URDF link name; confirm `getFrameId` resolves it — adjust the name mapping if the URDF uses a suffix.)

- [ ] **Step 5: Commit**

```bash
git add src/test_socp/pin_model.py tests/test_pin_fk_parity.py
git commit -m "feat(test_socp): pinocchio FK helpers with MuJoCo parity"
```

---

### Task 5: Body Jacobian parity (tangent space) vs MuJoCo

**Files:** Modify `src/test_socp/pin_model.py`; Test: `tests/test_pin_jac_parity.py`

MuJoCo's `mj_jacBody` gives a world-frame `(3×nv)` translational Jacobian wrt
qvel; today the solver maps it to qpos via `T`. pinocchio's
`computeFrameJacobian(..., LOCAL_WORLD_ALIGNED)` gives the world-aligned Jacobian
directly in the tangent space `nv`. The **translational** block must match
MuJoCo's `jacp` (which is already world-frame and tangent-space, before the `T`
bridge). Compare in tangent space, column-mapped MuJoCo qvel ↔ pinocchio v.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pin_jac_parity.py
import numpy as np
import mujoco
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.src.test_socp.pin_model import PinModel


def test_translational_frame_jacobian_matches_mujoco():
    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    pm = PinModel(rt.task_constants.ROBOT_URDF_FILE)
    pm.bind_mujoco_order(rt.robot_model)
    body = "left_ankle_roll_link"
    q_mj = rt.q_init_full[:36].copy()
    # MuJoCo world translational Jacobian wrt qvel (nv columns).
    rt.robot_data.qpos[:] = q_mj
    mujoco.mj_forward(rt.robot_model, rt.robot_data)
    bid = mujoco.mj_name2id(rt.robot_model, mujoco.mjtObj.mjOBJ_BODY, body)
    jacp = np.zeros((3, rt.robot_model.nv)); jacr = np.zeros((3, rt.robot_model.nv))
    mujoco.mj_jacBody(rt.robot_model, rt.robot_data, jacp, jacr, bid)
    # pinocchio world-aligned translational Jacobian (nv columns), column-mapped.
    q_pin = pm.qpos_mj_to_q_pin(q_mj)
    Jp_pin = pm.frame_translational_jacobian(q_pin, body)  # (3, nv) in MuJoCo qvel order
    np.testing.assert_allclose(Jp_pin, jacp, atol=1e-6)
```

- [ ] **Step 2: Run it** — FAIL (`frame_translational_jacobian` missing).

- [ ] **Step 3: Implement the Jacobian helper + tangent column map**

```python
# add to PinModel
def bind_mujoco_velocity_order(self, mj_model) -> None:
    """Map pinocchio v columns -> MuJoCo qvel columns (base 6 + joints by name)."""
    import mujoco
    self._pin_to_mj_v = np.zeros(self.model.nv, dtype=int)
    # base: pinocchio free-flyer v = [lin(3), ang(3)]; MuJoCo free dof = [lin(3), ang(3)].
    self._pin_to_mj_v[0:6] = np.arange(6)
    for jid in range(2, self.model.njoints):
        name = self.model.names[jid]
        pin_v = self.model.joints[jid].idx_v
        j_mj = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        self._pin_to_mj_v[pin_v] = int(mj_model.jnt_dofadr[j_mj])

def frame_translational_jacobian(self, q_pin, body_name) -> np.ndarray:
    """(3 x nv) world-aligned translational Jacobian, reordered to MuJoCo qvel cols."""
    pin.computeJointJacobians(self.model, self.data, q_pin)
    pin.updateFramePlacements(self.model, self.data)
    fid = self.model.getFrameId(body_name)
    J6 = pin.getFrameJacobian(self.model, self.data, fid, pin.LOCAL_WORLD_ALIGNED)
    Jp = np.asarray(J6[0:3, :])             # translational rows, pinocchio v order
    out = np.zeros((3, self.model.nv))
    out[:, self._pin_to_mj_v] = Jp          # reorder to MuJoCo qvel columns
    return out
```
(Call `pm.bind_mujoco_velocity_order(rt.robot_model)` after `bind_mujoco_order` — add it to the test setup; if Step 1 didn't call it, add the call there and re-run.)

- [ ] **Step 4: Run it** — PASS. (Angular Jacobian parity is implied by FK parity; if the centroidal brick needs `jacr` parity it adds its own test.)

- [ ] **Step 5: Commit**

```bash
git add src/test_socp/pin_model.py tests/test_pin_jac_parity.py
git commit -m "feat(test_socp): pinocchio frame translational Jacobian with MuJoCo parity"
```

---

### Task 6: Point Jacobian + CoM + CoM Jacobian helpers

**Files:** Modify `src/test_socp/pin_model.py`; Test: `tests/test_pin_point_com.py`

- [ ] **Step 1: Write the failing test** (point Jacobian via finite differences; CoM vs MuJoCo `subtree_com`)

```python
# tests/test_pin_point_com.py
import numpy as np
import mujoco
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.src.test_socp.pin_model import PinModel


def _setup():
    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    pm = PinModel(rt.task_constants.ROBOT_URDF_FILE)
    pm.bind_mujoco_order(rt.robot_model)
    pm.bind_mujoco_velocity_order(rt.robot_model)
    return rt, pm


def test_point_jacobian_matches_finite_difference():
    rt, pm = _setup()
    body = "left_ankle_roll_link"
    offset = np.array([0.02, -0.01, 0.03])
    q_mj = rt.q_init_full[:36].copy()
    q_pin = pm.qpos_mj_to_q_pin(q_mj)
    J = pm.point_translational_jacobian(q_pin, body, offset)   # (3, nv) MuJoCo qvel order
    # finite-difference the world point position along each MuJoCo qvel direction
    def point_world(qp):
        R = pm.body_rotation(qp, body); p = pm.body_position(qp, body)
        return p + R @ offset
    p0 = point_world(q_pin); eps = 1e-6
    for k in range(6):  # check the base 6 tangent dirs
        v = np.zeros(rt.robot_model.nv); v[k] = eps
        # map MuJoCo qvel dir back to a pinocchio tangent step
        v_pin = np.zeros(pm.model.nv); v_pin[pm._pin_to_mj_v == k] = eps
        qp1 = pinocchio_integrate(pm, q_pin, v_pin)
        fd = (point_world(qp1) - p0) / eps
        np.testing.assert_allclose(J[:, k], fd, atol=1e-4)


def pinocchio_integrate(pm, q_pin, v_pin):
    import pinocchio as pin
    return pin.integrate(pm.model, q_pin, v_pin)


def test_com_matches_mujoco():
    rt, pm = _setup()
    q_mj = rt.q_init_full[:36].copy()
    rt.robot_data.qpos[:] = q_mj
    mujoco.mj_forward(rt.robot_model, rt.robot_data)
    com_mj = rt.robot_data.subtree_com[0].copy()   # whole-body CoM at the world body
    q_pin = pm.qpos_mj_to_q_pin(q_mj)
    com_pin = pm.com(q_pin)
    np.testing.assert_allclose(com_pin, com_mj, atol=1e-4)
```

- [ ] **Step 2: Run it** — FAIL (`point_translational_jacobian` / `com` missing).

- [ ] **Step 3: Implement the helpers**

```python
# add to PinModel
def point_translational_jacobian(self, q_pin, body_name, offset_local) -> np.ndarray:
    """(3 x nv) world Jacobian of a point fixed on `body_name` at `offset_local`,
    reordered to MuJoCo qvel columns. J_point = J_frame_translation + (-[Rp]_x) J_frame_ang."""
    pin.computeJointJacobians(self.model, self.data, q_pin)
    pin.updateFramePlacements(self.model, self.data)
    fid = self.model.getFrameId(body_name)
    J6 = pin.getFrameJacobian(self.model, self.data, fid, pin.LOCAL_WORLD_ALIGNED)
    R = np.asarray(self.data.oMf[fid].rotation)
    rp = R @ np.asarray(offset_local, dtype=float)
    skew = np.array([[0, -rp[2], rp[1]], [rp[2], 0, -rp[0]], [-rp[1], rp[0], 0]])
    Jp = np.asarray(J6[0:3, :]) - skew @ np.asarray(J6[3:6, :])
    out = np.zeros((3, self.model.nv)); out[:, self._pin_to_mj_v] = Jp
    return out

def com(self, q_pin) -> np.ndarray:
    return np.array(pin.centerOfMass(self.model, self.data, q_pin))

def com_jacobian(self, q_pin) -> np.ndarray:
    """(3 x nv) CoM Jacobian, reordered to MuJoCo qvel columns."""
    Jc = np.asarray(pin.jacobianCenterOfMass(self.model, self.data, q_pin))
    out = np.zeros((3, self.model.nv)); out[:, self._pin_to_mj_v] = Jc
    return out
```

- [ ] **Step 4: Run it** — PASS. (If the CoM tolerance fails, the URDF and MJCF inertials differ; loosen to `atol=1e-3` and note it — masses come from the same source so they should agree closely.)

- [ ] **Step 5: Commit**

```bash
git add src/test_socp/pin_model.py tests/test_pin_point_com.py
git commit -m "feat(test_socp): pinocchio point Jacobian + CoM + CoM Jacobian (FD/MuJoCo validated)"
```

---

### Task 7: Swap the solver to the pinocchio backend (tangent-space dqa)

**Files:** Modify `src/test_socp/test_socp.py`; Test: `tests/test_pin_solver_swap.py`

This coordinated edit changes the kinematics backend AND the decision variable to
the tangent space. Make these changes together because they are coupled.

- [ ] **Step 1: Write a guard test** (active tangent size; one solved frame stays finite)

```python
# tests/test_pin_solver_swap.py
import numpy as np
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter


def test_tangent_decision_size_and_one_frame():
    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    # after migration the active decision variable is the tangent (nv_a),
    # base 6 + 29 joints = 35 with q_a_init_idx = -7.
    assert rt.nv_a == 35
    res = rt.retarget()
    assert np.all(np.isfinite(res.qpos))
    assert res.qpos.shape[1] >= 36   # output is still MuJoCo qpos
```

- [ ] **Step 2: Run it** — FAIL (`nv_a` missing).

- [ ] **Step 3: Make the coordinated edit.** In `TestSocpRetargeter.__init__`, after the MuJoCo model is built, construct the pinocchio backend and the tangent index set:

```python
        from HoloNew.src.test_socp.pin_model import PinModel
        self.pin = PinModel(task_constants.ROBOT_URDF_FILE)
        self.pin.bind_mujoco_order(self.robot_model)
        self.pin.bind_mujoco_velocity_order(self.robot_model)
        # tangent active indices: base is 6 (lin+ang) instead of 7 (pos+quat).
        # q_a_init_idx is expressed in the qpos convention; map to tangent.
        nv_base = 6
        self.v_a_indices = np.arange(nv_base + (self.q_a_init_idx + 7) - 7
                                     if self.q_a_init_idx < 0 else nv_base + self.q_a_init_idx,
                                     nv_base + task_constants.ROBOT_DOF)
        # for q_a_init_idx == -7 this is arange(0, 6+29) = all 35 tangent dofs.
        if self.q_a_init_idx == -7:
            self.v_a_indices = np.arange(0, nv_base + task_constants.ROBOT_DOF)
        self.nv_a = len(self.v_a_indices)
```

Replace the kinematics methods to delegate to `self.pin` (convert the incoming
MuJoCo `q` with `self.pin.qpos_mj_to_q_pin`):

```python
    def body_position(self, q, body_name):
        return self.pin.body_position(self.pin.qpos_mj_to_q_pin(q[:36]), body_name)

    def body_rotation(self, q, body_name):
        return self.pin.body_rotation(self.pin.qpos_mj_to_q_pin(q[:36]), body_name)

    def _body_jac(self, q, body_name):
        q_pin = self.pin.qpos_mj_to_q_pin(q[:36])
        Jp = self.pin.frame_translational_jacobian(q_pin, body_name)  # (3, nv) MuJoCo qvel order
        Jr = self.pin.frame_angular_jacobian(q_pin, body_name)        # add this helper (mirror of Jp using rows 3:6)
        return Jp[:, self.v_a_indices], Jr[:, self.v_a_indices]
```

Add `frame_angular_jacobian` to `PinModel` (rows `3:6` of the LOCAL_WORLD_ALIGNED
Jacobian, reordered like `frame_translational_jacobian`). Update
`_calc_contact_jacobian_from_point` and `_calc_manipulator_jacobians` to call
`self.pin.point_translational_jacobian(...)` and slice `self.v_a_indices`;
`_get_robot_link_positions` uses `self.pin.body_position`.

In `solve_single_iteration`: `dqa = cp.Variable(self.nv_a)`, the SOC trust region
unchanged (`cp.SOC(self.step_size, dqa)`), the joint-limit rows re-expressed in
tangent (see Task-7b note), and the integration becomes:

```python
        # integrate the tangent step into the full MuJoCo qpos
        v_full = np.zeros(self.pin.model.nv)
        v_full[self.v_a_indices] = dqa.value
        q_pin = self.pin.qpos_mj_to_q_pin(q[:36])
        q_pin_new = pin.integrate(self.pin.model, q_pin, v_full)
        q_star = np.copy(q)
        q_star[:36] = self.pin.q_pin_to_qpos_mj(q_pin_new)
        # (no manual quaternion renormalisation — pin.integrate keeps it unit)
```

Delete `_build_transform_qdot_to_qvel_fast` (now unused). All
`[:, self.q_a_indices]` Jacobian slices in `solve_single_iteration` (foot, self-
collision, non-penetration blocks) become `[:, self.v_a_indices]`; those
Jacobians already come from `self.pin.*` so they are in MuJoCo qvel/tangent order.

- [ ] **Step 3b: Re-express joint limits in tangent.** Replace the qpos-space
joint-limit rows with a tangent-space box using `pin.difference`:

```python
        if self.activate_joint_limits:
            q_pin = self.pin.qpos_mj_to_q_pin(q[:36])
            q_lb_pin = self.pin.qpos_mj_to_q_pin(self._q_full_lb)  # full lower-bound qpos
            q_ub_pin = self.pin.qpos_mj_to_q_pin(self._q_full_ub)
            lo = pin.difference(self.pin.model, q_pin, q_lb_pin)[self.v_a_indices]
            hi = pin.difference(self.pin.model, q_pin, q_ub_pin)[self.v_a_indices]
            constraints += [dqa >= lo, dqa <= hi]
```
Build `self._q_full_lb`/`self._q_full_ub` in `__init__` (full-length MuJoCo qpos
bound vectors; base components set to the current base ± a large value so the
free base stays unbounded, joints from the existing `complete_lower/upper`).

- [ ] **Step 4: Run it** — `PY -m pytest tests/test_pin_solver_swap.py -q` → PASS (finite output, tangent size correct).

- [ ] **Step 5: Commit**

```bash
git add src/test_socp/test_socp.py src/test_socp/pin_model.py tests/test_pin_solver_swap.py
git commit -m "feat(test_socp): swap solver kinematics to pinocchio, dqa in tangent space"
```

---

### Task 8: Re-baseline the parity snapshot + full regression + smoke

**Files:** Modify `tests/test_test_socp_parity.py`; full suite.

- [ ] **Step 1: Regenerate the parity baseline.** The kinematics backend changed,
so the frozen `BASELINE` in `tests/test_test_socp_parity.py` is now stale.
Run the solve once, capture `res.qpos[:3, :7]`, and paste it as the new
`BASELINE` literal (this is the **deliberate** re-baseline the meta-spec calls
for — record it, review that the values are sane robot root poses, not NaN).

Run: `PY -c "import numpy as np; from HoloNew.examples.robot_retarget import RetargetingConfig; from HoloNew.src.test_socp.test_socp import TestSocpRetargeter; rt=TestSocpRetargeter.from_config(RetargetingConfig(task_type='robot_only', task_name='sub3_largebox_003', data_format='smplh')); print(repr(rt.retarget().qpos[:3,:7]))"`
Paste the printed array as `BASELINE`. Keep `atol=1e-6`.

- [ ] **Step 2: Run the migrated regression**

Run: `PY -m pytest tests/test_test_socp_parity.py tests/test_holosoma_constraints.py tests/test_pin_fk_parity.py tests/test_pin_jac_parity.py tests/test_pin_point_com.py tests/test_pin_solver_swap.py -q`
Expected: PASS. (The holosoma-constraint tests confirm non-penetration / self-
collision / foot still work through the migrated point Jacobians.)

- [ ] **Step 3: Tracking-quality smoke.** Solve a full clip and assert the solved
pelvis tracks the ground reference within the same tolerance band as before the
migration (compute mean pelvis position error vs `gmr_ground['pos']`; assert it
is below a documented threshold, e.g. 5 cm). Add this as
`test_pin_tracking_quality` in `tests/test_pin_solver_swap.py`.

- [ ] **Step 4: Commit**

```bash
git add tests/test_test_socp_parity.py tests/test_pin_solver_swap.py
git commit -m "test(test_socp): re-baseline parity snapshot + tracking-quality smoke after pinocchio migration"
```

---

## Self-review notes
- **Spec coverage:** model loader (T2), `qpos↔q_pin` seam (T3), FK parity (T4),
  Jacobian parity (T5), point/CoM/CoM-Jac (T6), solver swap to tangent + integrate
  + joint limits + constraint slicing (T7), re-baseline + regression + tracking
  smoke (T8). MuJoCo kept for collision/SDF (untouched in this plan). pinocchio
  install (T1).
- **Naming consistency:** `PinModel`, `qpos_mj_to_q_pin` / `q_pin_to_qpos_mj`,
  `bind_mujoco_order` / `bind_mujoco_velocity_order`, `frame_translational_jacobian`
  / `frame_angular_jacobian`, `point_translational_jacobian`, `com` / `com_jacobian`,
  `self.pin`, `self.v_a_indices` / `self.nv_a`.
- **Open items to confirm while coding (do not guess):** (a) URDF link names vs
  MuJoCo body names — Task 4 surfaces any suffix mismatch; map if needed. (b) The
  free-flyer velocity layout (lin-then-ang) matches MuJoCo's free-joint dof layout
  — verified by Task 5's translational parity and Task 6's FD point-Jacobian test;
  if a sign/order differs on the angular base dofs, fix `bind_mujoco_velocity_order`.
  (c) CoM tolerance (Task 6) depends on URDF↔MJCF inertial agreement. (d) Whether
  `q[:36]` slicing is right when an object scene xml is loaded (object adds 7 qpos);
  for object tasks extend the conversion to carry the trailing object qpos, or keep
  the object outside the pinocchio model (it is driven, not part of the robot model)
  — the robot pin model is robot-only, so slice the robot qpos and leave the object
  qpos handled by MuJoCo as today.
