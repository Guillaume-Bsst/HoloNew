# Retargeting Scoreboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a scoreboard of retargeting-quality metrics (smoothness, effort, tracking, dynamics) as pure functions, wired into the offline eval sweep and shared with the in-process A/B metric tests.

**Architecture:** Pure metric functions in `evaluation/metrics/` take already-extracted numpy arrays. The offline `RetargetingEvaluator` extracts arrays (MuJoCo FK, subtree_com, model ranges, SMPL ref joints) and calls them; `test_*_metric.py` extract via `rt`/`RetargetResult` and call the same functions. Single source of truth per formula.

**Tech Stack:** numpy, scipy.spatial.transform (quaternions), pytest. Env: `/home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python`. Tests run from `HoloNew/` (relative model paths).

---

## Conventions

- Test runner: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest <path> -q`
- qpos columns: `[0:3]` base xyz, `[3:7]` base quat wxyz, `[7:7+dof]` joints, optional `[-7:]` object.
- accel = 2nd diff / dt²; jerk = 3rd diff / dt³.
- Commit messages: no Co-Authored-By / Claude trailer (user preference).

---

### Task 1: Package scaffold

**Files:**
- Create: `evaluation/metrics/__init__.py`

- [ ] **Step 1: Create the package init re-exporting the four entry points**

```python
"""Retargeting-quality metric functions (pure, array-in / dict-out)."""
from .smoothness import compute_smoothness
from .effort import compute_effort
from .tracking import compute_tracking
from .dynamics import compute_dynamics

__all__ = ["compute_smoothness", "compute_effort", "compute_tracking", "compute_dynamics"]
```

(Imports will fail until later tasks create the modules; commit after Task 5.)

---

### Task 2: smoothness.py

**Files:**
- Create: `evaluation/metrics/smoothness.py`
- Test: `tests/test_metrics_smoothness.py`

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
from HoloNew.evaluation.metrics.smoothness import compute_smoothness

def _qpos_const_vel(T=20, dof=5):
    t = np.arange(T)[:, None]
    base = np.hstack([0.01 * t * np.ones((T, 3)), np.tile([1, 0, 0, 0], (T, 1))])
    joints = 0.02 * t * np.ones((T, dof))
    return np.hstack([base, joints])

def test_constant_velocity_has_zero_accel_and_jerk():
    m = compute_smoothness(_qpos_const_vel(), dof=5, dt=1 / 30.0)
    assert m["joint_accel_rms"] < 1e-9
    assert m["joint_jerk_rms"] < 1e-9
    assert m["base_pos_accel_rms"] < 1e-9

def test_known_joint_accel():
    # joint angle = 0.5 * a * (t*dt)^2  => second derivative a
    dt = 0.1; a = 3.0; T = 10
    t = np.arange(T) * dt
    q = 0.5 * a * t**2
    base = np.tile([0, 0, 0, 1, 0, 0, 0], (T, 1)).astype(float)
    qpos = np.hstack([base, q[:, None]])
    m = compute_smoothness(qpos, dof=1, dt=dt)
    assert abs(m["joint_accel_rms"] - a) < 1e-6
```

- [ ] **Step 2: Run, expect ImportError/fail**

Run: `cd HoloNew && .../python -m pytest tests/test_metrics_smoothness.py -q`

- [ ] **Step 3: Implement**

```python
"""Smoothness metrics: acceleration / jerk RMS of base and joints (pure qpos)."""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation as R

def _rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0

def _base_angular_velocity(quat_wxyz, dt):
    # quat columns are [w, x, y, z]; scipy wants [x, y, z, w]
    q = quat_wxyz[:, [1, 2, 3, 0]]
    rot = R.from_quat(q)
    rel = rot[:-1].inv() * rot[1:]
    return rel.as_rotvec() / dt  # (T-1, 3)

def compute_smoothness(qpos, dof, dt):
    base_pos = qpos[:, 0:3]
    quat = qpos[:, 3:7]
    joints = qpos[:, 7:7 + dof]
    base_acc = np.diff(base_pos, n=2, axis=0) / dt**2
    omega = _base_angular_velocity(quat, dt)
    base_ang_acc = np.diff(omega, n=1, axis=0) / dt
    j_acc = np.diff(joints, n=2, axis=0) / dt**2
    j_jerk = np.diff(joints, n=3, axis=0) / dt**3
    return {
        "base_pos_accel_rms": _rms(base_acc),
        "base_ang_accel_rms": _rms(base_ang_acc),
        "joint_accel_rms": _rms(j_acc),
        "joint_jerk_rms": _rms(j_jerk),
        "joint_jerk_meanabs": float(np.mean(np.abs(np.diff(joints, n=3, axis=0)))),
    }
```

- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat(eval): smoothness metric module`

---

### Task 3: effort.py

**Files:**
- Create: `evaluation/metrics/effort.py`
- Test: `tests/test_metrics_effort.py`

- [ ] **Step 1: Failing tests**

```python
import numpy as np
from HoloNew.evaluation.metrics.effort import compute_effort

def test_at_limit_margin_zero_saturated():
    lo = np.array([-1.0, -1.0]); hi = np.array([1.0, 1.0])
    joints = np.array([[1.0, 0.0], [0.0, 0.0]])  # joint0 at upper limit in frame 0
    m = compute_effort(joints, lo, hi, dt=0.1)
    assert m["joint_limit_margin_min"] <= 0.0 + 1e-9
    assert m["joint_limit_saturation_frac"] > 0.0

def test_midrange_margin_half():
    lo = np.array([-1.0]); hi = np.array([1.0])
    joints = np.zeros((5, 1))  # always mid-range
    m = compute_effort(joints, lo, hi, dt=0.1)
    assert abs(m["joint_limit_margin_min"] - 0.5) < 1e-9
    assert m["joint_vel_rms"] < 1e-12
```

- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement**

```python
"""Effort metrics: joint-limit margin / saturation and joint velocity."""
from __future__ import annotations
import numpy as np

_SAT_EPS = 0.02

def compute_effort(joints, q_lower, q_upper, dt):
    rng = np.maximum(q_upper - q_lower, 1e-9)
    margin = np.minimum(joints - q_lower, q_upper - joints) / rng  # (T, dof)
    vel = np.diff(joints, n=1, axis=0) / dt
    return {
        "joint_limit_margin_min": float(np.min(margin)),
        "joint_limit_saturation_frac": float(np.mean(margin < _SAT_EPS)),
        "joint_vel_rms": float(np.sqrt(np.mean(np.square(vel)))) if vel.size else 0.0,
        "joint_vel_max": float(np.max(np.abs(vel))) if vel.size else 0.0,
    }
```

- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat(eval): effort metric module`

---

### Task 4: tracking.py

**Files:**
- Create: `evaluation/metrics/tracking.py`
- Test: `tests/test_metrics_tracking.py`

- [ ] **Step 1: Failing tests**

```python
import numpy as np
from HoloNew.evaluation.metrics.tracking import compute_tracking

def test_identical_zero_error():
    kp = np.random.RandomState(0).randn(8, 5, 3)
    m = compute_tracking(kp, kp.copy(), root_idx=0)
    assert m["mpjpe_global"] < 1e-12
    assert m["mpjpe_root_rel"] < 1e-12

def test_constant_offset():
    kp = np.random.RandomState(1).randn(8, 5, 3)
    d = np.array([0.0, 0.0, 0.3])
    m = compute_tracking(kp + d, kp, root_idx=0)
    assert abs(m["mpjpe_global"] - 0.3) < 1e-9
    assert m["mpjpe_root_rel"] < 1e-9  # offset cancels under root subtraction
```

- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement**

```python
"""Tracking fidelity: robot mapped keypoints vs SMPL reference joints."""
from __future__ import annotations
import numpy as np

def compute_tracking(robot_kpts, ref_kpts, root_idx, base_xyz=None, ref_root_xyz=None):
    """robot_kpts, ref_kpts: (T, J, 3), aligned by the joints_mapping order.
    base_xyz, ref_root_xyz: optional (T, 3) for base_track_err."""
    err = np.linalg.norm(robot_kpts - ref_kpts, axis=-1)  # (T, J)
    rb = robot_kpts - robot_kpts[:, root_idx:root_idx + 1, :]
    rf = ref_kpts - ref_kpts[:, root_idx:root_idx + 1, :]
    err_rr = np.linalg.norm(rb - rf, axis=-1)
    out = {
        "mpjpe_global": float(np.mean(err)),
        "mpjpe_root_rel": float(np.mean(err_rr)),
    }
    if base_xyz is not None and ref_root_xyz is not None:
        out["base_track_err"] = float(np.mean(np.linalg.norm(base_xyz - ref_root_xyz, axis=-1)))
    return out
```

- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat(eval): tracking metric module`

---

### Task 5: dynamics.py

**Files:**
- Create: `evaluation/metrics/dynamics.py`
- Test: `tests/test_metrics_dynamics.py`

- [ ] **Step 1: Failing tests**

```python
import numpy as np
from HoloNew.evaluation.metrics.dynamics import compute_dynamics

def test_matching_com_zero_accel_err():
    T = 12; dt = 0.1
    t = np.arange(T) * dt
    com = np.stack([t, np.zeros(T), -0.5 * 9.81 * t**2], axis=1)  # free fall
    m = compute_dynamics(com, com.copy(), dt=dt)
    assert m["com_accel_err"] < 1e-9

def test_offset_accel_consistent_and_L_rms():
    T = 12; dt = 0.1
    t = np.arange(T) * dt
    com = np.stack([t, 0 * t, 0 * t], axis=1)        # constant velocity -> zero accel
    ref = com.copy()
    L = np.zeros((T, 3))
    m = compute_dynamics(com, ref, dt=dt, L=L, L_ref=np.zeros((T, 3)))
    assert m["com_accel_err"] < 1e-9
    assert m["ang_momentum_rms"] < 1e-12
```

- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement**

```python
"""Dynamic-plausibility metrics: CoM acceleration error and angular-momentum RMS."""
from __future__ import annotations
import numpy as np

def _ddot(x, dt):
    return np.diff(x, n=2, axis=0) / dt**2

def compute_dynamics(com, ref_com, dt, *, L=None, L_ref=None):
    cdd = _ddot(com, dt)
    rdd = _ddot(ref_com, dt)
    out = {"com_accel_err": float(np.mean(np.linalg.norm(cdd - rdd, axis=-1)))}
    if L is not None:
        Lc = L if L_ref is None else (L - L_ref)
        out["ang_momentum_rms"] = float(np.sqrt(np.mean(np.sum(np.square(Lc), axis=-1))))
    return out
```

- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat(eval): dynamics metric module` (include `evaluation/metrics/__init__.py` from Task 1).

---

### Task 6: Wire into eval_retargeting.py

**Files:**
- Modify: `evaluation/eval_retargeting.py` (`Args`, `RetargetingEvaluator`, per-task result merge)

- [ ] **Step 1:** Read the existing `_evaluate_single_task` / per-task dict assembly and the
  `Args` dataclass to find the result-dict construction site and how qpos + reference + model are available.
- [ ] **Step 2:** Add to `Args`: `metrics: str = "smoothness,effort,tracking,dynamics"` and `dump_json: str | None = None`.
- [ ] **Step 3:** In the evaluator, add `_extended_metrics(self, qpos, ref) -> dict` that:
  extracts dof from the model, `dt = 1/fps`, joint ranges from `model.jnt_range`, robot mapped keypoints via `_get_robot_link_positions` over frames, SMPL ref joints from the loaded motion data, CoM via `subtree_com` (mj_forward per frame) and ref CoM proxy (SMPL pelvis); then calls the enabled `compute_*` and returns the merged dict. Gate each family on the `metrics` list.
- [ ] **Step 4:** Merge `_extended_metrics(...)` into the existing per-task scalar dict so `main()` aggregates the new keys.
- [ ] **Step 5:** Smoke-run on one task; confirm new keys print in the mean/std summary.

Run: `cd HoloNew && .../python evaluation/eval_retargeting.py --help` then a single-task run per existing usage.

- [ ] **Step 6: Commit** `feat(eval): integrate scoreboard metrics into the eval sweep`

---

### Task 7: De-duplicate existing A/B metric tests

**Files:**
- Modify: `tests/test_temporal_metric.py` (use `compute_smoothness` for jerk)
- Modify: `tests/test_centroidal_metric.py` (use `compute_dynamics` for com_accel_err)

- [ ] **Step 1:** In `test_temporal_metric.py`, replace the inline `jerk = mean(|diff(...,3)|)` with
  `compute_smoothness(res.qpos, dof, dt)["joint_jerk_meanabs"]`. Keep the off-vs-on assertion.
- [ ] **Step 2:** In `test_centroidal_metric.py`, replace `_com_accel_error` body with a call to
  `compute_dynamics(coms, ref_coms, dt)["com_accel_err"]` (extract `coms` via `rt.pin.com` as today,
  `ref_coms` from the existing reference pelvis proxy). Keep the off-vs-on assertion.
- [ ] **Step 3:** Run both tests, expect PASS (values unchanged).

Run: `cd HoloNew && .../python -m pytest tests/test_temporal_metric.py tests/test_centroidal_metric.py -q`

- [ ] **Step 4: Commit** `refactor(tests): share scoreboard metric fns in A/B metric tests`

---

## Self-Review

- **Spec coverage:** smoothness/effort/tracking/dynamics modules (Tasks 2-5) ✓; pure-function-on-arrays principle ✓; integration single-config + dump_json (Task 6) ✓; unit tests + refactor of existing A/B tests (Tasks 2-5, 7) ✓. ZMP/torque correctly absent (out of scope).
- **Placeholders:** none — all metric code is complete. Task 6 is the one integration task whose exact edit sites depend on reading `eval_retargeting.py` internals at execution time (its >800-line body is not reproduced here); the step list names the precise changes.
- **Type consistency:** `compute_smoothness(qpos, dof, dt)`, `compute_effort(joints, q_lower, q_upper, dt)`, `compute_tracking(robot_kpts, ref_kpts, root_idx, base_xyz=, ref_root_xyz=)`, `compute_dynamics(com, ref_com, dt, *, L=, L_ref=)` — consistent across tasks and `__init__`.
