# Brick 2 — Temporal regularization W^r — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add the paper's temporal regularization `W^r` (acceleration penalty on the configuration) to TEST-SOCP, default-off so the default solve is unchanged.

**Architecture:** In the pinocchio tangent space the paper's two `W^r` terms (joint acceleration `q̈` and base twist acceleration `V̇_B`) unify into ONE acceleration penalty on the full tangent vector: penalize the change in the per-frame tangent velocity `v_t = pin.difference(q_{t-1}, q_t)` versus the previous `v_{t-1} = pin.difference(q_{t-2}, q_{t-1})`. `v_t` is linearized in `dqa` via `pin.dDifference`. Per-DOF scales (`sigma_Vdot` on the 6 base DOFs, `sigma_qddot` on the joints) weight the rows. Built in a small helper, added to `obj_terms` behind a config weight `lambda_r` (default 0 ⇒ parity).

**Tech Stack:** Python, pinocchio, numpy, cvxpy, pytest. Run from `modules/01_retargeting/HoloNew/HoloNew` with `~/.holonew_deps/miniconda3/envs/holonew/bin/python` (PY).

**Design:** `docs/specs/2026-06-13-brick2-temporal-regularization-design.md`.

## Confirmed interfaces
- `rt.pin` is a `PinModel` (pinocchio free-flyer g1). `rt.pin.model`, `rt.pin.integrate(q_pin, v)`, `rt.pin.qpos_mj_to_q_pin(q[:36])`. `rt.v_a_indices` (active tangent cols), `rt.nv_a`. nv = 35 (base 6 + 29 joints; base is tangent indices 0:6).
- `solve_single_iteration(self, q_locked, q_a_n_last, ..., frame_idx, foot_sticking, q_t_last, obj_pose)` already receives `q_t_last` (previous-frame full MuJoCo config) and builds the local full config `q`. `retarget` threads `q_prev` (= previous solved frame) as `q_t_last`.
- pinocchio: `pin.difference(model, q0, q1) -> v (nv,)` (the tangent s.t. `integrate(q0, v) = q1`); `pin.dDifference(model, q0, q1, pin.ARG1) -> (nv, nv)` (Jacobian of `difference` wrt `q1`). Verify these exact names in pinocchio 4.0.0 in Task 1.

---

### Task 1: SE(3) tangent finite-difference helper on PinModel

**Files:** Modify `src/test_socp/pin_model.py`; Test: `tests/test_pin_temporal.py`

- [ ] **Step 1: Write the failing test** (the difference + its Jacobian are consistent with integrate, by finite differences)

```python
# tests/test_pin_temporal.py
import numpy as np, pinocchio as pin
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter


def _pm():
    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    return rt, rt.pin


def test_difference_and_jacobian_consistent():
    rt, pm = _pm()
    rng = np.random.default_rng(0)
    q0 = pm.qpos_mj_to_q_pin(rt.q_init_full[:36])
    q1 = pin.integrate(pm.model, q0, 0.05 * rng.standard_normal(pm.model.nv))
    v, J = pm.difference_and_jac(q0, q1)            # v=(nv,), J=d v / d(q1 tangent)
    np.testing.assert_allclose(pin.difference(pm.model, q0, q1), v, atol=1e-10)
    # FD: perturb q1 along tangent e_k, v should change by J[:,k]
    eps = 1e-6
    for k in range(pm.model.nv):
        d = np.zeros(pm.model.nv); d[k] = eps
        v2 = pin.difference(pm.model, q0, pin.integrate(pm.model, q1, d))
        np.testing.assert_allclose((v2 - v) / eps, J[:, k], atol=1e-4, err_msg=f"col {k}")
```

- [ ] **Step 2: Run** `PY -m pytest tests/test_pin_temporal.py -q` → FAIL (`difference_and_jac` missing).

- [ ] **Step 3: Implement on PinModel**

```python
def difference_and_jac(self, q0, q1):
    """Tangent velocity v = difference(q0, q1) and J = d v / d(q1 tangent)."""
    v = np.asarray(pin.difference(self.model, q0, q1))
    J = np.asarray(pin.dDifference(self.model, q0, q1, pin.ArgumentPosition.ARG1))
    return v, J
```
(VERIFY the enum: pinocchio 4.0.0 uses `pin.ARG1` or `pin.ArgumentPosition.ARG1` for the second-argument Jacobian — try both, keep the one that imports. If `dDifference` has a different signature, adapt; the FD test is the gate.)

- [ ] **Step 4: Run** → PASS.

- [ ] **Step 5: Commit**
```bash
git add src/test_socp/pin_model.py tests/test_pin_temporal.py
git commit -m "feat(test_socp): pinocchio tangent difference + Jacobian helper (FD-validated)"
```

---

### Task 2: Temporal W^r term assembly

**Files:** Create `src/test_socp/temporal.py`; Test: append to `tests/test_pin_temporal.py`

- [ ] **Step 1: Write the failing test** (the term assembles and the objective at a random dqa equals an independent numpy ground truth)

```python
def test_wr_term_matches_numpy():
    import cvxpy as cp
    rt, pm = _pm()
    from HoloNew.src.test_socp.temporal import build_temporal_term
    q_tm2 = pm.qpos_mj_to_q_pin(rt.q_init_full[:36])
    rng = np.random.default_rng(1)
    q_tm1 = pin.integrate(pm.model, q_tm2, 0.03 * rng.standard_normal(pm.model.nv))
    q_t0  = pin.integrate(pm.model, q_tm1, 0.03 * rng.standard_normal(pm.model.nv))
    dqa = cp.Variable(rt.nv_a); val = 0.01 * rng.standard_normal(rt.nv_a); dqa.value = val
    lam, s_q, s_V, dt = 2.0, 0.5, 0.5, 1.0 / 30.0
    term = build_temporal_term(rt, q_t0, q_tm1, q_tm2, dqa, lam, s_q, s_V, dt)
    # ground truth: v_t(dqa) = difference(q_tm1, integrate(q_t0, v_full)); accel = (v_t - v_tm1)/dt^2
    v_full = np.zeros(pm.model.nv); v_full[rt.v_a_indices] = val
    v_t = pin.difference(pm.model, q_tm1, pin.integrate(pm.model, q_t0, v_full))
    v_tm1 = pin.difference(pm.model, q_tm2, q_tm1)
    w = np.ones(pm.model.nv) / s_q**2; w[:6] = 1.0 / s_V**2     # base rows use sigma_V
    gt = lam * float(np.sum(w * ((v_t - v_tm1) / dt**2) ** 2))
    np.testing.assert_allclose(float(term.value), gt, rtol=1e-6)
```

- [ ] **Step 2: Run** → FAIL (`build_temporal_term` missing).

- [ ] **Step 3: Implement**

```python
# src/test_socp/temporal.py
"""Temporal regularization (W^r) for TEST-SOCP — acceleration penalty in the
pinocchio tangent space. See docs/specs/2026-06-13-brick2-temporal-regularization-design.md."""
from __future__ import annotations

import cvxpy as cp
import numpy as np


def build_temporal_term(rt, q_t0, q_tm1, q_tm2, dqa, lambda_r, sigma_qddot, sigma_Vdot, dt):
    """One cvxpy expression: lambda_r * || diag(sqrt(w)) * (v_t - v_tm1)/dt^2 ||^2,
    v_t = difference(q_tm1, integrate(q_t0, v_full)) linearized in dqa via dDifference,
    v_tm1 = difference(q_tm2, q_tm1) (constant). Base tangent rows (0:6) weighted by
    1/sigma_Vdot^2, joint rows by 1/sigma_qddot^2."""
    nv = rt.pin.model.nv
    v0, J = rt.pin.difference_and_jac(q_tm1, q_t0)     # v_t at dqa=0, and d/d q_t
    v_tm1 = rt.pin.difference_and_jac(q_tm2, q_tm1)[0]
    Jc = J[:, rt.v_a_indices]                          # (nv, nv_a)
    # per-DOF sqrt weights / dt^2
    w = np.full(nv, 1.0 / sigma_qddot); w[:6] = 1.0 / sigma_Vdot
    s = np.sqrt(lambda_r) * w / dt**2                  # (nv,)
    A = s[:, None] * Jc                                # (nv, nv_a)
    b = s * (v0 - v_tm1)                               # (nv,)
    return cp.sum_squares(A @ dqa + b)
```
(`v_t ≈ v0 + Jc @ dqa`; residual `(v_t - v_tm1)/dt^2`; the term is `lambda_r * sum_w ((residual))^2` folded into `A dqa + b`.)

- [ ] **Step 4: Run** → PASS.

- [ ] **Step 5: Commit**
```bash
git add src/test_socp/temporal.py tests/test_pin_temporal.py
git commit -m "feat(test_socp): temporal W^r acceleration term (tangent space, numpy-validated)"
```

---

### Task 3: Config weights + thread q_{t-2} + wire into the solve

**Files:** Modify `src/test_socp/config.py`, `src/test_socp/test_socp.py`; Test: append.

- [ ] **Step 1: Config.** In `TestSocpRetargeterConfig` add `lambda_r: float = 0.0`, `sigma_qddot: float = 1.0`, `sigma_Vdot: float = 1.0`. In `__init__` add params (defaults same) + store `self.lambda_r/sigma_qddot/sigma_Vdot`; in `from_config` set `kwargs[...] = sc...` for the three (NO object-task gating — temporal reg applies to all tasks, but default 0 ⇒ inert).

- [ ] **Step 2: Thread q_{t-2}.** `solve_single_iteration` already has `q_t_last` (= q_{t-1}). Add `q_t_last2=None` (q_{t-2}) param; thread through `iterate`. In `retarget`, keep a `q_prev2` (previous-previous solved config), pass it; for the first two frames `q_prev2`/`q_prev` default to the init config (so the acceleration is ~0 at warm-up).

- [ ] **Step 3: Wire.** In `solve_single_iteration`, before `cp.Problem`, append (guarded by `lambda_r>0` and `q_t_last is not None and q_t_last2 is not None`):
```python
        if self.lambda_r > 0 and q_t_last is not None and q_t_last2 is not None:
            from HoloNew.src.test_socp.temporal import build_temporal_term
            q_pin0 = self.pin.qpos_mj_to_q_pin(q[:36])
            q_pin1 = self.pin.qpos_mj_to_q_pin(q_t_last[:36])
            q_pin2 = self.pin.qpos_mj_to_q_pin(q_t_last2[:36])
            obj_terms += [build_temporal_term(self, q_pin0, q_pin1, q_pin2, dqa,
                                              self.lambda_r, self.sigma_qddot,
                                              self.sigma_Vdot, self._dt)]
```

- [ ] **Step 4: Test** — `tests/test_pin_temporal.py`: with `lambda_r=0` the parity test `tests/test_test_socp_parity.py` still passes (inert); with `lambda_r` set on the instance, `retarget(max_frames=6)` is finite. Run `PY -m pytest tests/test_pin_temporal.py tests/test_test_socp_parity.py -q` → PASS.

- [ ] **Step 5: Commit**
```bash
git add src/test_socp/config.py src/test_socp/test_socp.py tests/test_pin_temporal.py
git commit -m "feat(test_socp): wire temporal W^r into the solve (config weights, default off)"
```

---

### Task 4: Metric (jerk reduction) + enable + regression

**Files:** Test: `tests/test_temporal_metric.py`; config default.

- [ ] **Step 1: Metric test.** Solve `robot_only` `max_frames=20` with `lambda_r=0` and with a tuned `lambda_r>0`; assert the mean joint jerk (third difference of `qpos[:,7:]`) drops with W^r on, and the pelvis-tracking error does not degrade beyond a small tolerance. Tune `lambda_r`, `sigma_qddot`, `sigma_Vdot` so jerk genuinely decreases without hurting tracking; record the numbers.

- [ ] **Step 2: Enable.** Set the tuned `lambda_r` (and sigmas) defaults in `TestSocpRetargeterConfig`. Re-baseline `tests/test_test_socp_parity.py` deliberately (W^r changes the default solve — record the new BASELINE, sanity-check sane root poses, keep atol=1e-6). Note: W^r applies to robot_only too (unlike interaction), so the robot_only baseline DOES change — re-baseline it.

- [ ] **Step 3: Regression.** `PY -m pytest tests/test_pin_temporal.py tests/test_temporal_metric.py tests/test_test_socp_parity.py tests/test_interaction_dxp.py tests/test_holosoma_constraints.py -q` → PASS. Confirm a full clip stays finite and the runtime is essentially unchanged (W^r adds one cheap term).

- [ ] **Step 4: Commit**
```bash
git add src/test_socp/config.py tests/test_temporal_metric.py tests/test_test_socp_parity.py
git commit -m "feat(test_socp): enable temporal W^r by default after jerk-reduction validation + re-baseline"
```

---

## Self-review notes
- **Spec coverage:** SE(3) tangent tooling (T1), W^r term unified in tangent space (T2), config + q_{t-2} threading + wiring (T3), metric + enable + re-baseline (T4). The paper's `q̈` and `V̇_B` are unified by the tangent-space `difference`; per-DOF sigmas keep them separately tunable.
- **Naming:** `difference_and_jac`, `build_temporal_term`, `lambda_r`/`sigma_qddot`/`sigma_Vdot`, `q_t_last2`.
- **Open items to confirm:** (a) the exact `pin.dDifference` arg-enum in 4.0.0 (Task 1 FD test gates it). (b) warm-up at frames 0–1 (q_{t-2} unavailable → use init config / skip). (c) W^r changes robot_only ⇒ re-baseline the parity snapshot (unlike brick 1).
