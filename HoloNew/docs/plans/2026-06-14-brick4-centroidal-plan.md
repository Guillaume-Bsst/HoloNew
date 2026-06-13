# Brick 4 — Centroidal W^c / W^L — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Checkbox steps.

**Goal:** Track CoM acceleration (`W^c`) and centroidal angular momentum (`W^L`) so the pelvis is inherited from momentum rather than positionally scaffolded — behind a flag, validated, enabled by default **only if stable**.

**Architecture:** Gated `activate_centroidal` (default off ⇒ parity). When on, two quadratic terms are added in `solve_single_iteration`: (1) `W^c = λ_c‖c̈ − c̈_ref‖²` with `c̈ = (c − 2c_{t-1} + c_{t-2})/Δt²`, `c(ξ) = c_0 + J_c dqa` (`J_c` = CoM Jacobian, already in PinModel as `com_jacobian`); (2) `W^L = λ_L‖L‖²` driving the centroidal angular momentum to zero, `L = (A_G v)[3:6]`, `A_G` the centroidal momentum matrix (pinocchio), `v` the tangent velocity `difference(q_{t-1}, q_t)` (brick-2 tooling, linear in dqa). When centroidal is on, the Style pelvis scaffold is reduced/removed (`pelvis_anchor_weight → 0`), letting `W^c` hold the base — kept only if the pose stays stable.

**Divergences (documented):** `c̈_ref` is computed from the reference pelvis trajectory (`gmr_ground['pos']`, a dominant-mass CoM proxy), not a full reference-robot CoM. `W^L` targets `L=0` (regularize spurious spin); tracking the source's angular momentum is a later refinement.

**Tech Stack:** Python, pinocchio, numpy, cvxpy, pytest. Run from `modules/01_retargeting/HoloNew/HoloNew`, PY = `~/.holonew_deps/miniconda3/envs/holonew/bin/python`.

**Design:** `docs/specs/2026-06-13-brick4-centroidal-design.md`.

## Confirmed interfaces
- `PinModel.com(q_pin)` → CoM (3,); `PinModel.com_jacobian(q_pin)` → (3, nv) CoM Jacobian (FD-validated). `PinModel.difference_and_jac(q0, q1)` → (v (nv,), J (nv,nv)). `PinModel.integrate`, `qpos_mj_to_q_pin`. `rt.v_a_indices`, `rt.nv_a`, `rt._dt`.
- `solve_single_iteration` builds `obj_terms`; local full config `q`; decision var `dqa`. `retarget` keeps `q_prev` (t-1) and `q_prev2` (t-2) solved configs and `self.gmr_ground['pos']` (reference pelvis).
- pinocchio centroidal map: `pin.computeCentroidalMap(model, data, q) -> Ag (6, nv)` (momentum `h = Ag @ v`, `h[:3]` linear, `h[3:6]` angular). Verify the exact name (`computeCentroidalMap` / `ccrba`) in Task 1.

---

### Task 1: CoM-Jacobian recheck + centroidal momentum matrix A_G on PinModel

**Files:** Modify `src/test_socp/pin_model.py`; Test: `tests/test_pin_centroidal.py`

- [ ] **Step 1: Write the failing test** (A_G consistent with momentum; CoM Jacobian already FD-validated, re-assert briefly)
```python
# tests/test_pin_centroidal.py
import numpy as np, pinocchio as pin
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter

def _pm():
    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    return rt, rt.pin

def test_centroidal_map_matches_momentum():
    rt, pm = _pm()
    rng = np.random.default_rng(0)
    q = pm.qpos_mj_to_q_pin(rt.q_init_full[:36]); v = 0.1 * rng.standard_normal(pm.model.nv)
    Ag = pm.centroidal_map(q)                                # (6, nv)
    # pinocchio reference momentum about the CoM
    data2 = pm.model.createData()
    h_ref = pin.computeCentroidalMomentum(pm.model, data2, q, v)   # Force (linear, angular)
    h = Ag @ v
    np.testing.assert_allclose(h[:3], np.asarray(h_ref.linear), atol=1e-6)
    np.testing.assert_allclose(h[3:6], np.asarray(h_ref.angular), atol=1e-6)
```
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement on PinModel**
```python
def centroidal_map(self, q_pin):
    """(6, nv) centroidal momentum matrix A_G: h = A_G @ v (h[:3] linear, h[3:6] angular)."""
    return np.asarray(pin.computeCentroidalMap(self.model, self.data, pin.normalize(self.model, q_pin)))
```
(VERIFY in pinocchio 4.0.0: `pin.computeCentroidalMap(model, data, q)` returns A_G; `pin.computeCentroidalMomentum(model, data, q, v)` returns the momentum as a `Force` with `.linear`/`.angular`. If names differ, adapt; the test is the gate. Confirm `com_jacobian` exists and is FD-validated already — reuse it.)
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit**
```bash
git add src/test_socp/pin_model.py tests/test_pin_centroidal.py
git commit -m "feat(test_socp): pinocchio centroidal momentum matrix A_G (validated)"
```

---

### Task 2: Centroidal W^c + W^L term assembly

**Files:** Create `src/test_socp/centroidal.py`; Test: append to `tests/test_pin_centroidal.py`.

- [ ] **Step 1: Write the failing test** (objective at a random dqa equals an independent numpy ground truth)
```python
def test_centroidal_terms_match_numpy():
    import cvxpy as cp
    rt, pm = _pm()
    from HoloNew.src.test_socp.centroidal import build_centroidal_terms
    q0 = pm.qpos_mj_to_q_pin(rt.q_init_full[:36])
    rng = np.random.default_rng(2)
    q1 = pin.integrate(pm.model, q0, 0.02*rng.standard_normal(pm.model.nv))   # t-1
    q2 = pin.integrate(pm.model, q1, 0.02*rng.standard_normal(pm.model.nv))   # current q_t0
    c_tm1 = pm.com(q1); c_tm2 = pm.com(q0)
    cddot_ref = np.array([0.0, 0.0, -9.81])
    dqa = cp.Variable(rt.nv_a); val = 0.01*rng.standard_normal(rt.nv_a); dqa.value = val
    lam_c, lam_L, dt = 3.0, 1.0, 1.0/30.0
    terms = build_centroidal_terms(rt, q2, q1, c_tm1, c_tm2, cddot_ref, dqa, lam_c, lam_L, dt)
    # ground truth at val
    v_full = np.zeros(pm.model.nv); v_full[rt.v_a_indices] = val
    c0 = pm.com(q2); Jc = pm.com_jacobian(q2)[:, rt.v_a_indices]
    cddot = (c0 + Jc@val - 2*c_tm1 + c_tm2)/dt**2
    Ag = pm.centroidal_map(q2); vrel, Jd = pm.difference_and_jac(q1, q2)
    L = (Ag @ (vrel + Jd@v_full))[3:6]    # but only active cols of Jd matter; see note
    gt = lam_c*float(np.sum((cddot - cddot_ref)**2)) + lam_L*float(np.sum(L**2))
    np.testing.assert_allclose(float(sum(t.value for t in terms)), gt, rtol=1e-6)
```
(NOTE for the implementer: `v = difference(q1, q2)` is the velocity at dqa=0; the dqa contribution is `Jd[:, v_a_indices] @ dqa`. Build the W^L row matrix from `(Ag[3:6] @ Jd[:, v_a_indices])` and constant `(Ag[3:6] @ v)`. Make the ground truth and the term use the SAME slicing so they match.)
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement**
```python
# src/test_socp/centroidal.py
"""Centroidal W^c (CoM acceleration) + W^L (angular momentum) for TEST-SOCP.
See docs/specs/2026-06-13-brick4-centroidal-design.md."""
from __future__ import annotations
import cvxpy as cp
import numpy as np


def build_centroidal_terms(rt, q_t0, q_tm1, c_tm1, c_tm2, cddot_ref, dqa, lambda_c, lambda_L, dt):
    nv = rt.pin.model.nv
    # W^c: CoM acceleration
    c0 = rt.pin.com(q_t0)
    Jc = rt.pin.com_jacobian(q_t0)[:, rt.v_a_indices]          # (3, nv_a)
    # cddot = (c0 + Jc dqa - 2 c_tm1 + c_tm2)/dt^2 ; residual = cddot - cddot_ref
    A_c = (np.sqrt(lambda_c) / dt**2) * Jc
    b_c = (np.sqrt(lambda_c) / dt**2) * (c0 - 2*c_tm1 + c_tm2) - np.sqrt(lambda_c)*np.asarray(cddot_ref)
    # W^L: angular centroidal momentum -> 0
    Ag = rt.pin.centroidal_map(q_t0)                           # (6, nv)
    v0, Jd = rt.pin.difference_and_jac(q_tm1, q_t0)            # v at dqa=0, d v / d q_t0
    AgL = Ag[3:6, :]                                           # (3, nv)
    A_L = np.sqrt(lambda_L) * (AgL @ Jd[:, rt.v_a_indices])    # (3, nv_a)
    b_L = np.sqrt(lambda_L) * (AgL @ v0)                       # (3,)
    return [cp.sum_squares(A_c @ dqa + b_c), cp.sum_squares(A_L @ dqa + b_L)]
```
- [ ] **Step 4: Run** → PASS (rtol 1e-6).
- [ ] **Step 5: Commit**
```bash
git add src/test_socp/centroidal.py tests/test_pin_centroidal.py
git commit -m "feat(test_socp): centroidal W^c/W^L term assembly (numpy-validated)"
```

---

### Task 3: Config + thread CoM history & reference + wire

**Files:** Modify `src/test_socp/config.py`, `src/test_socp/test_socp.py`; Test: append.

- [ ] **Step 1: Config.** Add `activate_centroidal: bool = False`, `lambda_c: float = 0.0`, `lambda_L: float = 0.0`. `__init__` params + store; `from_config` wire.
- [ ] **Step 2: Reference + history.** In `retarget`, compute the reference CoM acceleration per frame from `self.gmr_ground['pos']` (the reference pelvis, CoM proxy): `cddot_ref[t] = (g[t] - 2 g[t-1] + g[t-2])/dt^2` (frames 0-1 → 0). Also keep the previous TWO solved CoMs `c_tm1`, `c_tm2`: after each solved frame compute `c = rt.pin.com(qpos_mj_to_q_pin(q[:36]))` and shift (`c_tm2 ← c_tm1`, `c_tm1 ← c`). Pass `c_tm1`, `c_tm2`, `cddot_ref[t]`, and the previous-frame config (q_t_last, already threaded) into the solve. Thread the needed values through `iterate` → `solve_single_iteration` (add params, default None).
- [ ] **Step 3: Wire.** In `solve_single_iteration`, before `cp.Problem`, append (guarded by `self.activate_centroidal and self.lambda_c>0 or self.lambda_L>0` AND `c_tm1/c_tm2/cddot_ref/q_t_last not None AND frame_idx>=2`):
```python
        if self.activate_centroidal and (self.lambda_c > 0 or self.lambda_L > 0) \
                and frame_idx >= 2 and c_tm1 is not None and q_t_last is not None:
            from HoloNew.src.test_socp.centroidal import build_centroidal_terms
            q_t0 = self.pin.qpos_mj_to_q_pin(q[:36])
            q_tm1 = self.pin.qpos_mj_to_q_pin(q_t_last[:36])
            obj_terms += build_centroidal_terms(self, q_t0, q_tm1, c_tm1, c_tm2,
                                                cddot_ref, dqa, self.lambda_c, self.lambda_L, self._dt)
```
- [ ] **Step 4: Test (append).** `activate_centroidal` default off; on with a few frames is finite. `PY -m pytest tests/test_pin_centroidal.py tests/test_test_socp_parity.py -q` → PASS (off ⇒ parity).
- [ ] **Step 5: Commit**
```bash
git add src/test_socp/config.py src/test_socp/test_socp.py tests/test_pin_centroidal.py
git commit -m "feat(test_socp): wire centroidal W^c/W^L into the solve (config, default off)"
```

---

### Task 4: Validate + remove scaffold + enable IF stable

**Files:** Test: `tests/test_centroidal_metric.py`; config defaults + parity re-baseline (only if enabling).

- [ ] **Step 1: Validation metric.** On `robot_only`, `retarget(max_frames=30)` with `activate_style=True` (default) AND `activate_centroidal=True`, tuning `lambda_c`, `lambda_L`, and lowering `pelvis_anchor_weight` toward 0. Assert:
  - finite, pelvis z in 0.4–1.0 m, no collapse;
  - CoM-acceleration tracking error (mean ‖c̈ − c̈_ref‖) is reduced vs `lambda_c=0`;
  - with `pelvis_anchor_weight=0` (scaffold removed), the pelvis stays sane (does not drift away). If removing the scaffold destabilizes, keep a small `pelvis_anchor_weight` and note it.
  Record the numbers.
- [ ] **Step 2: Enable IF clean.** If clean: set `activate_centroidal=True`, the tuned `lambda_c/lambda_L`, and `pelvis_anchor_weight` (0 if scaffold removed, else the small kept value) as defaults; re-baseline `tests/test_test_socp_parity.py` deliberately (record new BASELINE, sane root, atol=1e-6). If unstable: leave `activate_centroidal=False`, note why in the brick design doc, DONE_WITH_CONCERNS.
- [ ] **Step 3: Regression.** `PY -m pytest tests/test_pin_centroidal.py tests/test_centroidal_metric.py tests/test_test_socp_parity.py tests/test_style.py tests/test_pin_temporal.py tests/test_interaction_dxp.py -q` → PASS. Full clip finite, runtime reasonable.
- [ ] **Step 4: Commit**
```bash
git add -A
git commit -m "feat(test_socp): validate centroidal W^c/W^L; enable + remove pelvis scaffold IF stable + re-baseline"
```

---

## Self-review notes
- **Spec coverage:** A_G tooling (T1), W^c + W^L assembly (T2), config + CoM history + reference + wiring (T3), validate + scaffold removal + conditional enable (T4).
- **Naming:** `centroidal_map`, `centroidal.py::build_centroidal_terms`, `activate_centroidal`, `lambda_c`, `lambda_L`.
- **Safety:** default off ⇒ parity; enable ONLY if stable; keep a weak scaffold if removing it destabilizes the pelvis.
- **Documented divergences:** `c̈_ref` from the reference pelvis (CoM proxy); `W^L` targets 0 (source-L tracking deferred). Isolate this brick's metric from later-brick defaults (none yet, but keep the pattern). A final joint re-tuning of all weights is owed.
- **Open items:** (a) `pin.computeCentroidalMap` / `computeCentroidalMomentum` exact names + return types (Task 1 test gates). (b) whether the scaffold can be fully removed (Task 4 decides empirically). (c) frame-0/1 warm-up (centroidal needs t≥2).
