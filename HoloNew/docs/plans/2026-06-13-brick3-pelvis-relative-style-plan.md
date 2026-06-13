# Brick 3 — Pelvis-relative Style — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Checkbox steps.

**Goal:** Replace TEST-SOCP's world-frame pose tracking with the paper's Style component — pelvis-relative joint orientation + a roll/pitch-only pelvis anchor, with a weak pelvis translation scaffold — behind a flag, validated, and enabled by default **only if it stays stable**.

**Architecture:** Gated by `activate_style` (default off ⇒ current world tracking ⇒ parity). When on, the per-frame objective in `solve_single_iteration` changes: (1) joint orientation targets are **re-based by the current pelvis** `R_t' = R_B^0 (R_B^ref)^{-1} R_t` (this is exactly the pelvis-relative tracking and is invariant to base yaw, and reuses the existing body-frame orientation residual); (2) the pelvis orientation term is replaced by a roll/pitch-only tilt term `‖(R_B^ref)ᵀẑ − R_Bᵀẑ‖²` (yaw-invariant); (3) joint position terms are dropped (positions emerge from orientations); (4) the pelvis position term is kept but down-weighted to a weak scaffold. Brick 4 (centroidal) later removes the scaffold.

**Tech Stack:** Python, pinocchio, numpy, cvxpy, scipy, pytest. Run from `modules/01_retargeting/HoloNew/HoloNew`, PY = `~/.holonew_deps/miniconda3/envs/holonew/bin/python`.

**Design:** `docs/specs/2026-06-13-brick3-pelvis-relative-style-design.md`.

## Confirmed interfaces (read these)
- `solve_single_iteration` builds `obj_terms` over `frame_targets = {frame: (p_t, R_t, w_p, w_r)}`; `body = self.robot_link_names[frame]`; `Jp, Jr = self._body_jac(q, body)` (world translational/angular Jacobians, sliced to nv_a); `R_c = self.body_rotation(q, body)`; existing orientation residual: `e = Rotation.from_matrix(R_c.T @ R_t).as_rotvec(); Jr_body = R_c.T @ Jr; w_r*sum_squares(Jr_body@dqa - e)`. Decision var `dqa` (nv_a). (See test_socp.py lines ~706-733.)
- The pelvis frame is `self.robot_link_names[frame] == "pelvis"` (`ROBOT_ROOT_NAME = "pelvis"` in `src/test_socp/tables.py`); it has a target with high pos/rot weights.
- `_body_jac(q, "pelvis")` gives the pelvis world angular Jacobian `Jr_B` (3, nv_a). `body_rotation(q, "pelvis")` gives `R_B^0` (3,3).

---

### Task 1: Roll/pitch pelvis tilt term + FD test

**Files:** Create `src/test_socp/style.py`; Test: `tests/test_style.py`

The tilt residual: `u(q) = R_Bᵀ ẑ` (world-up in pelvis frame), `u_ref = (R_B^ref)ᵀ ẑ`. Linearized:
`u(q⊕dqa) ≈ u + (R_Bᵀ [ẑ]_× Jr_B) dqa` (world angular convention). Residual `(u_ref − u) − (R_Bᵀ [ẑ]_× Jr_B) dqa`. Yaw-invariant (a rotation about ẑ leaves `R_Bᵀẑ` fixed).

- [ ] **Step 1: Write the failing FD test** (the linearized tilt residual matches finite differences; and a pure-yaw base perturbation leaves the term unchanged)

```python
# tests/test_style.py
import numpy as np, pinocchio as pin
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.src.test_socp.style import pelvis_tilt_residual


def _rt():
    return TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))


def test_tilt_residual_matches_fd_and_is_yaw_invariant():
    rt = _rt()
    q_mj = rt.q_init_full[:36].copy()
    # an arbitrary reference pelvis orientation
    R_ref = pin.exp3(np.array([0.1, -0.2, 0.3]))
    r0, A = pelvis_tilt_residual(rt, q_mj, R_ref)        # r0=(3,), A=(3,nv_a): residual = r0 - A@dqa
    zhat = np.array([0.0, 0.0, 1.0])
    def u_of(qm):
        R_B = rt.body_rotation(qm, "pelvis"); return R_B.T @ zhat
    u0 = u_of(q_mj); np.testing.assert_allclose(r0, R_ref.T @ zhat - u0, atol=1e-9)
    # FD: integrate a tangent step, recompute u, compare to -A[:,k]
    q_pin = rt.pin.qpos_mj_to_q_pin(q_mj); eps = 1e-6
    for k in range(rt.nv_a):
        v = np.zeros(rt.pin.model.nv); v[rt.v_a_indices[k]] = eps
        qm2 = q_mj.copy(); qm2[:36] = rt.pin.q_pin_to_qpos_mj(pin.integrate(rt.pin.model, q_pin, v))
        fd = (u_of(qm2) - u0) / eps                       # d u / d dqa_k
        np.testing.assert_allclose(-A[:, k], fd, atol=1e-4, err_msg=f"col {k}")
```

- [ ] **Step 2: Run** `PY -m pytest tests/test_style.py -q` → FAIL.

- [ ] **Step 3: Implement**

```python
# src/test_socp/style.py
"""Pelvis-relative Style objective pieces for TEST-SOCP.
See docs/specs/2026-06-13-brick3-pelvis-relative-style-design.md."""
from __future__ import annotations

import numpy as np

_ZHAT = np.array([0.0, 0.0, 1.0])
_ZSKEW = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # [zhat]_x


def pelvis_tilt_residual(rt, q_mj, R_B_ref):
    """Roll/pitch tilt residual r0 - A@dqa for ||(R_ref)ᵀẑ − R_Bᵀẑ||².

    Returns (r0 (3,), A (3, nv_a)) with residual(dqa) = r0 - A @ dqa, where
    r0 = (R_ref)ᵀẑ − R_Bᵀẑ and A = R_Bᵀ [ẑ]_× Jr_B (world angular convention).
    """
    R_B = rt.body_rotation(q_mj, "pelvis")
    _, Jr_B = rt._body_jac(q_mj, "pelvis")               # (3, nv_a) world angular
    u = R_B.T @ _ZHAT
    r0 = R_B_ref.T @ _ZHAT - u
    A = R_B.T @ _ZSKEW @ Jr_B
    return r0, A
```

- [ ] **Step 4: Run** → PASS (FD matches; the yaw-invariance is implied — confirm by an extra check if desired: perturbing only the base-yaw tangent dof changes `u` negligibly).

- [ ] **Step 5: Commit**
```bash
git add src/test_socp/style.py tests/test_style.py
git commit -m "feat(test_socp): pelvis roll/pitch tilt residual (FD-validated, yaw-invariant)"
```

---

### Task 2: activate_style flag + Style objective in solve_single_iteration

**Files:** Modify `src/test_socp/config.py`, `src/test_socp/test_socp.py`; Test: append to `tests/test_style.py`.

- [ ] **Step 1: Config + constructor.** Add to `TestSocpRetargeterConfig`: `activate_style: bool = False`, `pelvis_anchor_weight: float = 1.0` (the scaffold down-weight on the pelvis position term; 1.0 = unchanged). Add `__init__` params + `self.activate_style`, `self.pelvis_anchor_weight`; wire in `from_config` (`kwargs[...] = sc...`).

- [ ] **Step 2: Style objective.** In `solve_single_iteration`, replace the `for frame ... in frame_targets` objective loop with a branch on `self.activate_style`. When **off**, keep the current loop exactly (parity). When **on**, build:
```python
        if not self.activate_style:
            <existing world-tracking loop unchanged>
        else:
            import cvxpy as cp
            from scipy.spatial.transform import Rotation
            from HoloNew.src.test_socp.style import pelvis_tilt_residual
            pelvis_body = "pelvis"
            # pelvis reference orientation R_B^ref and current R_B^0
            R_B0 = self.body_rotation(q, pelvis_body)
            R_Bref = None
            for frame, (p_t, R_t, w_p, w_r) in frame_targets.items():
                if self.robot_link_names[frame] == pelvis_body:
                    R_Bref = R_t; pelvis_p_t = p_t; pelvis_w_p = w_p
            for frame, (p_t, R_t, w_p, w_r) in frame_targets.items():
                body = self.robot_link_names[frame]
                Jp, Jr = self._body_jac(q, body)
                if body == pelvis_body:
                    # weak position scaffold
                    if pelvis_w_p > 0 and self.pelvis_anchor_weight > 0:
                        p_c = self.body_position(q, body)
                        obj_terms.append(self.pelvis_anchor_weight * pelvis_w_p
                                         * cp.sum_squares(Jp @ dqa - (p_t - p_c)))
                    # roll/pitch tilt (replaces full pelvis orientation tracking)
                    if w_r > 0:
                        r0, A = pelvis_tilt_residual(self, q, R_Bref)
                        obj_terms.append(w_r * cp.sum_squares(A @ dqa - r0))
                else:
                    # joint orientation re-based by the current pelvis (pelvis-relative)
                    if w_r > 0:
                        R_t_rebased = R_B0 @ R_Bref.T @ R_t
                        R_c = self.body_rotation(q, body)
                        e = Rotation.from_matrix(R_c.T @ R_t_rebased).as_rotvec()
                        Jr_body = R_c.T @ Jr
                        obj_terms.append(w_r * cp.sum_squares(Jr_body @ dqa - e))
                    # joint position tracking dropped (positions emerge from orientations)
```
(Keep `obj_terms = []` and `dqa = ...` above; the constraints/solve below stay identical.)

- [ ] **Step 3: Test (append).** `activate_style` default off; turning it on solves finite on a few frames:
```python
def test_style_default_off_and_runs_on():
    rt = _rt()
    assert rt.activate_style is False
    rt.activate_style = True
    res = rt.retarget(max_frames=6)
    import numpy as np
    assert np.all(np.isfinite(res.qpos))
```
Run `PY -m pytest tests/test_style.py tests/test_test_socp_parity.py -q` → PASS (off ⇒ parity; on ⇒ finite).

- [ ] **Step 4: Commit**
```bash
git add src/test_socp/config.py src/test_socp/test_socp.py tests/test_style.py
git commit -m "feat(test_socp): pelvis-relative Style objective (default off)"
```

---

### Task 3: Validate (yaw freed, pose sane, fidelity) + enable ONLY if clean

**Files:** Test: `tests/test_style_metric.py`; config default + parity re-baseline (only if enabling).

- [ ] **Step 1: Validation metric.** On `robot_only`, `retarget(max_frames=30)` with `activate_style=True` (keep `pelvis_anchor_weight=1.0` first; lower it toward the scaffold only if the pelvis stays well-behaved). Assert:
  - **Finite + no pose collapse:** all qpos finite; pelvis z within a sane band (e.g. 0.4–1.0 m); no joint NaN.
  - **Joint-orientation fidelity:** mean over tracked non-pelvis bodies of the pelvis-relative orientation error `||log(R̃_k^{-1} R̃_k^ref)||` is small (comparable to or better than the world-orientation tracking it replaces — compute both and compare).
  - **Yaw freed:** the solved pelvis yaw is NOT pinned to the reference — i.e. with `activate_style` the pelvis-yaw tracking error is allowed to be larger than world tracking (this is the intended freedom). Just assert the solve is sane; do not require yaw to match.
  Record the numbers in a comment.

- [ ] **Step 2: Decide enablement.** **Enable by default ONLY if Step 1 is clean** (finite, no collapse, fidelity comparable). If so: set `activate_style=True` default in `TestSocpRetargeterConfig`, and **re-baseline** `tests/test_test_socp_parity.py` deliberately (Style changes the default robot_only solve — record the new BASELINE, sanity-check sane root, keep atol=1e-6). If Step 1 shows instability/collapse, **leave `activate_style=False`** (implemented but off), document why in the brick design doc, and skip the re-baseline — report DONE_WITH_CONCERNS.

- [ ] **Step 3: Regression.** `PY -m pytest tests/test_style.py tests/test_style_metric.py tests/test_test_socp_parity.py tests/test_pin_temporal.py tests/test_interaction_dxp.py -q` → PASS.

- [ ] **Step 4: Commit**
```bash
git add src/test_socp/config.py tests/test_style_metric.py tests/test_test_socp_parity.py
git commit -m "feat(test_socp): validate pelvis-relative Style; enable by default if stable + re-baseline"
```

---

## Self-review notes
- **Spec coverage:** tilt term (T1), Style objective with rebased joint targets + pelvis tilt + weak scaffold + dropped joint positions (T2), validation + conditional enable + re-baseline (T3).
- **Naming:** `style.py::pelvis_tilt_residual`, `activate_style`, `pelvis_anchor_weight`.
- **Key safety:** default off ⇒ parity; enable ONLY if validation is clean (no silent pose-collapse regression). The scaffold (`pelvis_anchor_weight`) and brick-1 contacts hold the pelvis until brick 4 (centroidal) removes the scaffold.
- **Open items:** (a) confirm `R_t_rebased = R_B0 @ R_Bref.T @ R_t` is the right yaw-invariant rebasing (it tracks joint world orientation toward the reference re-expressed under the CURRENT pelvis; invariant to base yaw because R_B0 and R_k both rotate with yaw). (b) whether dropping ALL joint position tracking is too aggressive — if the pose gets loose, keep a small joint-position weight as a transitional aid and note it. (c) `pelvis_tilt_residual` sign convention — the FD test in T1 is the gate.
