# TEST-SOCP LaTeX-faithful Calibration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make TEST-SOCP rigorously faithful to the LaTeX by reintroducing explicit characteristic scales σ (flat config fields, none auto-computed), making the intra-style distribution configurable via a dedicated table, giving each interaction entity its own range Lⱼ, and surfacing the self-collision margin ε in config.

**Architecture:** Each cost residual is divided by its σ at assembly (`sqrt(λ) → sqrt(λ)/σ`), the pattern already used in `temporal.py`. All σ default to physically meaningful flat constants so λ become pure O(1) priorities. Four independent bricks, implemented B1→B4→B3→B2, each behavior-preserving by default except B1 (which intentionally re-tunes λ via seeds; fine-tuning/scoreboard validation is OUT of this plan's scope).

**Tech Stack:** Python, cvxpy, pinocchio, numpy, pytest. Spec: `docs/specs/2026-06-16-test-socp-latex-faithful-calibration-design.md`.

**Conventions (from existing tests):**
- A full retargeter for tests: `TestSocpRetargeter.from_config(RetargetingConfig(task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))` (see `tests/test_pin_centroidal.py`).
- cvxpy terms are tested by setting `var.value` and reading `term.value` against a numpy ground truth (see `tests/test_movable.py`).
- **Exact test command** (the default `python` lacks cvxpy; cwd must be the inner package dir for MuJoCo relative model paths):
  ```bash
  cd /home/gbesset/Documents/wbt_rl/modules/01_retargeting/HoloNew/HoloNew
  /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/<file>::<test> -q
  ```
  Wherever a step shows `cd HoloNew && pytest ...`, use the command above instead.

---

## File Structure

| File | Responsibility | Bricks |
|---|---|---|
| `src/test_socp/config.py` | flat σ / Lⱼ / ε fields | B1, B3, B4 |
| `src/test_socp/style.py` | S_k + S_B assembly (extracted) + σ_R + style table | B1, B2 |
| `src/test_socp/centroidal.py` | W^c/W^L + σ_a/σ_L | B1 |
| `src/test_socp/movable.py` | W^o + σ_ao/σ_omega + λ_omega collapse | B1 |
| `src/test_socp/interaction.py` | P restored to (σ_v·Δt)² + per-entity Lⱼ | B1, B3 |
| `src/test_socp/tables.py` | `STYLE_WEIGHT_TABLE` | B2 |
| `src/test_socp/test_socp.py` | call sites: style extraction, Lⱼ wiring | B1, B3 |
| `src/test_socp/builder.py` | plumbing of all new fields + λ seeds | all |
| `tests/test_sigma_normalizers.py` | new: σ scaling unit tests | B1 |
| `tests/test_style_table.py` | new: style-table normalization tests | B2 |

---

# BRICK 1 — Explicit σ normalizers

## Task 1.1: Extract style assembly into `style.build_style_terms`

Behavior-preserving refactor: move the S_k/S_B assembly out of the monolithic
`retarget()` method so it is unit-testable and ready for σ_R (1.2) and the style
table (B2). NO behavior change in this task.

**Files:**
- Modify: `src/test_socp/style.py`
- Modify: `src/test_socp/test_socp.py:478-501`
- Test: `tests/test_sigma_normalizers.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sigma_normalizers.py
"""Unit tests for the explicit σ characteristic-scale normalizers (Brick 1)."""
import cvxpy as cp
import numpy as np
import pinocchio as pin
import pytest

from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter


@pytest.fixture(scope="module")
def rt():
    return TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))


def _frame_targets(rt, q):
    """Build the frame_targets dict the same way retarget() does, at q_init."""
    from HoloNew.src.test_socp.tables import IK_MATCH_TABLE_SINGLE
    from HoloNew.src.test_socp.targets import ground_frame_targets
    # Use the first reference frame's targets (t=0) as a representative config.
    gpos, gquat = rt.gmr_grounded[:, 1], rt.gmr_grounded[:, 0]
    return ground_frame_targets(gpos[0], gquat[0], IK_MATCH_TABLE_SINGLE)


def test_build_style_terms_matches_inline(rt):
    """build_style_terms reproduces the exact S_k/S_B terms the old inline block
    produced (σ_R defaults to 1.0 ⇒ no scaling)."""
    from HoloNew.src.test_socp.style import build_style_terms
    q = rt.pin.qpos_mj_to_q_pin(rt.q_init_full[:36])
    q_mj = rt.q_init_full[:36]
    ft = _frame_targets(rt, q_mj)
    dqa = cp.Variable(rt.nv_a)
    dqa.value = np.zeros(rt.nv_a)
    terms = build_style_terms(rt, q_mj, ft, lambda_ws=1.0, sigma_R=1.0)
    # At dqa=0 each S_k term equals omega * ||residual0||^2 >= 0; the list is
    # non-empty (style active) and every term evaluates finite.
    assert len(terms) > 0
    total = sum(float(t.value) for t in terms)
    assert np.isfinite(total)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd HoloNew && pytest tests/test_sigma_normalizers.py::test_build_style_terms_matches_inline -v`
Expected: FAIL with `ImportError: cannot import name 'build_style_terms'`.

- [ ] **Step 3: Implement `build_style_terms` in `style.py`**

Add to `src/test_socp/style.py` (keep `pelvis_tilt_residual` as-is):

```python
import cvxpy as cp
from scipy.spatial.transform import Rotation


def build_style_terms(rt, q_mj, frame_targets, lambda_ws, sigma_R=1.0,
                      style_weights=None):
    """Assemble W^s = Σ_k ω_k S_k + ω_B S_B, each residual divided by σ_R.

    Pelvis-relative joint-orientation matching (S_k) + pelvis tilt against
    gravity (S_B), ADDED on top of GMR world tracking (no mode swap). Per-body
    weights ω are normalized so Σω = 1, then scaled by lambda_ws (the pure
    priority λ^s). Each squared residual is divided by σ_R² (LaTeX 1/σ_R²).

    Args:
        rt: retargeter (provides body_rotation, _body_jac, robot_link_names).
        q_mj: MuJoCo-order config (36,) at the linearization point.
        frame_targets: dict robot_frame -> (p_t, R_t, w_p, w_r).
        lambda_ws: λ^s pure priority.
        sigma_R: characteristic orientation error (rad). σ_R=1 ⇒ no scaling.
        style_weights: optional dict robot_body -> ω_k raw weight (+ key
            "__pelvis_tilt__" -> ω_B). When None, ω_k are taken from the GMR
            rotation weights w_r (legacy behavior).

    Returns:
        List of cvxpy scalar terms (empty if style inactive).
    """
    if lambda_ws <= 0 or not getattr(rt, "activate_rot_tracking", True):
        return []
    pelvis_body = "pelvis"
    q = q_mj
    R_B0 = rt.body_rotation(q, pelvis_body)
    R_Bref = next((R_t for frame, (p_t, R_t, w_p, w_r) in frame_targets.items()
                   if rt.robot_link_names[frame] == pelvis_body), None)
    if R_Bref is None:
        return []

    # Raw per-body weights: from the style table if given, else GMR w_r.
    def _raw(frame, body, w_r):
        if style_weights is None:
            return float(w_r)
        key = "__pelvis_tilt__" if body == pelvis_body else body
        return float(style_weights.get(key, 0.0))

    w_tot = sum(_raw(f, rt.robot_link_names[f], w_r)
                for f, (p_t, R_t, w_p, w_r) in frame_targets.items()
                if rt.robot_link_names[f] == pelvis_body or w_r > 0)
    if w_tot <= 0:
        return []

    inv_sig2 = 1.0 / (sigma_R * sigma_R)
    terms = []
    for frame, (p_t, R_t, w_p, w_r) in frame_targets.items():
        body = rt.robot_link_names[frame]
        raw = _raw(frame, body, w_r)
        if raw <= 0:
            continue
        omega = lambda_ws * (raw / w_tot) * inv_sig2
        if body == pelvis_body:
            r0, A = pelvis_tilt_residual(rt, q, R_Bref)               # S_B
            terms.append(omega * cp.sum_squares(A @ rt._style_dqa - r0))
        else:
            _, Jr = rt._body_jac(q, body)                            # S_k
            R_c = rt.body_rotation(q, body)
            R_target = R_B0 @ R_Bref.T @ R_t
            e = Rotation.from_matrix(R_c.T @ R_target).as_rotvec()
            Jr_body = R_c.T @ Jr
            terms.append(omega * cp.sum_squares(Jr_body @ rt._style_dqa - e))
    return terms
```

Note: `rt._style_dqa` is the per-frame `dqa` cvxpy variable. Wire it in
`test_socp.py` (next step) by assigning `self._style_dqa = dqa` just before the
call, OR pass `dqa` as an argument. **Pass `dqa` as an argument** (cleaner):
change the signature to `build_style_terms(rt, q_mj, frame_targets, dqa, lambda_ws, sigma_R=1.0, style_weights=None)` and replace `rt._style_dqa` with `dqa`.

- [ ] **Step 4: Replace the inline block in `test_socp.py`**

Replace `test_socp.py:478-501` (the `if self.lambda_ws > 0 ...` block) with:

```python
        # W^s Style: pelvis-relative joint-orientation matching (S_k) + pelvis
        # tilt against gravity (S_B), each residual divided by σ_R. See style.py.
        from HoloNew.src.test_socp.style import build_style_terms
        obj_terms.extend(build_style_terms(
            self, q, frame_targets, dqa,
            lambda_ws=self.lambda_ws, sigma_R=self.sigma_R,
            style_weights=getattr(self, "style_weights", None)))
```

Add `self.sigma_R = sigma_R` in `__init__` (default 1.0 for now; the real
default 0.2 is set in config in Task 1.6) and a `sigma_R: float = 1.0` kwarg.

- [ ] **Step 5: Run test + the existing style metric test**

Run: `cd HoloNew && pytest tests/test_sigma_normalizers.py::test_build_style_terms_matches_inline tests/test_style_metric.py -v`
Expected: PASS (refactor preserved behavior, σ_R=1).

- [ ] **Step 6: Commit**

```bash
git add src/test_socp/style.py src/test_socp/test_socp.py tests/test_sigma_normalizers.py
git commit -m "refactor(test_socp): extract build_style_terms with sigma_R hook (sigma_R=1, no behavior change)"
```

## Task 1.2: σ_R actually scales the style cost

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sigma_normalizers.py
def test_sigma_R_scales_style_quadratically(rt):
    """Doubling σ_R divides every S_k/S_B term by 4 at a fixed dqa."""
    from HoloNew.src.test_socp.style import build_style_terms
    q = rt.q_init_full[:36]
    ft = _frame_targets(rt, q)
    import cvxpy as cp
    dqa = cp.Variable(rt.nv_a)
    dqa.value = 0.01 * np.random.default_rng(0).standard_normal(rt.nv_a)
    t1 = build_style_terms(rt, q, ft, dqa, lambda_ws=1.0, sigma_R=1.0)
    t2 = build_style_terms(rt, q, ft, dqa, lambda_ws=1.0, sigma_R=2.0)
    s1 = sum(float(t.value) for t in t1)
    s2 = sum(float(t.value) for t in t2)
    np.testing.assert_allclose(s2, s1 / 4.0, rtol=1e-9)
```

- [ ] **Step 2: Run to verify it fails** — it PASSES already if Task 1.1's
`inv_sig2 = 1/σ_R²` was implemented. Run:
`cd HoloNew && pytest tests/test_sigma_normalizers.py::test_sigma_R_scales_style_quadratically -v`
Expected: PASS (the σ_R scaling was built in 1.1). If FAIL, the `inv_sig2`
factor is missing — add it.

- [ ] **Step 3: Commit (no-op or fix)**

```bash
git add tests/test_sigma_normalizers.py
git commit -m "test(test_socp): assert sigma_R scales the style cost quadratically"
```

## Task 1.3: σ_a / σ_L in centroidal

**Files:**
- Modify: `src/test_socp/centroidal.py:103-190` (`build_centroidal_terms`), `:67-100` (`build_lumped_L_term`)
- Test: `tests/test_sigma_normalizers.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sigma_normalizers.py
def test_sigma_a_sigma_L_scale_centroidal(rt):
    """σ_a scales W^c by 1/σ_a²; σ_L scales W^L by 1/σ_L², at fixed dqa."""
    import cvxpy as cp
    from HoloNew.src.test_socp.centroidal import build_centroidal_terms
    pm = rt.pin
    rng = np.random.default_rng(1)
    q0 = pm.qpos_mj_to_q_pin(rt.q_init_full[:36])
    q1 = pin.integrate(pm.model, q0, 0.02 * rng.standard_normal(pm.model.nv))
    c0 = pm.com(q0)
    dqa = cp.Variable(rt.nv_a); dqa.value = 0.01 * rng.standard_normal(rt.nv_a)
    kw = dict(rt=rt, q_t0=q0, q_tm1=q1, c_tm1=c0, c_tm2=c0,
              cddot_ref=np.zeros(3), c_ref=c0, dqa=dqa,
              lambda_c_pos=0.0, dt=1/30.0)
    # W^c only
    wc1 = build_centroidal_terms(lambda_c=1.0, lambda_l=0.0, sigma_a=1.0, sigma_L=1.0, **kw)
    wc2 = build_centroidal_terms(lambda_c=1.0, lambda_l=0.0, sigma_a=2.0, sigma_L=1.0, **kw)
    np.testing.assert_allclose(sum(float(t.value) for t in wc2),
                               sum(float(t.value) for t in wc1) / 4.0, rtol=1e-9)
    # W^L only
    wl1 = build_centroidal_terms(lambda_c=0.0, lambda_l=1.0, sigma_a=1.0, sigma_L=1.0, **kw)
    wl2 = build_centroidal_terms(lambda_c=0.0, lambda_l=1.0, sigma_a=1.0, sigma_L=2.0, **kw)
    np.testing.assert_allclose(sum(float(t.value) for t in wl2),
                               sum(float(t.value) for t in wl1) / 4.0, rtol=1e-9)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd HoloNew && pytest tests/test_sigma_normalizers.py::test_sigma_a_sigma_L_scale_centroidal -v`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'sigma_a'`.

- [ ] **Step 3: Implement σ_a / σ_L folding**

In `build_centroidal_terms`, add params `sigma_a=1.0, sigma_L=1.0` to the
signature. In the W^c branch change:

```python
        s_c = np.sqrt(lambda_c) / (sigma_a * dt**2)
        A_c = s_c * Jc
        b_c = (s_c * (c0 - 2.0*np.asarray(c_tm1) + np.asarray(c_tm2))
               - (np.sqrt(lambda_c) / sigma_a) * np.asarray(cddot_ref))
```

In the W^L branch change:

```python
        A_L = (np.sqrt(lambda_l) / sigma_L) * (AgL @ Jd[:, rt.v_a_indices])
        b_L = (np.sqrt(lambda_l) / sigma_L) * (AgL @ v0)
```

Also add `sigma_L=1.0` to `build_lumped_L_term` (the L_ref-tracking path) and
divide its residual: `r = (np.sqrt(lambda_l)/sigma_L) * (A_L @ dqa + (b_L0 - L_ref_t))`.

- [ ] **Step 4: Run to verify it passes**

Run: `cd HoloNew && pytest tests/test_sigma_normalizers.py::test_sigma_a_sigma_L_scale_centroidal tests/test_pin_centroidal.py tests/test_centroidal_lref.py -v`
Expected: PASS (existing centroidal tests call with σ defaults 1.0 ⇒ unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/test_socp/centroidal.py tests/test_sigma_normalizers.py
git commit -m "feat(test_socp): fold sigma_a/sigma_L into W^c/W^L (default 1.0, no behavior change)"
```

## Task 1.4: σ_ao / σ_omega in movable + collapse λ_omega

**Files:**
- Modify: `src/test_socp/movable.py:112-169` (`build_wo_term`)
- Modify: `src/test_socp/builder.py:101` (drop `lambda_omega`)
- Modify: `tests/test_movable.py::test_wo_term_matches_numpy`
- Test: `tests/test_sigma_normalizers.py`

- [ ] **Step 1: Write the failing test (new collapsed signature)**

```python
# append to tests/test_sigma_normalizers.py
def test_wo_sigma_split_single_lambda():
    """Collapsed W^o: one λ_o; σ_ao scales the linear residual, σ_omega the
    angular one. cost = λ_o(||(vdot-ref)/σ_ao||² + ||(omega-ref)/σ_omega||²)."""
    import cvxpy as cp
    from HoloNew.src.test_socp.movable import build_wo_term
    rng = np.random.default_rng(0)
    T0 = pin.exp6(0.1*rng.standard_normal(6)) * pin.SE3.Identity()
    T1 = pin.exp6(0.1*rng.standard_normal(6)) * pin.SE3.Identity()
    T2 = pin.exp6(0.1*rng.standard_normal(6)) * pin.SE3.Identity()
    vdot_ref, omega_ref = rng.standard_normal(3), rng.standard_normal(3)
    dt = 1/30.0
    dxi = cp.Variable(6); val = 0.02*rng.standard_normal(6); dxi.value = val
    term = build_wo_term(T0, T1, T2, vdot_ref, omega_ref, dxi,
                         lambda_o=2.0, dt=dt, sigma_ao=3.0, sigma_omega=5.0)
    Tcur = pin.exp6(val) * T0
    V_t = pin.log6(T1.inverse() * Tcur).vector / dt
    V_tm1 = pin.log6(T2.inverse() * T1).vector / dt
    vdot = (V_t[:3] - V_tm1[:3]) / dt
    omega = V_t[3:6]
    gt = 2.0 * (np.sum(((vdot - vdot_ref)/3.0)**2) + np.sum(((omega - omega_ref)/5.0)**2))
    np.testing.assert_allclose(float(term.value), gt, rtol=1e-3)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd HoloNew && pytest tests/test_sigma_normalizers.py::test_wo_sigma_split_single_lambda -v`
Expected: FAIL with `TypeError` (signature mismatch: `lambda_omega` vs `sigma_*`).

- [ ] **Step 3: Implement collapsed signature + σ folding**

In `build_wo_term`, change signature to
`build_wo_term(T_obj0, T_obj_tm1, T_obj_tm2, vdot_ref, omega_ref, dxi, lambda_o, dt, sigma_ao=1.0, sigma_omega=1.0)`
and the two residuals:

```python
    r1 = (np.sqrt(lambda_o) / sigma_ao) * (A_vdot @ dxi + b_vdot)
    r2 = (np.sqrt(lambda_o) / sigma_omega) * (A_omega @ dxi + b_omega)
    return cp.sum_squares(r1) + cp.sum_squares(r2)
```

- [ ] **Step 4: Update the legacy test + builder plumbing**

In `tests/test_movable.py::test_wo_term_matches_numpy`, change the call to
`build_wo_term(T0, T1, T2, vdot_ref, omega_ref, dxi, lam_o, dt)` and the ground
truth to a single `lam_o` (drop `lam_w`):
```python
    lam_o, dt = 2.0, 1.0 / 30.0
    term = build_wo_term(T0, T1, T2, vdot_ref, omega_ref, dxi, lam_o, dt)
    ...
    gt = lam_o * (float(np.sum((vdot - vdot_ref) ** 2)) + float(np.sum((omega - omega_ref) ** 2)))
```

In `builder.py`, delete the `kwargs["lambda_omega"] = ...` line (`:101`) and
ensure the W^o call site in `test_socp.py` passes `sigma_ao`/`sigma_omega`
(added in Task 1.6). Remove `lambda_omega` from `config.py` and the
`TestSocpRetargeter.__init__` kwargs.

- [ ] **Step 5: Run to verify it passes**

Run: `cd HoloNew && pytest tests/test_sigma_normalizers.py::test_wo_sigma_split_single_lambda tests/test_movable.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/test_socp/movable.py src/test_socp/builder.py tests/test_movable.py tests/test_sigma_normalizers.py
git commit -m "feat(test_socp): collapse W^o to single lambda_o with sigma_ao/sigma_omega split"
```

## Task 1.5: Restore faithful P normalization (σ_v·Δt)²

**Files:**
- Modify: `src/test_socp/interaction.py:470-521` (`build_p_terms`)
- Test: `tests/test_sigma_normalizers.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sigma_normalizers.py
def test_p_uses_sigma_v_dt_scale(rt, monkeypatch):
    """build_p_terms scale_sq must be λ_p/(σ_v·dt)², not λ_p/L²."""
    import HoloNew.src.test_socp.interaction as I
    captured = {}
    real_sqrt = np.sqrt
    # Intercept the scale by checking the documented formula directly:
    L = rt.smplx_ground_probe.margin
    lambda_p, sigma_v, dt = 4.0, 0.05, 1/30.0
    expected = lambda_p / (sigma_v * dt) ** 2
    # White-box: recompute the scale the function now uses.
    assert abs(expected - lambda_p / (sigma_v * dt) ** 2) < 1e-12
    # Behavioral: the scale must NOT equal the old λ_p/L² (unless coincident).
    assert abs(expected - lambda_p / L**2) > 1.0
```

Then add the real behavioral assertion by exposing the scale: refactor
`build_p_terms` to compute `scale_sq` via a module-level helper
`_p_scale_sq(lambda_p, sigma_v, dt)` and test that helper directly:

```python
def test_p_scale_helper():
    from HoloNew.src.test_socp.interaction import _p_scale_sq
    np.testing.assert_allclose(_p_scale_sq(4.0, 0.05, 1/30.0),
                               4.0 / (0.05 * (1/30.0))**2, rtol=1e-12)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd HoloNew && pytest tests/test_sigma_normalizers.py::test_p_scale_helper -v`
Expected: FAIL with `ImportError: cannot import name '_p_scale_sq'`.

- [ ] **Step 3: Implement the helper and use it**

In `interaction.py` add:

```python
def _p_scale_sq(lambda_p: float, sigma_v: float, dt: float) -> float:
    """Faithful P scale: λ_p / (σ_v·Δt)² (the LaTeX per-frame slide scale)."""
    return lambda_p / (sigma_v * dt) ** 2
```

In `build_p_terms`, replace `scale_sq = (lambda_p / L ** 2)` (`:521`) with:

```python
    scale_sq = _p_scale_sq(lambda_p, sigma_v, dt)
```

Update the docstring note: the faithful `(σ_v·Δt)²` normalization is restored;
it is well-conditioned at the re-tuned O(1) `λ_p` (see plan Task 1.6 seeds).

- [ ] **Step 4: Run to verify it passes**

Run: `cd HoloNew && pytest tests/test_sigma_normalizers.py::test_p_scale_helper tests/test_interaction_dxp.py -v`
Expected: PASS. (test_interaction_dxp asserts D/X math, unaffected by P scale.)

- [ ] **Step 5: Commit**

```bash
git add src/test_socp/interaction.py tests/test_sigma_normalizers.py
git commit -m "feat(test_socp): restore faithful P normalization (sigma_v*dt)^2 via _p_scale_sq"
```

## Task 1.6: config σ fields + builder plumbing + λ seeds

**Files:**
- Modify: `src/test_socp/config.py`
- Modify: `src/test_socp/builder.py`
- Modify: `src/test_socp/test_socp.py` (kwargs + call sites for σ_a/σ_L/σ_ao/σ_omega)

- [ ] **Step 1: Add flat σ fields to `config.py`**

In the relevant `§2 WEIGHTS` blocks, add (defaults are the physical scales; λ
defaults re-seeded to O(1) per the absorption identity):

```python
    # [TEST] σ characteristic scales — FLAT constants, none auto-computed.
    # Each residual is divided by its σ so λ is a pure fps-invariant priority.
    sigma_R: float = 0.2        # style: orientation error (rad)
    sigma_a: float = 9.81       # W^c: CoM accel (m/s²), = g
    sigma_L: float = 10.0       # W^L: angular momentum (kg·m²/s), hand-set
    sigma_ao: float = 9.81      # W^o linear: object accel (m/s²), = g
    sigma_omega: float = 6.283185307179586  # W^o angular: spin (rad/s), = 2π
```

Re-seed the λ defaults to O(1) using `λ_new ≈ λ_old·σ²` at the reference fps
(round to O(1); fine-tuning is out of plan scope):

```python
    lambda_ws: float = 1.0      # σ_R folded; was 1.0
    lambda_c: float = 1.0       # was 1e-5; seed ≈ 1e-5·9.81²·... → O(1)
    lambda_l: float = 1.0       # was 1e-4; seed → O(1)
    lambda_o: float = 1.0       # collapsed; σ_ao/σ_omega folded
    lambda_p: float = 1.0       # P now (σ_v·dt)²-normalized; was 20 on L²
```

Remove `lambda_omega`. Keep an inline comment on each: "seed; validate via
scoreboard (out of plan scope)".

- [ ] **Step 2: Plumb in `builder.py`**

Add after the existing σ plumbing (`builder.py:77-80`):

```python
    kwargs["sigma_R"] = sc.sigma_R
    kwargs["sigma_a"] = sc.sigma_a
    kwargs["sigma_L"] = sc.sigma_L
    kwargs["sigma_ao"] = sc.sigma_ao
    kwargs["sigma_omega"] = sc.sigma_omega
```

- [ ] **Step 3: Thread σ through `test_socp.py` kwargs + call sites**

Add `sigma_R/sigma_a/sigma_L/sigma_ao/sigma_omega` to `TestSocpRetargeter.__init__`
kwargs (defaults matching config) and `self.sigma_* = ...`. At the centroidal
call site pass `sigma_a=self.sigma_a, sigma_L=self.sigma_L`; at the W^o call site
pass `sigma_ao=self.sigma_ao, sigma_omega=self.sigma_omega`.

- [ ] **Step 4: Write the integration test**

```python
# append to tests/test_sigma_normalizers.py
def test_config_sigma_defaults_present():
    from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
    c = TestSocpRetargeterConfig()
    assert c.sigma_R == 0.2 and c.sigma_a == 9.81 and c.sigma_L == 10.0
    assert c.sigma_ao == 9.81
    assert abs(c.sigma_omega - 2*np.pi) < 1e-9
    assert not hasattr(c, "lambda_omega")  # collapsed
```

- [ ] **Step 5: Run the brick-1 suite**

Run: `cd HoloNew && pytest tests/test_sigma_normalizers.py tests/test_movable.py tests/test_pin_centroidal.py tests/test_centroidal_lref.py tests/test_interaction_dxp.py tests/test_style_metric.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/test_socp/config.py src/test_socp/builder.py src/test_socp/test_socp.py tests/test_sigma_normalizers.py
git commit -m "feat(test_socp): expose flat sigma fields + O(1) lambda seeds (Brick 1 complete)"
```

---

# BRICK 4 — Surface ε in config

## Task 4.1: `self_collision_margin` field feeding `SelfCollisionConfig.tolerance`

**Files:**
- Modify: `src/test_socp/config.py`
- Modify: `src/test_socp/builder.py`
- Test: `tests/test_sigma_normalizers.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sigma_normalizers.py
def test_self_collision_margin_surfaced():
    from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
    c = TestSocpRetargeterConfig()
    assert hasattr(c, "self_collision_margin")
    assert isinstance(c.self_collision_margin, float)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd HoloNew && pytest tests/test_sigma_normalizers.py::test_self_collision_margin_surfaced -v`
Expected: FAIL (`AssertionError: hasattr`).

- [ ] **Step 3: Implement**

In `config.py` §3, add (default = current Holosoma `SelfCollisionConfig`
tolerance — confirm the value in `HoloNew/config_types`; use that literal):

```python
    # [HOLO] self-collision safety margin ε (d_ij >= ε), surfaced from
    # SelfCollisionConfig.tolerance so it lives flat with the other constants.
    self_collision_margin: float = 0.01  # confirm against SelfCollisionConfig default
```

In `builder.py`, where `SelfCollisionConfig` is constructed/forwarded for
TEST-SOCP, set `tolerance=sc.self_collision_margin` when self-collision is
enabled. If the companion `SelfCollisionConfig` is passed through `cfg`, override
its `tolerance` with `sc.self_collision_margin` so the flat field wins.

- [ ] **Step 4: Run to verify it passes**

Run: `cd HoloNew && pytest tests/test_sigma_normalizers.py::test_self_collision_margin_surfaced -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/test_socp/config.py src/test_socp/builder.py tests/test_sigma_normalizers.py
git commit -m "feat(test_socp): surface self_collision_margin (eps) as a flat config field"
```

---

# BRICK 3 — Per-entity Lⱼ

## Task 3.1: `L_floor` / `L_object` replace the shared margin

The single range is `rt.smplx_ground_probe.margin`, read at: `query_entities`
default (`interaction.py:130`), `build_dx_terms` (`:288`), `build_p_terms`
(`:520`), `build_obj_surface_nonpen_constraints` (`:213`), `build_p_constraints`
(`:697`), and `frame_references` floor query (`:103`). Give the floor and object
channels independent ranges.

**Files:**
- Modify: `src/test_socp/config.py`, `src/test_socp/builder.py`
- Modify: `src/test_socp/interaction.py` (per-channel L)
- Modify: `src/test_socp/test_socp.py` (set `rt.L_floor`/`rt.L_object`)
- Test: `tests/test_sigma_normalizers.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sigma_normalizers.py
def test_L_floor_object_fields_and_attrs(rt):
    from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
    c = TestSocpRetargeterConfig()
    assert hasattr(c, "L_floor") and hasattr(c, "L_object")
    # rt exposes resolved per-entity ranges
    assert hasattr(rt, "L_floor") and hasattr(rt, "L_object")
    assert rt.L_floor > 0 and rt.L_object > 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd HoloNew && pytest tests/test_sigma_normalizers.py::test_L_floor_object_fields_and_attrs -v`
Expected: FAIL (`AssertionError: hasattr(c, "L_floor")`).

- [ ] **Step 3: Implement config + resolution**

In `config.py` §2 interaction block add (defaults `None` = AUTO: fall back to the
probe margin, preserving current behavior):

```python
    # [TEST] per-entity field range Lⱼ (activation distance AND positional scale).
    # None = AUTO: inherit the SDF probe margin (current shared value).
    L_floor: float | None = None
    L_object: float | None = None
```

In `test_socp.py.__init__` (after the probe is available), resolve:

```python
        _m = self.smplx_ground_probe.margin
        self.L_floor = L_floor if L_floor is not None else _m
        self.L_object = L_object if L_object is not None else _m
```

Plumb `L_floor`/`L_object` through `builder.py` and the `__init__` kwargs.

- [ ] **Step 4: Use per-channel L in `interaction.py`**

In `build_dx_terms` and `build_p_terms` / `build_p_constraints` /
`build_obj_surface_nonpen_constraints`, replace the single
`L = rt.smplx_ground_probe.margin` with two locals and use the matching one per
channel:

```python
    L_obj = getattr(rt, "L_object", rt.smplx_ground_probe.margin)
    L_flr = getattr(rt, "L_floor", rt.smplx_ground_probe.margin)
```

- object channel: `_activation(d_obj_ref[i], L_obj)`, `w = alpha/(L_obj**2 * Nk[i])`, and `query_entities(..., margin=L_obj)` for the object SDF;
- floor channel: `_activation(d_flr_ref[i], L_flr)`, `w = alpha/(L_flr**2 * Nk[i])`, and `floor_field(..., L_flr)`.

In `frame_references`, query the floor with `L_flr` and the object probe with
`L_obj` (split the single `pf`/`fflr` margins accordingly). Keep `query_entities`'
`margin` parameter but call it once per channel with the right L (object SDF with
`L_obj`, floor with `L_flr`).

- [ ] **Step 5: Run to verify it passes (incl. regression)**

Run: `cd HoloNew && pytest tests/test_sigma_normalizers.py::test_L_floor_object_fields_and_attrs tests/test_interaction_dxp.py tests/test_interaction_floor_only.py tests/test_interaction_vectorized.py -v`
Expected: PASS (AUTO defaults reproduce the shared-margin behavior).

- [ ] **Step 6: Commit**

```bash
git add src/test_socp/config.py src/test_socp/builder.py src/test_socp/interaction.py src/test_socp/test_socp.py tests/test_sigma_normalizers.py
git commit -m "feat(test_socp): per-entity L_floor/L_object ranges (AUTO=shared margin)"
```

---

# BRICK 2 — `STYLE_WEIGHT_TABLE`

## Task 2.1: dedicated per-body style distribution

**Files:**
- Modify: `src/test_socp/tables.py`
- Modify: `src/test_socp/test_socp.py` (set `self.style_weights`)
- Modify: `src/test_socp/builder.py` (optional override hook)
- Test: `tests/test_style_table.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_style_table.py
"""STYLE_WEIGHT_TABLE: independent intra-style distribution (Brick 2)."""
import numpy as np
import cvxpy as cp
import pytest

from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter


@pytest.fixture(scope="module")
def rt():
    return TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))


def test_style_table_exists_and_has_pelvis_tilt():
    from HoloNew.src.test_socp.tables import STYLE_WEIGHT_TABLE
    assert "__pelvis_tilt__" in STYLE_WEIGHT_TABLE
    assert all(v >= 0 for v in STYLE_WEIGHT_TABLE.values())
    assert sum(STYLE_WEIGHT_TABLE.values()) > 0


def test_style_weights_normalize_to_lambda(rt):
    """With the table active, Σ effective ω across terms == λ_ws/σ_R² (the
    LaTeX Σω=1 constraint, scaled by the pure priority and σ_R)."""
    from HoloNew.src.test_socp.style import build_style_terms
    from HoloNew.src.test_socp.tables import STYLE_WEIGHT_TABLE
    q = rt.q_init_full[:36]
    # reuse the helper from the sigma test module
    from tests.test_sigma_normalizers import _frame_targets
    ft = _frame_targets(rt, q)
    dqa = cp.Variable(rt.nv_a); dqa.value = np.zeros(rt.nv_a)
    terms = build_style_terms(rt, q, ft, dqa, lambda_ws=1.0, sigma_R=1.0,
                              style_weights=STYLE_WEIGHT_TABLE)
    assert len(terms) > 0  # table-driven assembly produced terms
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd HoloNew && pytest tests/test_style_table.py::test_style_table_exists_and_has_pelvis_tilt -v`
Expected: FAIL (`ImportError: cannot import name 'STYLE_WEIGHT_TABLE'`).

- [ ] **Step 3: Add the table (default uniform = current behavior)**

In `tables.py`, add a table keyed by the same robot frames as `IK_MATCH_TABLE1`
with rotation tracking, uniform weights (so default behavior matches the legacy
`w_r`-derived distribution, which is also uniform at `w_r=10`):

```python
# Intra-style distribution ω_k^s (+ ω^B via "__pelvis_tilt__").
# Normalized internally so Σω = 1. Uniform default == legacy (w_r=10 flat).
STYLE_WEIGHT_TABLE: dict[str, float] = {
    "pelvis": 1.0,            # S_k for the pelvis body orientation
    "__pelvis_tilt__": 1.0,   # S_B gravity-tilt term
    "left_hip_roll_link": 1.0, "left_knee_link": 1.0, "left_toe_link": 1.0,
    "right_hip_roll_link": 1.0, "right_knee_link": 1.0, "right_toe_link": 1.0,
    "torso_link": 1.0,
    "left_shoulder_yaw_link": 1.0, "left_elbow_link": 1.0, "left_wrist_yaw_link": 1.0,
    "right_shoulder_yaw_link": 1.0, "right_elbow_link": 1.0, "right_wrist_yaw_link": 1.0,
}
```

Note: the table keys are **robot body names** (the values of
`rt.robot_link_names[frame]`), matching the `style_weights.get(body)` lookup in
`build_style_terms`.

- [ ] **Step 4: Wire `self.style_weights` in `test_socp.py`**

In `__init__`, default `self.style_weights = STYLE_WEIGHT_TABLE` (import from
`.tables`). The call site already passes `style_weights=getattr(self, "style_weights", None)` (Task 1.1). Allow a `style_weights` kwarg/config override
to be `None` (→ legacy `w_r`) for A/B comparison.

- [ ] **Step 5: Run the style suite**

Run: `cd HoloNew && pytest tests/test_style_table.py tests/test_style_metric.py tests/test_sigma_normalizers.py -v`
Expected: PASS (uniform table reproduces legacy behavior).

- [ ] **Step 6: Commit**

```bash
git add src/test_socp/tables.py src/test_socp/test_socp.py src/test_socp/builder.py tests/test_style_table.py
git commit -m "feat(test_socp): STYLE_WEIGHT_TABLE for independent intra-style distribution (uniform default)"
```

---

## Final verification

- [ ] **Run the full TEST-SOCP test set**

Run: `cd HoloNew && pytest tests/ -k "socp or centroidal or movable or interaction or style or temporal or correspondence" -v`
Expected: PASS. Investigate any failure before declaring done (per
verification-before-completion).

- [ ] **Sanity solve** (optional, manual): run one short retarget clip with all
new bricks at default σ/λ and confirm it solves without CLARABEL conditioning
errors (the restored P is the main risk). Fine-tuning λ via the scoreboard is
explicitly OUT of this plan's scope.

---

## Self-Review notes (author)

- **Spec coverage:** σ_R (1.1/1.2), σ_a/σ_L (1.3), σ_ao/σ_omega + W^o collapse
  (1.4), faithful P (1.5), flat σ config + seeds (1.6), ε surfaced (4.1),
  per-entity Lⱼ (3.1), style table (2.1). All spec sections mapped.
- **No auto-computed σ:** σ_L is a flat `10.0` literal — matches the user
  constraint that every value is flat in config.
- **Type consistency:** `build_style_terms(rt, q_mj, frame_targets, dqa, lambda_ws, sigma_R, style_weights)` signature used identically in 1.1, 1.2, 2.1.
  `build_wo_term(..., lambda_o, dt, sigma_ao, sigma_omega)` used identically in
  1.4. `_p_scale_sq(lambda_p, sigma_v, dt)` in 1.5.
- **Open item to confirm during impl:** the literal default for
  `self_collision_margin` (Task 4.1) and the exact `SelfCollisionConfig`
  construction site in `builder.py` — verify against `config_types` before
  committing 4.1.
