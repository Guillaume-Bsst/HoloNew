# SMPL-derived centroidal targets (CoM + angular momentum) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the TEST-SOCP centroidal proxy targets with grounded-SMPL quantities — the real CoM (Z-anchored, un-scaled arc) for W^c/W^c_pos, and `I_robot(q)·ω_human` for W^L — so free-flight phases are ballistically coherent.

**Architecture:** A new pure module computes, from the posed grounded SMPL-X mesh, the per-frame human CoM `c_h` and effective angular velocity `ω_h = I_h⁻¹ L_h` (mass/scale-free). `builder.py` precomputes these once and stores `rt._c_ref_all` (anchored CoM target) and `rt._omega_ref_all`. The W^L tracking term is rewritten to track the robot's full centroidal angular momentum toward `I_robot(q)·ω_h`, using a new `pin_model` accessor for the centroidal composite inertia. The solver and viewer diagnostics read the new fields.

**Tech Stack:** Python, NumPy, pinocchio (`ccrba`, `computeCentroidalMap`), cvxpy, smplx (via existing `HumanBody`), pytest.

**Spec:** `docs/specs/2026-06-15-smpl-centroidal-targets-design.md`

---

## File Structure

- **Create** `src/test_socp/smpl_centroidal.py` — pure functions: rest-pose triangle masses, posed triangle centroids, and the `(c_h, ω_h)` sequence from posed centroids. No solver/pinocchio dependency; fully unit-testable.
- **Modify** `src/test_socp/pin_model.py` — add `centroidal_inertia(q_pin)` (3×3 `Ig.angular` via `pin.ccrba`).
- **Modify** `src/test_socp/centroidal.py` — rewrite `build_lumped_L_term` → `build_L_track_term` (centroidal L vs `I_robot·ω`); remove `mapped_frame_masses_and_names` + `reference_orbital_angular_momentum`.
- **Modify** `src/test_socp/builder.py` — build `HumanBody`; precompute `rt._c_ref_all` (SMPL CoM, anchored) and `rt._omega_ref_all`; drop `_L_ref_all`/`_lumped_*`.
- **Modify** `src/test_socp/test_socp.py` — read `self._c_ref_all`; derive `cddot_ref` from it; call `build_L_track_term`; fill `angular_momentum_ref = I_robot·ω` in `_fill_diagnostics`.
- **Modify** `src/viewer.py` — L target/solved arrows share one scale factor (now the same quantity).
- **Modify** `tests/test_centroidal_lref.py` — drop tests of removed helpers; rewrite the term test and the cartwheel integration test for the new formulation.
- **Create** `tests/test_smpl_centroidal.py` — unit tests for the new module.
- **Create** `tests/test_pin_centroidal_inertia.py` — unit test for the new accessor.

---

## Task 1: `pin_model.centroidal_inertia`

**Files:**
- Modify: `src/test_socp/pin_model.py` (after `centroidal_map`, ~line 323)
- Test: `tests/test_pin_centroidal_inertia.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pin_centroidal_inertia.py
"""Centroidal composite inertia accessor (Ig angular block) used by W^L tracking."""
import numpy as np


def _pin():
    from HoloNew.examples.robot_retarget import RetargetingConfig
    from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    return rt


def test_centroidal_inertia_properties():
    rt = _pin()
    q = rt.pin.qpos_mj_to_q_pin(rt.q_init_full[:36])
    Ig = rt.pin.centroidal_inertia(q)
    assert Ig.shape == (3, 3)
    np.testing.assert_allclose(Ig, Ig.T, atol=1e-9)            # symmetric
    assert np.all(np.linalg.eigvalsh(Ig) > 0)                  # positive definite
    # deterministic
    np.testing.assert_allclose(Ig, rt.pin.centroidal_inertia(q), atol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pin_centroidal_inertia.py -v`
Expected: FAIL with `AttributeError: 'PinModel' object has no attribute 'centroidal_inertia'` (or similar).

- [ ] **Step 3: Write minimal implementation**

Add to `src/test_socp/pin_model.py` immediately after `centroidal_map`:

```python
    def centroidal_inertia(self, q_pin: np.ndarray) -> np.ndarray:
        """(3, 3) centroidal composite rigid-body inertia about the CoM (Ig angular block).

        Computed via pin.ccrba (which also fills data.Ig). Used to map a desired angular
        velocity to centroidal angular momentum: L = Ig @ omega.
        """
        q = pin.normalize(self.model, q_pin)
        pin.ccrba(self.model, self.data, q, np.zeros(self.model.nv))
        return np.array(self.data.Ig.inertia)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pin_centroidal_inertia.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/test_socp/pin_model.py tests/test_pin_centroidal_inertia.py
git commit -m "feat(pin_model): centroidal composite inertia accessor (Ig)"
```

---

## Task 2: SMPL CoM + ω module

**Files:**
- Create: `src/test_socp/smpl_centroidal.py`
- Test: `tests/test_smpl_centroidal.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_smpl_centroidal.py
"""Pure SMPL centroidal helpers: rest-area masses, posed centroids, (c, omega) sequence.

omega = I^-1 L is mass- and scale-free, so a soft target built from a uniform-density
shell is robust. Tests use synthetic point sets (the (c, omega) function only needs
per-frame point clouds + masses, not real triangles)."""
import numpy as np


def test_triangle_rest_masses_are_areas():
    from HoloNew.src.test_socp.smpl_centroidal import triangle_rest_masses
    # One right triangle, legs 3 and 4 -> area 6.
    verts = np.array([[0, 0, 0], [3, 0, 0], [0, 4, 0]], float)
    faces = np.array([[0, 1, 2]])
    m = triangle_rest_masses(verts, faces)
    np.testing.assert_allclose(m, [6.0])


def test_triangle_centroids():
    from HoloNew.src.test_socp.smpl_centroidal import triangle_centroids
    verts = np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0]], float)
    faces = np.array([[0, 1, 2]])
    c = triangle_centroids(verts, faces)
    np.testing.assert_allclose(c, [[1.0, 1.0, 0.0]])


def test_com_omega_pure_rotation():
    # Two equal masses at +/-x rotating ccw at w about z. CoM=origin, omega_z=w.
    from HoloNew.src.test_socp.smpl_centroidal import com_omega_sequence
    T, dt, w, r = 40, 0.02, 3.0, 1.0
    m = np.array([2.0, 2.0])
    th = w * dt * np.arange(T)
    p = np.zeros((T, 2, 3))
    p[:, 0, 0], p[:, 0, 1] = r * np.cos(th), r * np.sin(th)
    p[:, 1, 0], p[:, 1, 1] = -r * np.cos(th), -r * np.sin(th)
    c, omega = com_omega_sequence(p, m, dt)
    np.testing.assert_allclose(c, 0.0, atol=1e-9)
    np.testing.assert_allclose(omega[3:, 2].mean(), w, rtol=0.05)
    assert np.allclose(omega[2:, :2], 0.0, atol=1e-6)


def test_com_omega_free_fall_no_spin():
    # All points share a downward-accelerating translation: omega == 0, c ballistic.
    from HoloNew.src.test_socp.smpl_centroidal import com_omega_sequence
    T, dt, g = 30, 0.02, 9.81
    base = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
    m = np.ones(3)
    t = dt * np.arange(T)
    z = 1.0 + 2.0 * t - 0.5 * g * t**2
    p = np.broadcast_to(base, (T, 3, 3)).copy()
    p[:, :, 2] += z[:, None]
    c, omega = com_omega_sequence(p, m, dt)
    np.testing.assert_allclose(omega, 0.0, atol=1e-6)
    cddot = (c[2:] - 2 * c[1:-1] + c[:-2]) / dt**2
    np.testing.assert_allclose(cddot[:, 2], -g, rtol=1e-3)


def test_com_omega_scale_and_mass_invariant():
    from HoloNew.src.test_socp.smpl_centroidal import com_omega_sequence
    rng = np.random.default_rng(0)
    T, dt = 20, 0.02
    p = rng.standard_normal((T, 12, 3))
    m = rng.uniform(0.5, 2.0, 12)
    _, omega = com_omega_sequence(p, m, dt)
    _, omega_s = com_omega_sequence(p * 5.0, m * 3.0, dt)   # scale length & mass
    np.testing.assert_allclose(omega, omega_s, atol=1e-8)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_smpl_centroidal.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'HoloNew.src.test_socp.smpl_centroidal'`.

- [ ] **Step 3: Write the module**

```python
# src/test_socp/smpl_centroidal.py
"""Grounded-SMPL centroidal quantities for the ballistic targets.

Per-triangle point masses (mass = rest-pose triangle area, a uniform-density shell)
tracked across the posed mesh give the human CoM c_h and the effective angular
velocity omega_h = I_h^-1 L_h. omega is mass- and scale-free, so the uniform-density
shell is an adequate proxy. See docs/specs/2026-06-15-smpl-centroidal-targets-design.md.
"""
from __future__ import annotations

import numpy as np


def triangle_rest_masses(verts_rest: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """(F,) per-triangle rest-pose area = uniform-surface-density mass."""
    v = np.asarray(verts_rest)[faces]                 # (F, 3, 3)
    e1 = v[:, 1] - v[:, 0]
    e2 = v[:, 2] - v[:, 0]
    return 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)


def triangle_centroids(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """(F, 3) centroid of each triangle at a pose."""
    return np.asarray(verts)[faces].mean(axis=1)


def com_omega_sequence(centroids_seq, masses: np.ndarray, dt: float):
    """Human CoM and effective angular velocity per frame.

    Args:
        centroids_seq: iterable of (F, 3) posed point (triangle-centroid) clouds, one
            per frame (an ndarray (T, F, 3) also works).
        masses: (F,) point masses (rest-pose triangle areas).
        dt: timestep.

    Returns:
        c (T, 3): CoM = sum(m_e p_e) / M.
        omega (T, 3): I_h^-1 L_h via causal finite-difference velocities. Frame 0
            (no previous frame) gets omega = 0.
    """
    masses = np.asarray(masses, dtype=np.float64)
    M = masses.sum()
    cs, omegas = [], []
    p_prev = None
    c_prev = None
    for p in centroids_seq:
        p = np.asarray(p, dtype=np.float64)
        c = (masses[:, None] * p).sum(0) / M
        if p_prev is None:
            omega = np.zeros(3)
        else:
            v = (p - p_prev) / dt
            cd = (c - c_prev) / dt
            r = p - c                                          # (F, 3)
            L = (masses[:, None] * np.cross(r, v - cd)).sum(0)  # (3,)
            r2 = (r * r).sum(1)                                 # (F,)
            I = np.einsum(
                "e,eij->ij", masses,
                r2[:, None, None] * np.eye(3)[None] - r[:, :, None] * r[:, None, :])
            omega = np.linalg.solve(I, L)
        cs.append(c)
        omegas.append(omega)
        p_prev = p
        c_prev = c
    return np.asarray(cs), np.asarray(omegas)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_smpl_centroidal.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/test_socp/smpl_centroidal.py tests/test_smpl_centroidal.py
git commit -m "feat(smpl_centroidal): grounded-SMPL CoM + effective angular velocity"
```

---

## Task 3: Builder precompute (HumanBody + `_c_ref_all` + `_omega_ref_all`)

**Files:**
- Modify: `src/test_socp/builder.py` (the centroidal precompute block, currently ~lines 233-244)
- Test: `tests/test_centroidal_smpl_targets.py` (new)

Context: at this point in `builder.py`, the following are set: `rt.gmr_grounded` (T,52,3), `rt.gmr_ground` (scaled targets), `rt.q_init_full` (base placed at frame-0 ground pelvis), `rt._dt`, `rt.human_quat` (T,52,4 wxyz), `rt._smplx_orientations` (T,22,3 rotvec or None), `rt._smplx_betas`, `rt._smplx_gender`, and `sc` (the scale config with `scale_xy_robot`). `rt.pin` is the pinocchio backend; `rt.q_a_indices` etc. exist.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_centroidal_smpl_targets.py
"""Builder precomputes the SMPL CoM target (Z-anchored) and omega reference."""
import numpy as np


def _rt():
    from HoloNew.examples.robot_retarget import RetargetingConfig
    from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
    return TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))


def test_com_target_shapes_and_anchor():
    rt = _rt()
    T = rt.gmr_grounded.shape[0]
    assert rt._c_ref_all.shape == (T, 3)
    assert rt._omega_ref_all is None or rt._omega_ref_all.shape == (T, 3)
    # Z anchored to the robot's own init CoM at frame 0.
    q0 = rt.pin.qpos_mj_to_q_pin(rt.q_init_full[:36])
    c_robot0 = rt.pin.com(q0)
    np.testing.assert_allclose(rt._c_ref_all[0, 2], c_robot0[2], atol=0.05)


def test_com_target_z_arc_unscaled():
    rt = _rt()
    # The Z arc (deviation from frame 0) is the raw human CoM Z deviation, un-scaled.
    # Reconstruct the raw human CoM Z to compare the *delta*, not the absolute height.
    from HoloNew.src.test_socp.correspondence.human_body import HumanBody
    from HoloNew.src.test_socp.correspondence.constants import SMPLX_MODEL_DIR_DEFAULT
    from HoloNew.src.test_socp.smpl_centroidal import (
        triangle_rest_masses, triangle_centroids, com_omega_sequence)
    hb = HumanBody(SMPLX_MODEL_DIR_DEFAULT, rt._smplx_betas, rt._smplx_gender)
    faces = hb.faces.astype(np.int64)
    masses = triangle_rest_masses(hb.rest_verts(), faces)
    pelvis = rt.gmr_grounded[:, 0]
    n = 10
    cents = (triangle_centroids(hb.placed_verts(rt.human_quat[t], pelvis[t], frame_idx=t), faces)
             for t in range(n))
    c_h, _ = com_omega_sequence(cents, masses, rt._dt)
    dz_target = rt._c_ref_all[:n, 2] - rt._c_ref_all[0, 2]
    dz_human = c_h[:n, 2] - c_h[0, 2]
    np.testing.assert_allclose(dz_target, dz_human, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_centroidal_smpl_targets.py -v`
Expected: FAIL — `AttributeError: ... no attribute '_c_ref_all'` (builder does not set it yet).

- [ ] **Step 3: Replace the centroidal precompute block in `builder.py`**

Find the existing block (the W^L `_L_ref_all` precompute):

```python
    # Precompute the lumped reference angular momentum L_ref(t) for W^L tracking
    # (opt-in). Built from the GMR target mapped-body trajectory + robot link
    # masses; consumed by build_lumped_L_term in the solve.
    rt._lumped_frames = None
    rt._lumped_masses = None
    rt._L_ref_all = None
    if rt.activate_wl_track:
        from .centroidal import (
            mapped_frame_masses_and_names, reference_orbital_angular_momentum)
        rt._lumped_frames, rt._lumped_masses = mapped_frame_masses_and_names(rt)
        rt._L_ref_all = reference_orbital_angular_momentum(
            rt.gmr_ground["pos"], rt._lumped_masses, rt._dt)
```

Replace it with:

```python
    # Precompute the grounded-SMPL centroidal targets for W^c / W^L:
    #  - rt._c_ref_all (T,3): CoM target. XY scaled by scale_xy_robot (placement
    #    convention of preprocess.scale); Z un-scaled but anchored to the robot's own
    #    init CoM so stance has no height bias and the flight arc stays ballistic.
    #  - rt._omega_ref_all (T,3): human effective angular velocity (I_h^-1 L_h),
    #    mass/scale-free; consumed by build_L_track_term.
    # Falls back to the pelvis proxy (and omega=None) if the SMPL-X body is unavailable.
    from .correspondence.human_body import HumanBody
    from .correspondence.constants import SMPLX_MODEL_DIR_DEFAULT
    from .smpl_centroidal import (
        triangle_rest_masses, triangle_centroids, com_omega_sequence)

    _q_init_pin = rt.pin.qpos_mj_to_q_pin(rt.q_init_full[:36])
    _c_robot_init = rt.pin.com(_q_init_pin)
    _pelvis_grounded = rt.gmr_grounded[:, 0]            # (T,3) raw grounded pelvis
    rt._omega_ref_all = None
    try:
        _hb = HumanBody(SMPLX_MODEL_DIR_DEFAULT, rt._smplx_betas, rt._smplx_gender)
        _faces = _hb.faces.astype(np.int64)
        _masses = triangle_rest_masses(_hb.rest_verts(), _faces)
        _orient = rt._smplx_orientations

        def _centroids():
            for _t in range(T):
                if _orient is not None:
                    _v = _hb.placed_verts_smpl(_orient[_t], _pelvis_grounded[_t], frame_idx=_t)
                else:
                    _v = _hb.placed_verts(rt.human_quat[_t], _pelvis_grounded[_t], frame_idx=_t)
                yield triangle_centroids(_v, _faces)

        _c_h, _omega_h = com_omega_sequence(_centroids(), _masses, rt._dt)
        _c_ref = np.empty_like(_c_h)
        _c_ref[:, 0] = sc.scale_xy_robot * _c_h[:, 0]
        _c_ref[:, 1] = sc.scale_xy_robot * _c_h[:, 1]
        _c_ref[:, 2] = _c_robot_init[2] + (_c_h[:, 2] - _c_h[0, 2])
        rt._c_ref_all = _c_ref
        rt._omega_ref_all = _omega_h
    except Exception as _exc:  # noqa: BLE001
        print(f"[TestSocp] SMPL centroidal targets unavailable ({_exc}); "
              f"using pelvis-proxy CoM, W^L tracking disabled.")
        _g_pelvis = rt.gmr_ground["pos"][:, 0, :]
        _pelvis_init = rt.pin.body_position(_q_init_pin, "pelvis")
        rt._c_ref_all = _g_pelvis + (_c_robot_init - _pelvis_init)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_centroidal_smpl_targets.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/test_socp/builder.py tests/test_centroidal_smpl_targets.py
git commit -m "feat(builder): precompute grounded-SMPL CoM target + omega reference"
```

---

## Task 4: Rewrite the W^L tracking term

**Files:**
- Modify: `src/test_socp/centroidal.py` (remove `mapped_frame_masses_and_names` + `reference_orbital_angular_momentum`; replace `build_lumped_L_term` with `build_L_track_term`)
- Modify: `tests/test_centroidal_lref.py` (drop tests of removed helpers; rewrite term test)

- [ ] **Step 1: Rewrite the term test in `tests/test_centroidal_lref.py`**

Replace the entire file contents with:

```python
"""W^L ballistic tracking: track the robot's full centroidal angular momentum toward
I_robot(q)*omega_ref. build_L_track_term builds the residual linearized in dqa; this is
checked against an independent numpy ground truth, then end-to-end on an aerial clip.
"""
import numpy as np
import cvxpy as cp
import pytest


def test_L_track_term_matches_numpy():
    from HoloNew.examples.robot_retarget import RetargetingConfig
    from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
    from HoloNew.src.test_socp.centroidal import build_L_track_term

    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))

    rng = np.random.default_rng(0)
    q_pin = rt.pin.qpos_mj_to_q_pin(rt.q_init_full[:36])
    q_prev = rt.pin.integrate(q_pin, rng.standard_normal(rt.pin.model.nv) * 0.01)
    omega_ref = rng.standard_normal(3)
    dqa_val = rng.standard_normal(rt.nv_a) * 0.01
    dqa = cp.Variable(rt.nv_a); dqa.value = dqa_val
    lam, dt = 2.0, 1.0 / 30.0

    term = build_L_track_term(rt, q_pin, q_prev, dqa, omega_ref, lam, dt)

    # Independent numpy ground truth (A_G, Jd, Ig fixed at q_pin).
    A_G = rt.pin.centroidal_map(q_pin)
    v0, Jd = rt.pin.difference_and_jac(q_prev, q_pin)
    A_ang = A_G[3:6, :]
    A_L = (A_ang @ Jd[:, rt.v_a_indices]) / dt
    b_L0 = (A_ang @ v0) / dt
    Ig = rt.pin.centroidal_inertia(q_pin)
    target = Ig @ omega_ref
    L = A_L @ dqa_val + b_L0
    gt = lam * float(np.sum((L - target) ** 2))
    np.testing.assert_allclose(float(term.value), gt, rtol=1e-9)


def test_L_track_term_zero_omega_is_spin_regularizer():
    # omega_ref = 0 -> the residual reduces to ||L_robot||^2 (drive-to-zero behaviour).
    from HoloNew.examples.robot_retarget import RetargetingConfig
    from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
    from HoloNew.src.test_socp.centroidal import build_L_track_term

    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    rng = np.random.default_rng(1)
    q_pin = rt.pin.qpos_mj_to_q_pin(rt.q_init_full[:36])
    q_prev = rt.pin.integrate(q_pin, rng.standard_normal(rt.pin.model.nv) * 0.01)
    dqa = cp.Variable(rt.nv_a); dqa.value = np.zeros(rt.nv_a)
    term = build_L_track_term(rt, q_pin, q_prev, dqa, np.zeros(3), 1.0, 1.0 / 30.0)

    A_G = rt.pin.centroidal_map(q_pin)
    v0, _ = rt.pin.difference_and_jac(q_prev, q_pin)
    L0 = (A_G[3:6, :] @ v0) / (1.0 / 30.0)
    np.testing.assert_allclose(float(term.value), float(np.sum(L0 ** 2)), rtol=1e-9)


def test_L_track_reproduces_cartwheel_spin():
    """On a real aerial clip, W^L tracking makes the solved centroidal angular momentum
    follow the I_robot*omega target. Correlation > 0.8."""
    from pathlib import Path
    from HoloNew.examples.robot_retarget import RetargetingConfig
    from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
    from HoloNew.src.test_socp.test_socp import TestSocpRetargeter

    clip = Path(__file__).resolve().parent.parent / "demo_data" / "SFU" / "0007_Cartwheel001.npz"
    if not clip.exists():
        pytest.skip("cartwheel clip not present")
    N, dt = 50, 1.0 / 30.0
    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="0007_Cartwheel001",
        data_format="smplx", data_path=clip.parent,
        retargeter=TestSocpRetargeterConfig(activate_wl_track=True, lambda_l_track=5.0)))
    assert rt._omega_ref_all is not None
    res = rt.retarget(max_frames=N)
    assert np.all(np.isfinite(res.qpos))

    # Solved centroidal L vs the target I_robot(q_t) * omega_ref(t).
    Lsol = np.zeros((N, 3))
    Ltar = np.zeros((N, 3))
    for t in range(1, N):
        q = rt.pin.qpos_mj_to_q_pin(res.qpos[t, :36])
        qm = rt.pin.qpos_mj_to_q_pin(res.qpos[t - 1, :36])
        v = (rt.pin.difference_and_jac(qm, q)[0]) / dt
        Lsol[t] = (rt.pin.centroidal_map(q) @ v)[3:6]
        Ltar[t] = rt.pin.centroidal_inertia(q) @ rt._omega_ref_all[t]
    corr = (np.sum(Lsol[3:] * Ltar[3:]) /
            (np.linalg.norm(Lsol[3:]) * np.linalg.norm(Ltar[3:]) + 1e-9))
    assert corr > 0.8, f"L tracking failed: solved-vs-target corr={corr:.2f}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_centroidal_lref.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_L_track_term'`.

- [ ] **Step 3: Edit `src/test_socp/centroidal.py`**

Delete `mapped_frame_masses_and_names` (the `def` through its `return`) and `reference_orbital_angular_momentum` (its `def` through `return L`). Replace the `build_lumped_L_term` function (its `def` through `return cp.sum_squares(r)`) with:

```python
def build_L_track_term(rt, q_pin, q_pin_prev, dqa, omega_ref_t, lambda_l, dt):
    """W^L ballistic tracking: lambda_l * ||L_robot(dqa) - Ig(q)·omega_ref||^2.

    Robot full centroidal angular momentum, linearized in dqa (A_G, Ig fixed at q_pin,
    mirroring how A_G(q0)@v fixes q0):
        L(dqa) = (A_G(q0) @ v)[3:6],  v = difference(q_prev,q0)/dt + (Jd/dt) dqa
    Target = Ig(q0) @ omega_ref_t: give the robot the human's rotation rate realized with
    its own composite inertia (ballistically coherent regardless of mass). Returns a
    scalar cvxpy expression.
    """
    import cvxpy as cp

    A_G = rt.pin.centroidal_map(q_pin)                       # (6, nv)
    v0, Jd = rt.pin.difference_and_jac(q_pin_prev, q_pin)    # v0 (nv,), Jd (nv, nv)
    A_ang = A_G[3:6, :]                                      # (3, nv)
    A_L = (A_ang @ Jd[:, rt.v_a_indices]) / dt              # (3, nv_a)
    b_L0 = (A_ang @ v0) / dt                                 # (3,)
    target = rt.pin.centroidal_inertia(q_pin) @ np.asarray(omega_ref_t, dtype=np.float64)
    r = np.sqrt(lambda_l) * (A_L @ dqa + (b_L0 - target))
    return cp.sum_squares(r)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_centroidal_lref.py::test_L_track_term_matches_numpy tests/test_centroidal_lref.py::test_L_track_term_zero_omega_is_spin_regularizer -v`
Expected: PASS (the cartwheel test needs Task 5 wiring + the clip; it may xfail/skip until then).

- [ ] **Step 5: Commit**

```bash
git add src/test_socp/centroidal.py tests/test_centroidal_lref.py
git commit -m "feat(centroidal): rewrite W^L as centroidal-L tracking of I_robot*omega"
```

---

## Task 5: Solver wiring (`test_socp.py`)

**Files:**
- Modify: `src/test_socp/test_socp.py` — the CoM reference block in `retarget()` (~lines 895-919), the `_cddot_ref_all` computation (~lines 901-905), and the W^L tracking block (~lines 718-730).

- [ ] **Step 1: Replace the CoM-reference precompute in `retarget()`**

Find (the pelvis-proxy CoM block, ~lines 895-919, including the `self._c_ref_all = _c_ref_all` line added earlier this session):

```python
        # Brick 4 — Centroidal: precompute reference CoM acceleration from the
        # reference pelvis trajectory (a dominant-mass CoM proxy).  Uses central
        # finite differences; frames 0-1 are set to zero (warm-up, guard inactive).
        # The reference pelvis positions live in gpos[:, pelvis_body_idx, :].
        # Ground targets index: 'pos' is (T, N_bodies, 3); index 0 is the pelvis.
        _g_pelvis = gpos[:, 0, :]  # (T, 3) reference pelvis positions
        _cddot_ref_all = np.zeros((T, 3), dtype=np.float64)
        for _t in range(2, T):
            _cddot_ref_all[_t] = (
                _g_pelvis[_t] - 2.0 * _g_pelvis[_t - 1] + _g_pelvis[_t - 2]
            ) / (self._dt ** 2)
```

Replace with:

```python
        # Brick 4 — Centroidal: the CoM reference is the grounded-SMPL CoM target
        # precomputed in the builder (rt._c_ref_all: XY scaled, Z anchored to the robot
        # init CoM, un-scaled arc). cddot_ref is its central second difference; frames
        # 0-1 stay zero (warm-up, guard inactive).
        _c_ref_all = self._c_ref_all
        _cddot_ref_all = np.zeros((T, 3), dtype=np.float64)
        for _t in range(2, T):
            _cddot_ref_all[_t] = (
                _c_ref_all[_t] - 2.0 * _c_ref_all[_t - 1] + _c_ref_all[_t - 2]
            ) / (self._dt ** 2)
```

- [ ] **Step 2: Remove the now-redundant pelvis-proxy CoM computation**

Immediately below, find (the block that recomputed `_c_ref_all` from the pelvis proxy and the duplicate store):

```python
        # Precompute reference CoM positions for W^c_pos.
        # The reference is the pelvis trajectory (CoM proxy), but the robot CoM sits
        # below and behind the pelvis (a constant structural offset).  To avoid
        # pulling the robot CoM to the pelvis position (which would push the pelvis
        # upward by ~7 cm), we compute the offset at init and apply it to all frames:
        #   c_ref[t] = g_pelvis[t] + (c_init - pelvis_init)
        # This anchors the CoM to its reference trajectory *in the robot's own frame
        # relative to the pelvis*, so W^c_pos corrects drift without biasing height.
        _q_init_pin = self.pin.qpos_mj_to_q_pin(self.q_init_full[:36])
        _com_init = self.pin.com(_q_init_pin)
        _pelvis_init = self.pin.body_position(_q_init_pin, "pelvis")
        _com_pelvis_offset = _com_init - _pelvis_init  # (3,) structural offset
        _c_ref_all = _g_pelvis + _com_pelvis_offset     # (T, 3) reference CoM positions
        # Keep the grounded CoM target around for the viewer diagnostics (the W^c_pos
        # anchor target), so the solved-vs-target CoM gap can be shown geometrically.
        self._c_ref_all = _c_ref_all
```

Replace with (keep `_com_init` for the warm-up stencil below; `_c_ref_all`/`self._c_ref_all` already set):

```python
        # _com_init seeds the W^c finite-difference warm-up stencil below.
        _q_init_pin = self.pin.qpos_mj_to_q_pin(self.q_init_full[:36])
        _com_init = self.pin.com(_q_init_pin)
```

- [ ] **Step 3: Rewrite the W^L tracking block**

Find (~lines 718-730):

```python
        # W^L reference tracking (opt-in): track the lumped reference angular
        # momentum instead of driving L to 0. Needs the previous solved config for
        # the velocity finite difference, so fires from frame_idx >= 1.
        if (self.activate_wl_track
                and getattr(self, "_L_ref_all", None) is not None
                and frame_idx >= 1 and q_t_last is not None):
            from HoloNew.src.test_socp.centroidal import build_lumped_L_term
            q_pin_cur = _q_pin
            q_pin_prev = self.pin.qpos_mj_to_q_pin(q_t_last[:36])
            obj_terms.append(build_lumped_L_term(
                self, q_pin_cur, q_pin_prev, dqa, self._lumped_frames,
                self._lumped_masses, self._L_ref_all[frame_idx],
                self.lambda_l_track, self._dt))
```

Replace with:

```python
        # W^L ballistic tracking (opt-in): track the robot's centroidal angular
        # momentum toward Ig(q)·omega_ref (the human's rotation rate at the robot's own
        # inertia). Needs the previous solved config for the velocity finite difference,
        # so fires from frame_idx >= 1.
        if (self.activate_wl_track
                and getattr(self, "_omega_ref_all", None) is not None
                and frame_idx >= 1 and q_t_last is not None):
            from HoloNew.src.test_socp.centroidal import build_L_track_term
            q_pin_prev = self.pin.qpos_mj_to_q_pin(q_t_last[:36])
            obj_terms.append(build_L_track_term(
                self, _q_pin, q_pin_prev, dqa, self._omega_ref_all[frame_idx],
                self.lambda_l_track, self._dt))
```

- [ ] **Step 4: Run the integration + parity tests**

Run: `pytest tests/test_centroidal_lref.py -v tests/test_retarget_golden.py -v`
Expected: PASS (`test_L_track_reproduces_cartwheel_spin` passes if the SFU clip is present, else skips). If `test_retarget_golden` asserts a changed CoM-reference trajectory, regenerate its golden per that test's documented procedure and note it in the commit.

- [ ] **Step 5: Commit**

```bash
git add src/test_socp/test_socp.py
git commit -m "feat(test_socp): wire SMPL CoM target + I_robot*omega W^L tracking"
```

---

## Task 6: Diagnostics + viewer

**Files:**
- Modify: `src/test_socp/test_socp.py` — `_fill_diagnostics` (the `res.angular_momentum_ref` line, ~line 1138)
- Modify: `src/viewer.py` — `_draw_centroidal` L arrows (~lines 557-618)
- Test: `tests/test_viewer.py` (add a diagnostics-target assertion) and a solver diagnostic check in `tests/test_centroidal_smpl_targets.py`

- [ ] **Step 1: Write the failing diagnostic test**

Append to `tests/test_centroidal_smpl_targets.py`:

```python
def test_fill_diagnostics_angular_momentum_ref_is_I_omega():
    from HoloNew.examples.robot_retarget import RetargetingConfig
    from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    rt.collect_diagnostics = True
    res = rt.retarget(max_frames=12)
    assert res.angular_momentum_ref is not None
    assert res.angular_momentum_ref.shape == res.angular_momentum.shape
    # Frame 5: target == Ig(q_t) @ omega_ref(t).
    t = 5
    q = rt.pin.qpos_mj_to_q_pin(res.qpos[t, :36])
    expected = rt.pin.centroidal_inertia(q) @ rt._omega_ref_all[t]
    np.testing.assert_allclose(res.angular_momentum_ref[t], expected, atol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_centroidal_smpl_targets.py::test_fill_diagnostics_angular_momentum_ref_is_I_omega -v`
Expected: FAIL — `angular_momentum_ref` still equals the old `_L_ref_all` (now `None`) → assertion/`None` error.

- [ ] **Step 3: Update `_fill_diagnostics` in `test_socp.py`**

Find:

```python
        res.com = com
        res.angular_momentum = L
        # Targets used by the centroidal weights, for the viewer to draw alongside the
        # solved quantities (the grounded reference CoM and the W^L reference L).
        res.com_ref = getattr(self, "_c_ref_all", None)
        res.angular_momentum_ref = getattr(self, "_L_ref_all", None)
```

Replace with:

```python
        res.com = com
        res.angular_momentum = L
        # Targets the centroidal weights actually see, for the viewer to draw alongside
        # the solved quantities: the grounded-SMPL CoM target, and the W^L target
        # Ig(q_t)·omega_ref (same quantity as the solved centroidal L → directly
        # comparable).
        res.com_ref = getattr(self, "_c_ref_all", None)
        _omega_ref = getattr(self, "_omega_ref_all", None)
        if _omega_ref is not None:
            L_ref = np.zeros((T, 3))
            for t in range(T):
                q_pin = self.pin.qpos_mj_to_q_pin(res.qpos[t, :36])
                L_ref[t] = self.pin.centroidal_inertia(q_pin) @ _omega_ref[t]
            res.angular_momentum_ref = L_ref
        else:
            res.angular_momentum_ref = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_centroidal_smpl_targets.py::test_fill_diagnostics_angular_momentum_ref_is_I_omega -v`
Expected: PASS

- [ ] **Step 5: Update the viewer L arrows to a shared scale**

In `src/viewer.py`, `_draw_centroidal`, the solved (cyan) and target (orange) L arrows currently each normalise to their own clip peak. Since both are now the same quantity, use one shared scale factor. Replace the arrows block (from `# Angular-momentum arrows from the CoM:` through the orange tip `add_point_cloud` call) with:

```python
        # Angular-momentum arrows from the CoM: cyan = solved L, orange = target L
        # (Ig·omega_ref). Both are now the same physical quantity, so they share one
        # scale factor (the larger clip peak reads ~0.5 m) → the magnitude gap is visible.
        anchor = com if com is not None else com_ref
        if self._tog_L.value and anchor is not None and frame < len(anchor):
            c = np.asarray(anchor[frame], dtype=np.float32)
            L = None if method is None else method.angular_momentum
            L_ref = None if method is None else method.angular_momentum_ref
            peaks = [float(np.linalg.norm(np.asarray(x), axis=1).max())
                     for x in (L, L_ref) if x is not None]
            peak = max(peaks) if peaks else 0.0
            k = 0.5 / peak if peak > 1e-9 else 0.0
            if L is not None and frame < len(L):
                tip = (c + k * np.asarray(L[frame], dtype=np.float32)).astype(np.float32)
                seg = np.stack([c, tip])[None]
                cols = np.broadcast_to((0, 180, 230), (1, 2, 3)).astype(np.uint8)
                self._dynamic_handles.append(self.server.scene.add_line_segments(
                    "/test/L_arrow", seg, cols, line_width=4.0))
                self._dynamic_handles.append(self.server.scene.add_point_cloud(
                    "/test/L_tip", tip[None], np.array([[0, 180, 230]], np.uint8), point_size=0.04))
            if L_ref is not None and frame < len(L_ref):
                tip_r = (c + k * np.asarray(L_ref[frame], dtype=np.float32)).astype(np.float32)
                seg_r = np.stack([c, tip_r])[None]
                cols_r = np.broadcast_to((255, 150, 0), (1, 2, 3)).astype(np.uint8)
                self._dynamic_handles.append(self.server.scene.add_line_segments(
                    "/test/L_ref_arrow", seg_r, cols_r, line_width=4.0))
                self._dynamic_handles.append(self.server.scene.add_point_cloud(
                    "/test/L_ref_tip", tip_r[None], np.array([[255, 150, 0]], np.uint8), point_size=0.04))
```

- [ ] **Step 6: Run the viewer tests**

Run: `pytest tests/test_viewer.py -v`
Expected: PASS (no-op when fields are `None`; arrows created when both present).

- [ ] **Step 7: Commit**

```bash
git add src/test_socp/test_socp.py src/viewer.py tests/test_centroidal_smpl_targets.py
git commit -m "feat(diagnostics): angular_momentum_ref = Ig*omega; shared-scale L arrows"
```

---

## Task 7: Full-suite sanity + cleanup

**Files:**
- Verify no dangling references to removed symbols.

- [ ] **Step 1: Grep for removed symbols**

Run:
```bash
grep -rn "build_lumped_L_term\|reference_orbital_angular_momentum\|mapped_frame_masses_and_names\|_L_ref_all\|_lumped_frames\|_lumped_masses" src/ tests/ examples/
```
Expected: no matches (all removed/renamed).

- [ ] **Step 2: Run the centroidal + retarget + viewer test subset**

Run: `pytest tests/test_centroidal_lref.py tests/test_smpl_centroidal.py tests/test_centroidal_smpl_targets.py tests/test_pin_centroidal_inertia.py tests/test_viewer.py tests/test_retarget_golden.py -v`
Expected: PASS (cartwheel/golden may skip if clips/goldens absent).

- [ ] **Step 3: Run the full suite**

Run: `pytest -q`
Expected: PASS (or pre-existing unrelated skips only).

- [ ] **Step 4: Commit (if any cleanup was needed)**

```bash
git add -A
git commit -m "chore(centroidal): remove dangling lumped-L references"
```

---

## Notes for the implementer

- **Shared working tree:** another session may be editing `src/test_socp/config.py`. Stage only the files each task lists; never `git add -A` across unrelated changes (Task 7's `-A` is only after the grep confirms the tree is yours). Line numbers in this plan are approximate — match on the quoted code, not the line numbers.
- **`scale_xy_robot`** is on the scale config bound as `sc` in `builder.py`; default `1.0` (XY scaling is a no-op today but kept correct for the future).
- **Fallback path:** if the SMPL-X body fails to construct, the builder falls back to the pelvis-proxy CoM and disables W^L tracking (`_omega_ref_all = None`), so non-flight clips keep working.
- **Golden regeneration:** changing the CoM reference from pelvis-proxy to SMPL CoM shifts `W^c_pos`/`W^c`; if `test_retarget_golden` (or any metric golden) is keyed on the default config with centroidal active, regenerate the golden via its documented procedure and call it out in the commit message.
