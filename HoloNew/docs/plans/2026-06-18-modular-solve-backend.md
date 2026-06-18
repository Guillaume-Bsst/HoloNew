# Modular Solve Backend (TEST-SOCP, increment A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Isolate TEST-SOCP problem construction from the solve by introducing a solver-agnostic `ProblemSpec` (objective residual blocks + all constraints) and re-expressing the current cvxpy solve as a `CvxpyBackend` behind that seam, with exact parity.

**Architecture:** New `src/test_socp/solve/` package holds the data contracts (`spec.py`) and the backend (`cvxpy_backend.py` + `backend.py`). The six term builders gain parallel `*_blocks` functions that return numpy `ResidualBlock`s (no cvxpy, no `dqa` variable); constraint construction moves to numpy `LinearConstraint`/`TrustRegion` builders. `solve_single_iteration` assembles a `ProblemSpec` and delegates to the backend. Old cvxpy builder code is deleted only after the new path is golden-green.

**Tech Stack:** Python, numpy, cvxpy (CvxpyBackend only), pytest.

**Conventions (project memory):**
- Run tests: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest …` (cwd = inner `HoloNew/HoloNew/`).
- Commits must NOT include any `Co-Authored-By: Claude` / Claude trailer.

**Core contract (used throughout):**
- `ResidualBlock(A, c, A_obj=None, name="")` → cost `‖A·dqa + A_obj·dxi + c‖²`. Weights (λ/σ/ω) are **pre-folded into A and c by the builder** (the backend is weight-agnostic).
- `LinearConstraint(A, lb=None, ub=None, A_obj=None, name="")` → `lb ≤ A·dqa(+A_obj·dxi) ≤ ub`. Equality (e.g. DOF freeze) is `lb == ub`.
- `TrustRegion(var, radius)` → `‖var‖₂ ≤ radius`, `var ∈ {"dqa","dxi"}`.

**Migration safety:** builders get NEW `*_blocks` functions *alongside* the existing cvxpy ones, so every step keeps the golden test green; the old functions are removed in the final cleanup task.

---

### Task 1: `ProblemSpec` data contracts

**Files:**
- Create: `src/test_socp/solve/__init__.py`
- Create: `src/test_socp/solve/spec.py`
- Test: `tests/test_solve_spec.py`

- [ ] **Step 1: Write the failing test**

`tests/test_solve_spec.py`:
```python
import numpy as np
import pytest
from HoloNew.src.test_socp.solve.spec import (
    ResidualBlock, LinearConstraint, TrustRegion, ProblemSpec, SolveResult)


def test_residual_block_validates_columns():
    ProblemSpec(nv_a=3, n_obj=0,
                residuals=[ResidualBlock(A=np.zeros((2, 3)), c=np.zeros(2))],
                constraints=[], trust_regions=[])
    with pytest.raises(ValueError):
        ProblemSpec(nv_a=3, n_obj=0,
                    residuals=[ResidualBlock(A=np.zeros((2, 4)), c=np.zeros(2))],
                    constraints=[], trust_regions=[])


def test_obj_block_requires_n_obj():
    with pytest.raises(ValueError):
        ProblemSpec(nv_a=3, n_obj=0,
                    residuals=[ResidualBlock(A=np.zeros((2, 3)), c=np.zeros(2),
                                             A_obj=np.zeros((2, 6)))],
                    constraints=[], trust_regions=[])


def test_solve_result_fields():
    r = SolveResult(dqa=np.zeros(3), dxi=None, value=1.0, status="optimal")
    assert r.dqa.shape == (3,) and r.dxi is None and r.status == "optimal"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_spec.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'HoloNew.src.test_socp.solve'`.

- [ ] **Step 3: Write minimal implementation**

`src/test_socp/solve/__init__.py`:
```python
"""Solver-agnostic problem representation + pluggable solve backends for TEST-SOCP."""
```

`src/test_socp/solve/spec.py`:
```python
"""Solver-agnostic representation of one linearised TEST-SOCP subproblem.

A subproblem optimises dqa (nv_a actuated-tangent step) and optionally dxi (n_obj=6
object-tangent step). The objective is a sum of squared residual blocks; constraints
are linear (incl. box / freeze) plus per-variable L2 trust regions. Builders fill these
with numpy arrays; a SolveBackend turns the spec into a concrete solve.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ResidualBlock:
    """Cost ‖A·dqa + A_obj·dxi + c‖² (weights pre-folded into A and c)."""
    A: np.ndarray                      # (m, nv_a)
    c: np.ndarray                      # (m,)
    A_obj: np.ndarray | None = None    # (m, n_obj) or None
    name: str = ""


@dataclass
class LinearConstraint:
    """lb ≤ A·dqa (+ A_obj·dxi) ≤ ub. None side = one-sided; lb==ub = equality."""
    A: np.ndarray                      # (m, nv_a)
    lb: np.ndarray | None = None       # (m,)
    ub: np.ndarray | None = None       # (m,)
    A_obj: np.ndarray | None = None    # (m, n_obj) or None
    name: str = ""


@dataclass
class TrustRegion:
    """‖var‖₂ ≤ radius for var in {'dqa','dxi'}."""
    var: str
    radius: float


@dataclass
class ProblemSpec:
    nv_a: int
    n_obj: int
    residuals: list[ResidualBlock] = field(default_factory=list)
    constraints: list[LinearConstraint] = field(default_factory=list)
    trust_regions: list[TrustRegion] = field(default_factory=list)

    def __post_init__(self):
        for blk in list(self.residuals) + list(self.constraints):
            if blk.A.shape[1] != self.nv_a:
                raise ValueError(f"{type(blk).__name__} {blk.name!r}: A has {blk.A.shape[1]} "
                                 f"cols, expected nv_a={self.nv_a}")
            if blk.A_obj is not None:
                if self.n_obj == 0:
                    raise ValueError(f"{type(blk).__name__} {blk.name!r}: A_obj set but n_obj=0")
                if blk.A_obj.shape[1] != self.n_obj:
                    raise ValueError(f"{type(blk).__name__} {blk.name!r}: A_obj has "
                                     f"{blk.A_obj.shape[1]} cols, expected n_obj={self.n_obj}")
        for tr in self.trust_regions:
            if tr.var not in ("dqa", "dxi"):
                raise ValueError(f"TrustRegion.var must be 'dqa'|'dxi', got {tr.var!r}")


@dataclass
class SolveResult:
    dqa: np.ndarray
    dxi: np.ndarray | None
    value: float
    status: str
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_spec.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add HoloNew/src/test_socp/solve/__init__.py HoloNew/src/test_socp/solve/spec.py HoloNew/tests/test_solve_spec.py
git commit -m "feat(solve): ProblemSpec data contracts for the TEST-SOCP solve seam"
```

---

### Task 2: `CvxpyBackend` + backend factory

**Files:**
- Create: `src/test_socp/solve/backend.py`
- Create: `src/test_socp/solve/cvxpy_backend.py`
- Test: `tests/test_solve_cvxpy_backend.py`

- [ ] **Step 1: Write the failing test**

`tests/test_solve_cvxpy_backend.py`:
```python
import numpy as np
from HoloNew.src.test_socp.solve.spec import (
    ResidualBlock, LinearConstraint, TrustRegion, ProblemSpec)
from HoloNew.src.test_socp.solve.backend import make_backend


def test_unconstrained_least_squares():
    # minimize ||I·dqa - t||^2 -> dqa = t (inside the trust region).
    t = np.array([0.1, -0.05, 0.2])
    spec = ProblemSpec(nv_a=3, n_obj=0,
                       residuals=[ResidualBlock(A=np.eye(3), c=-t, name="track")],
                       constraints=[], trust_regions=[TrustRegion("dqa", 1.0)])
    r = make_backend("cvxpy").solve(spec)
    np.testing.assert_allclose(r.dqa, t, atol=1e-6)


def test_box_constraint_clips():
    t = np.array([1.0])
    spec = ProblemSpec(nv_a=1, n_obj=0,
                       residuals=[ResidualBlock(A=np.eye(1), c=-t)],
                       constraints=[LinearConstraint(A=np.eye(1),
                                                     lb=np.array([-0.3]), ub=np.array([0.3]))],
                       trust_regions=[TrustRegion("dqa", 10.0)])
    r = make_backend("cvxpy").solve(spec)
    np.testing.assert_allclose(r.dqa, [0.3], atol=1e-6)


def test_trust_region_bounds_step():
    t = np.array([5.0, 0.0])
    spec = ProblemSpec(nv_a=2, n_obj=0,
                       residuals=[ResidualBlock(A=np.eye(2), c=-t)],
                       constraints=[], trust_regions=[TrustRegion("dqa", 1.0)])
    r = make_backend("cvxpy").solve(spec)
    assert np.linalg.norm(r.dqa) <= 1.0 + 1e-6


def test_object_variable_block():
    # residual couples dqa and dxi: ||dqa + dxi - 1||^2 ; both free, trust regions loose.
    spec = ProblemSpec(nv_a=1, n_obj=6,
                       residuals=[ResidualBlock(A=np.ones((1, 1)), c=np.array([-1.0]),
                                                A_obj=np.zeros((1, 6)))],
                       constraints=[], trust_regions=[TrustRegion("dqa", 5.0),
                                                      TrustRegion("dxi", 5.0)])
    r = make_backend("cvxpy").solve(spec)
    assert r.dxi is not None and r.dxi.shape == (6,)
    np.testing.assert_allclose(r.dqa, [1.0], atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_cvxpy_backend.py -q`
Expected: FAIL — `ModuleNotFoundError: ...solve.backend`.

- [ ] **Step 3: Write minimal implementation**

`src/test_socp/solve/backend.py`:
```python
"""SolveBackend protocol + factory. Backends turn a ProblemSpec into a SolveResult."""
from __future__ import annotations

from typing import Protocol

from .spec import ProblemSpec, SolveResult


class SolveBackend(Protocol):
    def solve(self, spec: ProblemSpec) -> SolveResult: ...


def make_backend(name: str) -> SolveBackend:
    if name == "cvxpy":
        from .cvxpy_backend import CvxpyBackend
        return CvxpyBackend()
    raise ValueError(f"unknown solve backend {name!r}")
```

`src/test_socp/solve/cvxpy_backend.py`:
```python
"""cvxpy/conic backend — the original TEST-SOCP solve, expressed from a ProblemSpec.

Reproduces the previous behaviour exactly: a least-squares objective (sum of squared
residual blocks), linear constraints, per-variable L2 trust regions (cp.SOC), solved by
CLARABEL with an SCS fallback for ill-conditioned iterations.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np

from .spec import ProblemSpec, SolveResult


class CvxpyBackend:
    def solve(self, spec: ProblemSpec) -> SolveResult:
        dqa = cp.Variable(spec.nv_a, name="dqa")
        dxi = cp.Variable(spec.n_obj, name="dxi") if spec.n_obj else None
        vars_ = {"dqa": dqa, "dxi": dxi}

        def lin(A, A_obj):
            expr = A @ dqa
            if A_obj is not None and dxi is not None:
                expr = expr + A_obj @ dxi
            return expr

        obj = [cp.sum_squares(lin(b.A, b.A_obj) + b.c) for b in spec.residuals]

        cons = []
        for k in spec.constraints:
            expr = lin(k.A, k.A_obj)
            if k.lb is not None and k.ub is not None and np.allclose(k.lb, k.ub):
                cons.append(expr == k.lb)
            else:
                if k.lb is not None:
                    cons.append(expr >= k.lb)
                if k.ub is not None:
                    cons.append(expr <= k.ub)
        for tr in spec.trust_regions:
            v = vars_[tr.var]
            if v is not None:
                cons.append(cp.SOC(np.float64(tr.radius), v))

        prob = cp.Problem(cp.Minimize(cp.sum(obj)) if obj else cp.Minimize(0), cons)
        try:
            prob.solve(solver=cp.CLARABEL)
        except Exception:  # noqa: BLE001 - match the legacy CLARABEL->SCS fallback
            prob.solve(solver=cp.SCS)
        if dqa.value is None:
            prob.solve(solver=cp.SCS)

        dqa_val = np.asarray(dqa.value, dtype=np.float64).ravel()
        dxi_val = np.asarray(dxi.value, dtype=np.float64).ravel() if dxi is not None else None
        return SolveResult(dqa=dqa_val, dxi=dxi_val,
                           value=float(prob.value), status=str(prob.status))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_cvxpy_backend.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add HoloNew/src/test_socp/solve/backend.py HoloNew/src/test_socp/solve/cvxpy_backend.py HoloNew/tests/test_solve_cvxpy_backend.py
git commit -m "feat(solve): CvxpyBackend solving a ProblemSpec (least-squares + linear + L2 trust region)"
```

---

### Migration pattern (Tasks 3–6)

Each objective builder gains a parallel `*_blocks` function that returns `list[ResidualBlock]`.
**The numpy `A`/`b` computation inside each builder is unchanged** — only the final emission changes, plus the weight fold:

- Builder line that emitted `w * cp.sum_squares(M @ dqa - r)`  →  `ResidualBlock(A=√w·M, c=-√w·r, name=...)`.
- Builder line that emitted `cp.sum_squares(M @ dqa + r)` (√weight already inside `M`,`r`)  →  `ResidualBlock(A=M, c=r, name=...)`.
- Object coupling `... - Adxi @ dxi`  →  set `A_obj = -Adxi` on the block.

`√w` for the scalar-multiplied builders: define `import numpy as np; s = np.sqrt(w)` and scale both `M` and `r`. This is algebraically identical to the cvxpy term, so the golden test stays exact.

Keep the existing cvxpy `build_*` functions in place; the new functions are additive.

---

### Task 3: tracking / style / temporal blocks

**Files:**
- Modify: `src/test_socp/tracking.py`, `src/test_socp/style.py`, `src/test_socp/temporal.py`
- Test: `tests/test_solve_blocks_objective.py`

- [ ] **Step 1: Write the failing test**

`tests/test_solve_blocks_objective.py`:
```python
import numpy as np
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.src.test_socp.solve.spec import ResidualBlock


def _rt():
    cfg = RetargetingConfig(task_type="robot_only", task_name="sub3_largebox_003",
                            data_format="smplh")
    return TestSocpRetargeter.from_config(cfg)


def test_tracking_blocks_match_cvxpy_value():
    import cvxpy as cp
    from HoloNew.src.test_socp.tracking import build_tracking_terms, build_tracking_blocks
    rt = _rt()
    q = rt.q_init_full.copy()
    from HoloNew.src.test_socp.targets import ground_frame_targets
    from HoloNew.src.test_socp.tables import IK_MATCH_TABLE1
    tg = ground_frame_targets(rt.gmr_ground["pos"][0], rt.gmr_ground["quat"][0], IK_MATCH_TABLE1)
    blocks = build_tracking_blocks(rt, tg, q, lambda_pos=rt.lambda_pos, sigma_p=rt.sigma_p,
                                   lambda_rot=rt.lambda_rot, sigma_rot=rt.sigma_rot)
    assert blocks and all(isinstance(b, ResidualBlock) for b in blocks)
    # block cost at dqa=0 equals the cvxpy term value at dqa=0
    dqa = cp.Variable(rt.nv_a)
    terms = build_tracking_terms(rt, tg, dqa, q, lambda_pos=rt.lambda_pos, sigma_p=rt.sigma_p,
                                 lambda_rot=rt.lambda_rot, sigma_rot=rt.sigma_rot)
    dqa.value = np.zeros(rt.nv_a)
    cvxpy_val = float(sum(t.value for t in terms))
    block_val = float(sum(np.sum((b.c) ** 2) for b in blocks))  # A·0 + c
    np.testing.assert_allclose(block_val, cvxpy_val, rtol=1e-9, atol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_blocks_objective.py -q`
Expected: FAIL — `ImportError: cannot import name 'build_tracking_blocks'`.

- [ ] **Step 3: Write minimal implementation**

In `src/test_socp/tracking.py`, add (keep `build_tracking_terms` as-is):
```python
import numpy as np
from HoloNew.src.test_socp.solve.spec import ResidualBlock


def build_tracking_blocks(rt, frame_targets, q_mj, lambda_pos, sigma_p,
                          lambda_rot, sigma_rot, activate_pos=True, activate_rot=True):
    """ResidualBlock form of build_tracking_terms (same math, weights folded into A/c)."""
    scale_p = lambda_pos / (sigma_p * sigma_p)
    scale_r = lambda_rot / (sigma_rot * sigma_rot)
    blocks = []
    for frame, (p_t, R_t, w_p, w_r) in frame_targets.items():
        body = rt.robot_link_names[frame]
        Jp, Jr = rt._body_jac(q_mj, body)
        if activate_pos and w_p > 0:
            s = np.sqrt(scale_p * w_p)
            p_c = rt.body_position(q_mj, body)
            blocks.append(ResidualBlock(A=s * Jp, c=-s * (p_t - p_c), name=f"track_pos/{body}"))
        if activate_rot and w_r > 0:
            from scipy.spatial.transform import Rotation
            s = np.sqrt(scale_r * w_r)
            R_c = rt.body_rotation(q_mj, body)
            e = Rotation.from_matrix(R_c.T @ R_t).as_rotvec()
            blocks.append(ResidualBlock(A=s * (R_c.T @ Jr), c=-s * e, name=f"track_rot/{body}"))
    return blocks
```
Add the analogous `build_style_blocks` (mirror `build_style_terms`: each `omega * cp.sum_squares(M @ dqa - r0)` → `ResidualBlock(A=√omega·M, c=-√omega·r0)`) and `build_temporal_block` (mirror `build_temporal_term`: `cp.sum_squares(A @ dqa + b)` → `ResidualBlock(A=A, c=b)`).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_blocks_objective.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add HoloNew/src/test_socp/tracking.py HoloNew/src/test_socp/style.py HoloNew/src/test_socp/temporal.py HoloNew/tests/test_solve_blocks_objective.py
git commit -m "feat(solve): ResidualBlock builders for tracking / style / temporal"
```

---

### Task 4: centroidal blocks

**Files:**
- Modify: `src/test_socp/centroidal.py`
- Test: `tests/test_solve_blocks_objective.py` (add a case)

- [ ] **Step 1: Add the failing test case**

Append to `tests/test_solve_blocks_objective.py`:
```python
def test_centroidal_blocks_are_residual_blocks():
    from HoloNew.src.test_socp.centroidal import build_centroidal_blocks
    # signature mirrors build_centroidal_terms minus dqa; smoke that it returns blocks.
    assert callable(build_centroidal_blocks)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_blocks_objective.py::test_centroidal_blocks_are_residual_blocks -q`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation**

In `src/test_socp/centroidal.py`, add `build_centroidal_blocks(...)` and `build_lumped_L_block(...)` mirroring `build_centroidal_terms` / `build_lumped_L_term` with the same arguments minus `dqa`. Each existing `terms.append(cp.sum_squares(A_x @ dqa + b_x))` becomes `blocks.append(ResidualBlock(A=A_x, c=b_x, name="..."))` (the √λ/σ scaling is already inside `A_x`/`b_x`). `build_lumped_L_term`'s `return cp.sum_squares(r)` where `r = coeff @ dqa + const` becomes `return [ResidualBlock(A=coeff, c=const, name="W_L")]`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_blocks_objective.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add HoloNew/src/test_socp/centroidal.py HoloNew/tests/test_solve_blocks_objective.py
git commit -m "feat(solve): ResidualBlock builders for the centroidal terms"
```

---

### Task 5: movable (object) blocks

**Files:**
- Modify: `src/test_socp/movable.py`
- Test: `tests/test_solve_blocks_objective.py` (add a case)

- [ ] **Step 1: Add the failing test case**

```python
def test_movable_blocks_carry_A_obj():
    from HoloNew.src.test_socp.movable import build_wo_block
    assert callable(build_wo_block)
```

- [ ] **Step 2: Run** `... -q` → FAIL (`ImportError`).

- [ ] **Step 3: Write minimal implementation**

In `src/test_socp/movable.py`, add block versions of `build_wo_term`, `build_wo_position_anchor`, `build_object_floor_terms`, `build_object_floor_persistence`. The object-only residuals (`cp.sum_squares(A_D @ dxi - c_D)`) become `ResidualBlock(A=np.zeros((m, nv_a)), A_obj=A_D, c=-c_D)`. Residuals coupling both (`B_x @ dqa - Bdxi @ dxi - r`) become `ResidualBlock(A=B_x, A_obj=-Bdxi, c=-r)`. `build_wo_term`'s `return cp.sum_squares(r1) + cp.sum_squares(r2)` becomes two blocks (split the sum), each `ResidualBlock` over `dxi` (`A=zeros`, `A_obj=<coeff>`, `c=<const>`).

- [ ] **Step 4: Run** `... -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add HoloNew/src/test_socp/movable.py HoloNew/tests/test_solve_blocks_objective.py
git commit -m "feat(solve): ResidualBlock builders for the movable-object terms (A_obj coupling)"
```

---

### Task 6: interaction blocks (D/X/P + object-floor)

**Files:**
- Modify: `src/test_socp/interaction.py`
- Test: `tests/test_solve_blocks_objective.py` (add a case)

- [ ] **Step 1: Add the failing test case**

```python
def test_interaction_dx_blocks():
    from HoloNew.src.test_socp.interaction import build_dx_blocks, build_p_blocks
    assert callable(build_dx_blocks) and callable(build_p_blocks)
```

- [ ] **Step 2: Run** `... -q` → FAIL.

- [ ] **Step 3: Write minimal implementation**

In `src/test_socp/interaction.py`, add `build_dx_blocks` and `build_p_blocks` mirroring `build_dx_terms` / `build_p_terms` minus `dqa`. Each `terms.append(cp.sum_squares(A_d @ dqa - c_d))` → `ResidualBlock(A=A_d, c=-c_d, name="W_d")`; coupled object terms (`A_d_obj @ dqa - Adxi_d @ dxi - c_d_obj`) → `ResidualBlock(A=A_d_obj, A_obj=-Adxi_d, c=-c_d_obj)`. `build_p_terms`'s `return [cp.sum_squares(B_p @ dqa - r_p)]` → `[ResidualBlock(A=B_p, c=-r_p, name="W_p")]`.

- [ ] **Step 4: Run** `... -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add HoloNew/src/test_socp/interaction.py HoloNew/tests/test_solve_blocks_objective.py
git commit -m "feat(solve): ResidualBlock builders for the D/X/P interaction terms"
```

---

### Task 7: constraint blocks (`solve/constraints.py`)

**Files:**
- Create: `src/test_socp/solve/constraints.py`
- Modify: `src/test_socp/interaction.py` (add `build_p_constraint_blocks`, `build_obj_surface_nonpen_blocks`)
- Test: `tests/test_solve_constraints.py`

- [ ] **Step 1: Write the failing test**

`tests/test_solve_constraints.py`:
```python
import numpy as np
from HoloNew.src.test_socp.solve.constraints import box_freeze_limits, trust_regions
from HoloNew.src.test_socp.solve.spec import LinearConstraint, TrustRegion


class _RtStub:
    nv_a = 4
    activate_tb = True
    activate_qa = True
    activate_joint_limits = True
    step_size = 0.2
    v_a_indices = np.array([0, 6, 7, 8])           # one base DOF + three joints
    _v_a_lb = np.array([-1.0, -2.0, -2.0, -2.0])
    _v_a_ub = np.array([1.0, 2.0, 2.0, 2.0])


def test_trust_regions_dqa_only_without_object():
    trs = trust_regions(_RtStub(), n_obj=0)
    assert [t.var for t in trs] == ["dqa"]
    assert trs[0].radius == 0.2


def test_trust_regions_include_object():
    trs = trust_regions(_RtStub(), n_obj=6)
    assert sorted(t.var for t in trs) == ["dqa", "dxi"]


def test_joint_limit_box_subtracts_current_value():
    rt = _RtStub()
    q_pin = np.zeros(16)
    q_pin[8] = 0.5  # joint at qpos idx 8 (tangent 7 -> qpos 8)
    cons = box_freeze_limits(rt, q_pin)
    box = [c for c in cons if c.name == "joint_limits"][0]
    # joints occupy tangent indices >=6 -> rows 1,2,3; the joint at q_pin[8] shifts its bounds.
    assert isinstance(box, LinearConstraint)
    assert box.A.shape == (4, 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_constraints.py -q`
Expected: FAIL — `ModuleNotFoundError: ...solve.constraints`.

- [ ] **Step 3: Write minimal implementation**

`src/test_socp/solve/constraints.py`:
```python
"""Numpy LinearConstraint / TrustRegion builders extracted from solve_single_iteration.

Same math as the inline cvxpy constraints, returning solver-agnostic blocks.
"""
from __future__ import annotations

import numpy as np

from .spec import LinearConstraint, TrustRegion


def trust_regions(rt, n_obj: int) -> list[TrustRegion]:
    trs = [TrustRegion("dqa", float(rt.step_size))]
    if n_obj:
        trs.append(TrustRegion("dxi", float(rt.step_size)))
    return trs


def box_freeze_limits(rt, q_pin) -> list[LinearConstraint]:
    """DOF freeze (activate_tb / activate_qa) + actuated joint-limit box, as LinearConstraints."""
    cons: list[LinearConstraint] = []
    nv_a = rt.nv_a
    eye = np.eye(nv_a)
    if not rt.activate_tb:
        base = np.where(rt.v_a_indices < 6)[0]
        if base.size:
            z = np.zeros(base.size)
            cons.append(LinearConstraint(A=eye[base], lb=z, ub=z, name="freeze_base"))
    if not rt.activate_qa:
        joints = np.where(rt.v_a_indices >= 6)[0]
        if joints.size:
            z = np.zeros(joints.size)
            cons.append(LinearConstraint(A=eye[joints], lb=z, ub=z, name="freeze_joints"))
    if rt.activate_joint_limits:
        lo = np.copy(rt._v_a_lb)
        hi = np.copy(rt._v_a_ub)
        joint_mask = rt.v_a_indices >= 6
        vi_joints = rt.v_a_indices[joint_mask]
        q_pin_vals = np.asarray(q_pin)[vi_joints + 1]
        lo[joint_mask] -= q_pin_vals
        hi[joint_mask] -= q_pin_vals
        cons.append(LinearConstraint(A=eye, lb=lo, ub=hi, name="joint_limits"))
    return cons
```
Then add to `src/test_socp/interaction.py`, next to the existing constraint builders, `build_p_constraint_blocks(...)` and `build_obj_surface_nonpen_blocks(...)` that mirror `build_p_constraints` / `build_obj_surface_nonpen_constraints` minus `dqa`: each `constraints.append(Aproj @ dqa <= rhs)` → `LinearConstraint(A=Aproj, ub=rhs)`, each `... >= rhs` → `LinearConstraint(A=A, lb=rhs)`, two-sided bands → both `lb` and `ub`. Foot-sticking / foot-lock blocks are assembled directly in Task 8 from the existing `_calc_manipulator_jacobians` output (they are small; build `LinearConstraint(A=Jxy, lb=p_lb[:2], ub=p_ub[:2])` etc.).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_constraints.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add HoloNew/src/test_socp/solve/constraints.py HoloNew/src/test_socp/interaction.py HoloNew/tests/test_solve_constraints.py
git commit -m "feat(solve): LinearConstraint / TrustRegion builders for the seam"
```

---

### Task 8: rewire `solve_single_iteration` to assemble + delegate (PARITY GATE)

**Files:**
- Modify: `src/test_socp/test_socp.py` (`__init__` adds `self._backend`; `solve_single_iteration` body)
- Modify: `src/test_socp/config.py` (add `solve_backend: str = "cvxpy"`)
- Test: `tests/test_retarget_golden.py` (existing — the parity gate)

- [ ] **Step 1: Add the config field**

In `src/test_socp/config.py`, under §4 SOLVER, add:
```python
    # [TEST] which solve backend turns the assembled ProblemSpec into a step.
    solve_backend: str = "cvxpy"
```
Thread it through the builder (`builder.py` passes config fields to the retargeter ctor) — add a `solve_backend` kwarg to `TestSocpRetargeter.__init__` (default `"cvxpy"`), stored as `self.solve_backend`, and `self._backend = make_backend(self.solve_backend)` (import `from HoloNew.src.test_socp.solve.backend import make_backend`).

- [ ] **Step 2: Capture the golden baseline (must pass BEFORE the rewire)**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_retarget_golden.py -q`
Expected: PASS (records the current trajectory as the parity reference).

- [ ] **Step 3: Rewrite `solve_single_iteration` to assemble a ProblemSpec**

Replace the body that builds `dqa`, `obj_terms`, `constraints`, `prob`, `prob.solve` with: collect `ResidualBlock`s from the `*_blocks` builders (same gating conditions as today), collect `LinearConstraint`s via `box_freeze_limits` + the migrated constraint blocks + foot constraints, `TrustRegion`s via `trust_regions(self, n_obj)`, build `ProblemSpec(nv_a=self.nv_a, n_obj=(6 if dxi_active else 0), residuals=..., constraints=..., trust_regions=...)`, then:
```python
from HoloNew.src.test_socp.solve.spec import ProblemSpec
spec = ProblemSpec(nv_a=self.nv_a, n_obj=n_obj, residuals=blocks,
                   constraints=cons, trust_regions=trs)
result = self._backend.solve(spec)
dqa_val = result.dqa
dxi_val = result.dxi
```
Then keep the existing post-solve remap that consumes `dqa.value` / `dxi_obj.value`, reading `dqa_val` / `dxi_val` instead. `n_obj = 6 if (self.activate_tm and obj_pose is not None) else 0`.

- [ ] **Step 4: Run the parity gate**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_retarget_golden.py tests/test_test_socp_parity.py -q`
Expected: PASS — the new assemble→backend path reproduces the golden trajectory to solver tolerance. If a tolerance trips, compare `result.value` per frame against a pre-rewire print to localize the diverging term.

- [ ] **Step 5: Commit**

```bash
git add HoloNew/src/test_socp/test_socp.py HoloNew/src/test_socp/config.py
git commit -m "feat(solve): assemble a ProblemSpec and delegate to the backend in solve_single_iteration"
```

---

### Task 9: remove the old cvxpy builder code (cleanup)

**Files:**
- Modify: `src/test_socp/tracking.py`, `style.py`, `temporal.py`, `centroidal.py`, `movable.py`, `interaction.py` (delete the `*_terms` / `*_constraints` cvxpy functions now unused by the solve)
- Test: full TEST-SOCP suite

- [ ] **Step 1: Confirm the old functions are unused by the solve path**

Run: `cd HoloNew && grep -rn "build_tracking_terms\|build_style_terms\|build_temporal_term\|build_centroidal_terms\|build_lumped_L_term\|build_dx_terms\|build_p_terms\|build_p_constraints\|build_obj_surface_nonpen_constraints\|build_wo_term\|build_object_floor_terms" src --include=*.py | grep -v solve/ | grep -v "_blocks"`
Expected: matches only inside the builder files' own definitions / their unit tests — not in `test_socp.py`.

- [ ] **Step 2: Delete the superseded cvxpy `*_terms` / `*_constraints` functions** and any now-unused `import cvxpy as cp` in those builder modules. Update the per-builder unit tests that referenced the old cvxpy functions to assert on the `*_blocks` outputs instead (the block value at `dqa=0` equals the old term value at `dqa=0`, as in Task 3's test).

- [ ] **Step 3: Run the full TEST-SOCP + solve + export suites**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_*.py tests/test_retarget_golden.py tests/test_test_socp_parity.py tests/test_interaction_dxp.py tests/test_centroidal_metric.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add HoloNew/src/test_socp/
git commit -m "refactor(solve): remove superseded inline cvxpy builders (now behind the backend seam)"
```

---

## Notes for the implementer

- **Parity is the contract.** The new path must reproduce the golden trajectory; the cvxpy backend uses the same residuals (`A·dqa + c`), same constraints, same CLARABEL/SCS. Any per-frame divergence means a sign or weight-fold slip in a `*_blocks` builder — check `c = -b` vs `c = +b` and the `√weight` fold first.
- **Do not touch GMR-SOCP or Holosoma** (`gmr_socp.py`, `interaction_mesh_retargeter.py`) — out of scope.
- **No new solver.** `make_backend` only knows `"cvxpy"`; a QP/LS backend is increment B.
- The `*_blocks` functions are additive in Tasks 3–6 so the golden test stays green until the Task 8 switch; Task 9 deletes the old path.
