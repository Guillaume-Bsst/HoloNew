# solve — cœur solveur (Plan A) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Poser le cœur **solveur-agnostique** de l'étage `solve/` : les contrats `Problem`/`Step` (un sous-problème QP linéarisé) et le backend enfichable (`SolveBackend` Protocol + `CvxpyBackend`), testables sur des `Problem` synthétiques — **sans aucune dépendance à `targets`**.

**Architecture:** `solve/contracts.py` définit le `Problem` (liste de `ResidualBlock` `‖A·dv + A_obj·dξ + c‖²` + `LinearConstraint` + `TrustRegion`) et le `Step` (sortie backend). `solve/backend/` = le Protocol + `CvxpyBackend`, qui traduit un `Problem` en `cp.Problem` (objectif `sum_squares`, contraintes affines, trust-region **box** → QP) et le résout (CVXPY route vers OSQP). Premier des 3 incréments (B = termes, C = orchestration).

**Tech Stack:** Python, numpy (float64), cvxpy 1.9.1 (OSQP/CLARABEL présents), pytest.

## Global Constraints

- `solve/contracts.py` reste **numpy-only à l'import** ; **cvxpy importé UNIQUEMENT dans `solve/backend/cvxpy.py`**. `solve` reste pinocchio/torch-free.
- Compute en **float64**.
- Imports **relatifs** dans `src/`, **absolus** (`from src.…`) dans `tests/`.
- Invariants de contrat → `raise ValueError` explicite au `__post_init__` (cf. `MultiChannelField`).
- Trust-region **v1 = box** (`norm = -1`, `‖dv‖∞ ≤ r` par-DOF) ; `norm = 2` (L2/SOC) = incrément futur → le backend lève `NotImplementedError`.
- Convention `Step.dxi` : `(n_obj, 6)` ; `dv` : `(nv,)`. `nv` = tangente free-flyer (= 6 + n_joints) — mais le cœur est **agnostique** (il ne voit que des dimensions).
- Tests dans `HoloV2/tests/`, lancés depuis `HoloV2/` avec
  `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/<f> -q`.
- Commits **conventionnels, français**. **NE JAMAIS tagger Claude** (aucun `Co-Authored-By`/mention). Auteur : `Guillaume-Bsst`.

---

### Task 1 : `solve/contracts.py` — les types du problème

**Files:**
- Create: `src/solve/__init__.py`
- Create: `src/solve/contracts.py`
- Test: `tests/test_solve_contracts.py`

**Interfaces:**
- Produces : `ResidualBlock(A (m,nv), c (m,), A_obj (m,n_obj*6)|None, name)`, `LinearConstraint(A, lb|None, ub|None, A_obj|None, name)`, `TrustRegion(var: 'dv'|'dxi', radius (k,), norm: int)`, `Problem(nv, n_obj, residuals, constraints, trust_regions)` (validé), `Step(dv (nv,), dxi (n_obj,6)|None, value, status)`.

- [ ] **Step 1 : Écrire le test**

```python
# tests/test_solve_contracts.py
"""Contrats solve : construction valide + rejet des formes incohérentes au __post_init__."""
import numpy as np
import pytest

from src.solve.contracts import (LinearConstraint, Problem, ResidualBlock, Step, TrustRegion)


def test_problem_valid_construction():
    nv, n_obj = 5, 1
    r = ResidualBlock(A=np.zeros((3, nv)), c=np.zeros(3),
                      A_obj=np.zeros((3, n_obj * 6)), name="C-D")
    lc = LinearConstraint(A=np.zeros((2, nv)), lb=np.zeros(2), ub=None, A_obj=None, name="jl")
    tr = TrustRegion(var="dv", radius=np.full(nv, 0.1), norm=-1)
    p = Problem(nv=nv, n_obj=n_obj, residuals=(r,), constraints=(lc,), trust_regions=(tr,))
    assert p.nv == nv and len(p.residuals) == 1


def test_residual_block_bad_A_cols_raises():
    with pytest.raises(ValueError):
        Problem(nv=5, n_obj=0,
                residuals=(ResidualBlock(A=np.zeros((3, 4)), c=np.zeros(3), A_obj=None, name="x"),),
                constraints=(), trust_regions=())


def test_residual_block_A_obj_without_n_obj_raises():
    with pytest.raises(ValueError):
        Problem(nv=5, n_obj=0,
                residuals=(ResidualBlock(A=np.zeros((3, 5)), c=np.zeros(3),
                                         A_obj=np.zeros((3, 6)), name="x"),),
                constraints=(), trust_regions=())


def test_residual_block_row_mismatch_raises():
    with pytest.raises(ValueError):                       # A 3 rows, c 2 rows
        Problem(nv=5, n_obj=0,
                residuals=(ResidualBlock(A=np.zeros((3, 5)), c=np.zeros(2), A_obj=None, name="x"),),
                constraints=(), trust_regions=())


def test_trust_region_bad_var_and_radius():
    with pytest.raises(ValueError):
        TrustRegion(var="dz", radius=np.ones(3), norm=-1)
    with pytest.raises(ValueError):
        TrustRegion(var="dv", radius=np.array([-1.0, 1.0]), norm=-1)


def test_step_shapes():
    s = Step(dv=np.zeros(5), dxi=np.zeros((1, 6)), value=0.0, status="optimal")
    assert s.dv.shape == (5,) and s.dxi.shape == (1, 6)
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_contracts.py -q`
Expected: FAIL (`ModuleNotFoundError: src.solve.contracts`).

- [ ] **Step 3 : Créer `src/solve/__init__.py`**

```python
"""``solve`` stage — online, q-DEPENDENT: turns the per-frame ``targets`` outputs (Evaluator + Refs)
into the retargeted ``qpos`` trajectory by a linearised QP (SQP/trust-region) loop. Public surface:
``solve.contracts`` (the data types), ``solve.config`` (knobs), and ``solve.runner.solve`` (entry).
Imports the upstream ``targets`` public surface; never a ``targets`` internal. cvxpy is confined to
``solve/backend/cvxpy.py`` — ``solve`` stays pinocchio/torch-free."""
```

- [ ] **Step 4 : Écrire `src/solve/contracts.py`**

```python
"""Data contracts of the ``solve`` stage — the solver-AGNOSTIC representation of ONE linearised
subproblem + the backend output. FROZEN dataclasses of numpy arrays, numpy-only (no cvxpy, no logic),
importable everywhere.

A subproblem optimises ``dv`` (nv robot free-flyer tangent step) and optionally ``dxi`` (n_obj object
SE(3) tangent steps). Objective = Σ squared residual blocks (a QP objective); constraints = linear
(incl. box / joint limits) + per-DOF trust regions. Builders (``solve/terms``) fill these; a
``SolveBackend`` turns the ``Problem`` into a ``Step``."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResidualBlock:
    """Cost ``‖A·dv + A_obj·dxi + c‖²`` — weights ALREADY folded into A and c. ``m`` rows."""

    A: np.ndarray            # (m, nv)
    c: np.ndarray            # (m,)
    A_obj: np.ndarray | None  # (m, n_obj*6) or None  (robot<->object coupling)
    name: str                # "C-D", "S-rot"… (diagnostic + per-term cost breakdown)


@dataclass(frozen=True)
class LinearConstraint:
    """``lb ≤ A·dv (+ A_obj·dxi) ≤ ub``. ``None`` side = one-sided ; ``lb == ub`` = equality."""

    A: np.ndarray             # (m, nv)
    lb: np.ndarray | None     # (m,)
    ub: np.ndarray | None     # (m,)
    A_obj: np.ndarray | None  # (m, n_obj*6) or None
    name: str


@dataclass(frozen=True)
class TrustRegion:
    """``‖var‖_p ≤ radius`` (PER-DOF radius — handles the m/rad/joint unit heterogeneity).
    ``norm = -1`` => box ``|var| ≤ radius`` (∞-norm → QP, v1) ; ``norm = 2`` => L2 ellipsoid
    ``‖var ⊘ radius‖₂ ≤ 1`` (SOC → SOCP, future)."""

    var: str                  # 'dv' | 'dxi'
    radius: np.ndarray        # (nv,) or (n_obj*6,)
    norm: int                 # -1 (box) | 2 (L2)

    def __post_init__(self) -> None:
        if self.var not in ("dv", "dxi"):
            raise ValueError(f"TrustRegion.var must be 'dv'|'dxi', got {self.var!r}")
        if self.norm not in (-1, 2):
            raise ValueError(f"TrustRegion.norm must be -1 (box) or 2 (L2), got {self.norm}")
        if np.any(np.asarray(self.radius) <= 0.0):
            raise ValueError("TrustRegion.radius must be > 0 (per-DOF)")


@dataclass(frozen=True)
class Problem:
    """One linearised subproblem: Σ ``ResidualBlock`` (objective) + ``LinearConstraint`` + ``TrustRegion``."""

    nv: int
    n_obj: int
    residuals: tuple[ResidualBlock, ...]
    constraints: tuple[LinearConstraint, ...]
    trust_regions: tuple[TrustRegion, ...]

    def __post_init__(self) -> None:
        for blk in list(self.residuals) + list(self.constraints):
            if blk.A.ndim != 2 or blk.A.shape[1] != self.nv:
                raise ValueError(f"{type(blk).__name__} {blk.name!r}: A has shape {blk.A.shape}, "
                                 f"expected (m, nv={self.nv})")
            m = blk.A.shape[0]
            ref = blk.c if isinstance(blk, ResidualBlock) else (blk.lb if blk.lb is not None else blk.ub)
            if ref is not None and ref.shape[0] != m:
                raise ValueError(f"{type(blk).__name__} {blk.name!r}: A has {m} rows but a "
                                 f"vector has {ref.shape[0]}")
            if blk.A_obj is not None:
                if self.n_obj == 0:
                    raise ValueError(f"{type(blk).__name__} {blk.name!r}: A_obj set but n_obj=0")
                if blk.A_obj.shape != (m, self.n_obj * 6):
                    raise ValueError(f"{type(blk).__name__} {blk.name!r}: A_obj has shape "
                                     f"{blk.A_obj.shape}, expected ({m}, {self.n_obj * 6})")
        for tr in self.trust_regions:
            k = self.nv if tr.var == "dv" else self.n_obj * 6
            if np.asarray(tr.radius).shape != (k,):
                raise ValueError(f"TrustRegion {tr.var!r}: radius shape {np.asarray(tr.radius).shape} "
                                 f"!= ({k},)")


@dataclass(frozen=True)
class Step:
    """Backend output: the optimal step + solver status."""

    dv: np.ndarray            # (nv,)
    dxi: np.ndarray | None    # (n_obj, 6)
    value: float
    status: str
```

- [ ] **Step 5 : Lancer, vérifier le succès + import numpy-only**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_contracts.py -q`
Expected: PASS (6 tests).

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -c "import sys; import src.solve.contracts; assert 'cvxpy' not in sys.modules and 'torch' not in sys.modules; print('contracts numpy-only ok')"`
Expected: `contracts numpy-only ok`

- [ ] **Step 6 : Commit**

```bash
git add src/solve/__init__.py src/solve/contracts.py tests/test_solve_contracts.py
git commit -m "feat(holov2): solve/contracts — Problem/ResidualBlock/LinearConstraint/TrustRegion/Step (QP linéarisé, validé)"
```

---

### Task 2 : `solve/backend/` — Protocol + `CvxpyBackend`

**Files:**
- Create: `src/solve/backend/__init__.py`
- Create: `src/solve/backend/base.py`
- Create: `src/solve/backend/cvxpy.py`
- Test: `tests/test_solve_backend_cvxpy.py`

**Interfaces:**
- Consumes : `Problem`, `Step` (Task 1).
- Produces : `SolveBackend` Protocol (`solve(problem: Problem) -> Step`) ; `CvxpyBackend()` ; `make_backend(name: str) -> SolveBackend`.

- [ ] **Step 1 : Écrire le test (QP box connu → optimum analytique)**

```python
# tests/test_solve_backend_cvxpy.py
"""CvxpyBackend : un Problem connu -> Step. Box trust-region (QP, route OSQP)."""
import numpy as np
import pytest

from src.solve.contracts import LinearConstraint, Problem, ResidualBlock, TrustRegion
from src.solve.backend.cvxpy import CvxpyBackend


def test_box_trust_region_clamps_to_optimum():
    # min ‖dv - [1,2,3]‖²  s.c. |dv| ≤ 0.5  ->  dv = [0.5, 0.5, 0.5] (chaque coord saturée).
    nv = 3
    r = ResidualBlock(A=np.eye(nv), c=-np.array([1.0, 2.0, 3.0]), A_obj=None, name="track")
    tr = TrustRegion(var="dv", radius=np.full(nv, 0.5), norm=-1)
    p = Problem(nv=nv, n_obj=0, residuals=(r,), constraints=(), trust_regions=(tr,))
    step = CvxpyBackend().solve(p)
    assert step.status in ("optimal", "optimal_inaccurate")
    assert np.allclose(step.dv, [0.5, 0.5, 0.5], atol=1e-4)
    assert step.dxi is None


def test_linear_constraint_active():
    # min ‖dv‖²  s.c. dv[0] ≥ 1 (one-sided lb) -> dv = [1, 0].
    nv = 2
    r = ResidualBlock(A=np.eye(nv), c=np.zeros(nv), A_obj=None, name="reg")
    lc = LinearConstraint(A=np.array([[1.0, 0.0]]), lb=np.array([1.0]), ub=None, A_obj=None, name="c")
    tr = TrustRegion(var="dv", radius=np.full(nv, 10.0), norm=-1)
    p = Problem(nv=nv, n_obj=0, residuals=(r,), constraints=(lc,), trust_regions=(tr,))
    step = CvxpyBackend().solve(p)
    assert np.allclose(step.dv, [1.0, 0.0], atol=1e-4)


def test_object_coupling_dxi():
    # min ‖dv - 1‖² + ‖dxi - 2‖²  s.c. box 0.5 -> dv=0.5, dxi=0.5 ; dxi shape (1,6).
    nv, n_obj = 1, 1
    r1 = ResidualBlock(A=np.ones((1, nv)), c=-np.ones(1), A_obj=None, name="dv")
    r2 = ResidualBlock(A=np.zeros((6, nv)), c=-2.0 * np.ones(6),
                       A_obj=np.eye(6), name="dxi")          # ‖dxi - 2‖²
    trv = TrustRegion(var="dv", radius=np.full(nv, 0.5), norm=-1)
    trx = TrustRegion(var="dxi", radius=np.full(n_obj * 6, 0.5), norm=-1)
    p = Problem(nv=nv, n_obj=n_obj, residuals=(r1, r2), constraints=(), trust_regions=(trv, trx))
    step = CvxpyBackend().solve(p)
    assert step.dxi.shape == (1, 6)
    assert np.allclose(step.dv, [0.5], atol=1e-4) and np.allclose(step.dxi, 0.5, atol=1e-4)


def test_l2_norm_not_implemented():
    nv = 2
    p = Problem(nv=nv, n_obj=0,
                residuals=(ResidualBlock(A=np.eye(nv), c=np.zeros(nv), A_obj=None, name="r"),),
                constraints=(), trust_regions=(TrustRegion(var="dv", radius=np.ones(nv), norm=2),))
    with pytest.raises(NotImplementedError):
        CvxpyBackend().solve(p)
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_backend_cvxpy.py -q`
Expected: FAIL (`ModuleNotFoundError: src.solve.backend.cvxpy`).

- [ ] **Step 3 : `src/solve/backend/base.py` — le Protocol**

```python
"""SolveBackend protocol + factory. A backend turns a ``Problem`` into a ``Step``. The Problem is
solver-agnostic; each backend (cvxpy, later proxqp) interprets it. ``base`` is numpy-only — the heavy
solver import lives in the concrete backend module."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..contracts import Problem, Step


@runtime_checkable
class SolveBackend(Protocol):
    def solve(self, problem: Problem) -> Step:
        """Solve the linearised subproblem -> optimal Step (with status)."""


def make_backend(name: str) -> SolveBackend:
    """Factory: ``'cvxpy'`` -> ``CvxpyBackend``. The cvxpy import happens here (lazily), keeping the
    package import torch/cvxpy-free until a backend is actually built."""
    if name == "cvxpy":
        from .cvxpy import CvxpyBackend
        return CvxpyBackend()
    raise ValueError(f"unknown solve backend {name!r} (known: 'cvxpy')")
```

- [ ] **Step 4 : `src/solve/backend/cvxpy.py` — `CvxpyBackend`**

```python
"""cvxpy backend — translates a ``Problem`` into a ``cp.Problem`` and solves it. Objective = sum of
``cp.sum_squares`` over the residual blocks ; constraints = affine ; trust regions = box (∞-norm →
QP ; CVXPY routes to OSQP). **cvxpy is imported ONLY here** — the rest of ``solve`` stays cvxpy-free."""
from __future__ import annotations

import numpy as np

from ..contracts import Problem, Step


class CvxpyBackend:
    """``SolveBackend`` impl over cvxpy. v1 : box trust regions only (QP)."""

    def solve(self, problem: Problem) -> Step:
        import cvxpy as cp

        dv = cp.Variable(problem.nv)
        dxi = cp.Variable(problem.n_obj * 6) if problem.n_obj > 0 else None

        def lin(A, A_obj):
            e = A @ dv
            if A_obj is not None and dxi is not None:
                e = e + A_obj @ dxi
            return e

        cost = 0
        for b in problem.residuals:
            cost = cost + cp.sum_squares(lin(b.A, b.A_obj) + b.c)

        cons = []
        for lc in problem.constraints:
            e = lin(lc.A, lc.A_obj)
            if lc.lb is not None:
                cons.append(e >= lc.lb)
            if lc.ub is not None:
                cons.append(e <= lc.ub)
        for tr in problem.trust_regions:
            var = dv if tr.var == "dv" else dxi
            if var is None:
                continue
            if tr.norm == -1:                                   # box (∞-norm) -> QP
                cons.append(var <= tr.radius)
                cons.append(var >= -tr.radius)
            else:                                               # norm == 2 (L2/SOC) — v1 = box only
                raise NotImplementedError("L2 trust region (norm=2) is a future increment; v1 = box")

        cp_prob = cp.Problem(cp.Minimize(cost), cons)
        cp_prob.solve()

        dv_val = np.asarray(dv.value, np.float64) if dv.value is not None else np.full(problem.nv, np.nan)
        dxi_val = (np.asarray(dxi.value, np.float64).reshape(problem.n_obj, 6)
                   if (dxi is not None and dxi.value is not None) else None)
        return Step(dv=dv_val, dxi=dxi_val,
                    value=float(cp_prob.value) if cp_prob.value is not None else float("nan"),
                    status=str(cp_prob.status))
```

- [ ] **Step 5 : `src/solve/backend/__init__.py` + surface publique `solve/__init__.py`**

Créer `src/solve/backend/__init__.py` :

```python
"""Pluggable solve backends. ``SolveBackend`` protocol + ``make_backend`` factory ; the cvxpy impl is
``cvxpy.CvxpyBackend`` (cvxpy imported there only)."""
from .base import SolveBackend, make_backend

__all__ = ["SolveBackend", "make_backend"]
```

Étendre `src/solve/__init__.py` ( APPEND après le docstring) :

```python
from .contracts import LinearConstraint, Problem, ResidualBlock, Step, TrustRegion
from .backend import SolveBackend, make_backend

__all__ = ["Problem", "ResidualBlock", "LinearConstraint", "TrustRegion", "Step",
           "SolveBackend", "make_backend"]
```

- [ ] **Step 6 : Lancer, vérifier le succès**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_solve_backend_cvxpy.py -q`
Expected: PASS (4 tests).

- [ ] **Step 7 : Vérifier que le package `solve` reste cvxpy-free à l'import (cvxpy seulement quand un backend résout)**

Run: `~/.holonew_deps/miniconda3/envs/holonew/bin/python -c "import sys; import src.solve; assert 'cvxpy' not in sys.modules, 'cvxpy leaked at import!'; from src.solve import make_backend; print('solve import cvxpy-free ok')"`
Expected: `solve import cvxpy-free ok`

> Note : `make_backend('cvxpy')` importe cvxpy (lazy, dans la factory) ; `CvxpyBackend.solve` aussi. Le seul module qui importe cvxpy au chargement serait `backend/cvxpy.py` — mais il n'est importé que via la factory/lazy, donc `import src.solve` reste cvxpy-free.

- [ ] **Step 8 : Commit**

```bash
git add src/solve/backend/ src/solve/__init__.py tests/test_solve_backend_cvxpy.py
git commit -m "feat(holov2): solve/backend — SolveBackend protocol + CvxpyBackend (QP box -> OSQP), cvxpy confiné"
```

---

## Self-Review

**1. Spec coverage** (vs `2026-06-30-solve-stage-design.md`, périmètre Plan A) :
- `Problem`/`ResidualBlock`/`LinearConstraint`/`TrustRegion`/`Step` (contrats) → Task 1. ✅
- `TrustRegion` box par-DOF (norm=-1), champ `norm` pour L2 futur → Task 1 + Task 2 (box ; L2 `NotImplementedError`). ✅
- `SolveBackend` Protocol + `CvxpyBackend`, **cvxpy confiné à `backend/cvxpy.py`** → Task 2 + Step 7 (assert). ✅
- `solve` pinocchio/torch-free + contracts numpy-only → Task 1 Step 5, Task 2 Step 7. ✅
- Validation des formes au `__post_init__` (style `MultiChannelField`) → Task 1. ✅
- (Hors Plan A → Plan B : `config.py`, `terms/` + `_ops`, `constraints` ; → Plan C : `init`, `retract`, `assemble`, `loop`, `runner`, `SolveTrajectory`/`FrameInfo`.)

**2. Placeholder scan** : aucun `TBD`/`TODO` ; chaque step porte code/commande réels. ✅

**3. Type consistency** : `Problem(nv, n_obj, residuals, constraints, trust_regions)`, `ResidualBlock(A, c, A_obj, name)`, `TrustRegion(var, radius, norm)`, `Step(dv, dxi, value, status)`, `SolveBackend.solve(Problem)->Step`, `make_backend(name)->SolveBackend` — cohérents entre Task 1, Task 2 et les tests. `dxi` partout `(n_obj, 6)`. ✅
