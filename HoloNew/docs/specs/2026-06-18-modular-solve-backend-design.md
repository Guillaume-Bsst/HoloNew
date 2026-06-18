# Modular solve backend for TEST-SOCP (increment A — the seam) — design

**Date:** 2026-06-18
**Status:** Design — pending implementation plan
**Scope:** `modules/01_retargeting/HoloNew`, TEST-SOCP solver only (`src/test_socp/`)
**Builds on:** the TEST-SOCP SQP solve (`test_socp.py::solve_single_iteration`) and its
six term builders (`tracking, style, interaction, centroidal, movable, temporal`).

## Goal

Let multiple solve methods (the current cvxpy/conic solve, and later a QP / damped-least-squares
IK) **coexist and be compared** inside TEST-SOCP, by **isolating problem construction from the
solve**. Today the two are fused: each builder computes a numpy `A`/`b` then immediately wraps it
in `cp.sum_squares(A @ dqa - b)`, and `solve_single_iteration` builds the cvxpy `Problem` and calls
`prob.solve()` inline. A non-cvxpy backend can't be plugged in because the problem only exists as a
cvxpy graph, not as explicit `(A, c)` data.

**This spec is increment A only: the seam.** Extract an abstract, solver-agnostic problem
representation (objective residual blocks **and all constraints**, including the currently-OFF hard
ones), and re-express the existing cvxpy solve as one backend behind that seam. **No new solver is
added** — but the seam is complete so a QP/LS backend (a later increment) can consume the whole
problem without re-touching it. Exact parity with the current solve is the success criterion.

(The "SOCP" name is incidental and expected to be dropped later; this design does not rename
anything.)

## Decisions (from brainstorming)

1. **Scope = TEST-SOCP only.** GMR-SOCP and Holosoma keep their own cvxpy solves untouched.
2. **Increment A = the seam only**, no alternative backend yet.
3. **Abstract ALL constraints now** (option 2), not just the active ones — box joint limits, the
   L2 trust regions, AND the hard linear inequalities that are OFF by default (non-penetration,
   foot-sticking, foot-lock, persistence, self-collision). The seam is complete from the start so a
   future QP backend supports them without changing the contract.
4. **Numpy block dataclasses + a backend protocol** (Approach 1), not operator-overloading or a
   solver-only swap — because an explicit `(A, c)` is what a QP/LS backend needs (`H = Σ AᵀA`).
5. **Separate variable references per block** (`A` over `dqa`, optional `A_obj` over `dxi`), not a
   single stacked `x` — keeps each builder local (it knows only its own columns).
6. **Weights pre-folded by the builders.** A `ResidualBlock` is the *effective* squared residual
   `‖A·dqa + A_obj·dxi + c‖²`; the λ/σ/ω weighting logic stays in the builder. The backend is purely
   numeric and weight-agnostic.

## Architecture

```
linearization (Pinocchio Jacobians, unchanged)
        │
        ▼
builders  ── each returns numpy blocks (no cvxpy, no dqa variable):
  tracking/style/interaction/centroidal/movable/temporal
        │   ResidualBlock(s) + LinearConstraint(s)
        ▼
solve_single_iteration  ── assembles a ProblemSpec
        │
        ▼
SolveBackend.solve(spec) ─►  CvxpyBackend  (current behaviour, exact parity)
        │                    (future: QpBackend, DampedLsBackend)
        ▼
   SolveResult(dqa, dxi, value, status)  ── remapped into the iterate as today
```

### New package: `src/test_socp/solve/` (the "solve" side, isolated from construction)

**`spec.py` — the solver-agnostic contract (pure data):**
- `ResidualBlock(A, c, A_obj=None, name="")` — cost `‖A·dqa + A_obj·dxi + c‖²`. `A` is `(m, nv_a)`,
  `c` is `(m,)`, `A_obj` optional `(m, n_obj)`. Weights already folded in by the builder.
- `LinearConstraint(A, lb=None, ub=None, A_obj=None, name="")` — `lb ≤ A·dqa(+A_obj·dxi) ≤ ub`
  (one-sided when `lb`/`ub` is None). Covers box joint limits (`A=I` on joint rows), foot-sticking
  (`Jxy` box), foot-lock (`Jz` box), non-penetration (`A=Ja_n, lb=rhs`), persistence (two-sided).
- `TrustRegion(var, radius)` — `‖var‖₂ ≤ radius`, `var ∈ {"dqa", "dxi"}` (per variable block, as the
  two current `cp.SOC` constraints).
- `ProblemSpec(nv_a, n_obj, residuals, constraints, trust_regions)` — `n_obj=0` when the object
  variable is inactive.
- `SolveResult(dqa, dxi, value, status)` — `dxi=None` when `n_obj=0`.

**`backend.py` — the plug point:**
- `SolveBackend` protocol: `solve(spec: ProblemSpec) -> SolveResult`.
- `make_backend(name: str) -> SolveBackend` factory (only `"cvxpy"` for now).

**`cvxpy_backend.py` — `CvxpyBackend`:** rebuilds the cvxpy problem from a `ProblemSpec` — one
`cp.Variable` per active variable block, `cp.sum_squares(A@dqa + A_obj@dxi + c)` per residual,
`cp.SOC` per trust region, the linear constraints, `cp.Problem(cp.Minimize(cp.sum(...)))`,
`solve(CLARABEL)` with the existing `SCS` fallback. Returns `SolveResult` (status propagated).
**This reproduces the current behaviour exactly.**

### Construction side (the builders) — changed return type

Each builder loses its `dqa` argument and returns numpy blocks instead of cvxpy expressions:
- objective builders (`build_tracking_terms`, `build_style_terms`, `build_dx_terms`,
  `build_temporal_term`, `build_centroidal_terms`, `build_lumped_L_term`, movable `build_wo_*` /
  `build_object_floor_*`) → `list[ResidualBlock]`.
- the inline constraint construction in `solve_single_iteration` (joint limits, foot-stick,
  foot-lock, non-penetration, persistence, self-collision, the two trust regions) → `LinearConstraint`
  / `TrustRegion` blocks.

The weighting (λ, σ, per-point ω) stays inside each builder, folded into `A`/`c`.

### `solve_single_iteration` — becomes assemble-then-solve

1. Compute the linearization (Jacobians) — unchanged.
2. Call builders to collect `residuals`, `constraints`, `trust_regions`.
3. Build `ProblemSpec(nv_a, n_obj, ...)`.
4. `result = self._backend.solve(spec)`.
5. Remap `result.dqa` / `result.dxi` into the iterate (the current post-solve update path).

A config field `solve_backend: str = "cvxpy"` selects the backend; `self._backend = make_backend(...)`
is built once at construction.

## Data flow / contracts

- A `ResidualBlock`'s `A` has `nv_a` columns; `A_obj` (when present) has `n_obj` columns. The
  backend stacks/uses them; builders never see the global layout.
- Residual sign convention is uniform: cost is `‖A·dqa + A_obj·dxi + c‖²`. Builders that previously
  wrote `A@dqa - b` set `c = -b`; those that wrote `A@dqa + b` set `c = b`.
- `n_obj` is 0 when `activate_tm` is off or the object variable is absent; the object trust region
  and object-coupled `A_obj` blocks are omitted accordingly.

## Error handling

- The CLARABEL→SCS fallback lives inside `CvxpyBackend`; on solver failure it sets
  `SolveResult.status` (e.g. `"infeasible"`, `"solver_error"`) and returns the best available step
  (matching today's behaviour where SCS retries an ill-conditioned iteration).
- A malformed block (column count ≠ `nv_a`/`n_obj`) raises a clear `ValueError` at spec assembly.

## Testing

- **spec (pure):** block shape validation; the `c`-sign convention (`A@dqa - b` ↔ `c=-b`).
- **CvxpyBackend (pure):** a small synthetic `ProblemSpec` (1–2 residuals + a box limit + a trust
  region) whose least-squares solution is known analytically; assert the backend matches.
- **builders:** each returns blocks with the expected `A`/`c` for a tiny synthetic linearization
  (extends the existing per-builder tests, which currently assert the cvxpy term values).
- **Parity (the safety net):** `tests/test_retarget_golden.py` must stay green — the cvxpy backend
  reproduces the current trajectory to solver tolerance. The existing TEST-SOCP parity tests
  (`test_test_socp_parity.py`, `test_parity_native_vs_holosoma.py`) likewise.

## New layout

```
src/test_socp/solve/
  __init__.py
  spec.py            # ResidualBlock, LinearConstraint, TrustRegion, ProblemSpec, SolveResult
  backend.py         # SolveBackend protocol + make_backend
  cvxpy_backend.py   # CvxpyBackend (current behaviour, exact parity)
```

Builders stay in place (`src/test_socp/*.py`); only their return type changes.
`solve_single_iteration` assembles the spec and delegates to the backend.

## Out of scope (later increments)

- Any alternative backend (QP/OSQP, damped-LS) — increment B.
- A comparison harness running one problem through several backends — increment C.
- DPP-parametrizing the cvxpy backend — orthogonal; can layer on later.
- Renaming "SOCP" anywhere.
