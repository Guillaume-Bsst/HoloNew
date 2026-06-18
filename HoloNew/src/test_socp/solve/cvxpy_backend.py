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
            ok = prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        except cp.error.SolverError:
            ok = False
        if not ok:
            # CLARABEL occasionally fails on ill-conditioned iterations; SCS (first-order)
            # is more robust to conditioning. Matches the legacy TEST-SOCP fallback.
            prob.solve(solver=cp.SCS)
            if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                raise RuntimeError(f"CvxpyBackend solve failed: {prob.status}")

        dqa_val = np.asarray(dqa.value, dtype=np.float64).ravel()
        dxi_val = np.asarray(dxi.value, dtype=np.float64).ravel() if dxi is not None else None
        return SolveResult(dqa=dqa_val, dxi=dxi_val,
                           value=float(prob.value), status=str(prob.status))
