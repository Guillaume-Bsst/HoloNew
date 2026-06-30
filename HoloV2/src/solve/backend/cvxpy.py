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
