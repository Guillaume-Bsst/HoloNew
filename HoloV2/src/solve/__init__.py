"""``solve`` stage — online, q-DEPENDENT: turns the per-frame ``targets`` outputs (Evaluator + Refs)
into the retargeted ``qpos`` trajectory by a linearised QP (SQP/trust-region) loop. Public surface:
``solve.contracts`` (the data types), ``solve.config`` (knobs), and ``solve.runner.solve`` (entry).
Imports the upstream ``targets`` public surface; never a ``targets`` internal. cvxpy is confined to
``solve/backend/cvxpy.py`` — ``solve`` stays pinocchio/torch-free."""

from .contracts import LinearConstraint, Problem, ResidualBlock, Step, TrustRegion
from .backend import SolveBackend, make_backend

__all__ = ["Problem", "ResidualBlock", "LinearConstraint", "TrustRegion", "Step",
           "SolveBackend", "make_backend"]
