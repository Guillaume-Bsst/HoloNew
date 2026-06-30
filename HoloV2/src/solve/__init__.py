"""``solve`` stage — online, q-DEPENDENT: turns the per-frame ``targets`` outputs (Evaluator + Refs)
into the retargeted ``qpos`` trajectory by a linearised QP (SQP/trust-region) loop. Public surface:
``solve.contracts`` (the data types), ``solve.config`` (knobs), and ``solve.runner.solve`` (entry).
Imports the upstream ``targets`` public surface; never a ``targets`` internal. cvxpy is confined to
``solve/backend/cvxpy.py`` — ``solve`` stays pinocchio/torch-free."""

from .contracts import LinearConstraint, Problem, ResidualBlock, Step, TrustRegion
from .backend import SolveBackend, make_backend

__all__ = ["Problem", "ResidualBlock", "LinearConstraint", "TrustRegion", "Step",
           "SolveBackend", "make_backend"]

from .config import SolveConfig
from .contracts import FrameEval, FrameInfo, SolveTrajectory

# ``solve`` is re-exported lazily via module __getattr__ so that a bare ``import src.solve``
# does NOT transitively pull ``src.solve.terms`` as side-effect attribute (which would break
# the light-import guard assertion ``not hasattr(src.solve, "terms")``).  Accessing
# ``src.solve.solve`` or ``from src.solve import solve`` both work correctly.
def __getattr__(name: str):
    if name == "solve":
        from .runner import solve  # noqa: PLC0415
        globals()[name] = solve    # cache — subsequent access is O(1), no repeated import
        return solve
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ += ["solve", "SolveConfig", "SolveTrajectory", "FrameInfo", "FrameEval"]
