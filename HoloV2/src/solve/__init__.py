"""Étage ``solve`` — en ligne, dépendant de q : transforme les sorties ``targets`` par-trame (Evaluator + Refs)
en trajectoire ``qpos`` reorientée par une boucle QP linéarisée (SQP/région-de-confiance). Surface publique :
``solve.contracts`` (les types de données), ``solve.config`` (knobs), et ``solve.runner.solve`` (point d'entrée).
Importe la surface publique en amont ``targets`` ; jamais un interne ``targets``. cvxpy est confiné à
``solve/backend/cvxpy.py`` — ``solve`` reste pinocchio/torch-free."""

from .contracts import LinearConstraint, Problem, ResidualBlock, Step, TrustRegion
from .backend import SolveBackend, make_backend

__all__ = ["Problem", "ResidualBlock", "LinearConstraint", "TrustRegion", "Step",
           "SolveBackend", "make_backend"]

from .config import SolveConfig
from .contracts import FrameEval, FrameInfo, SolveTrajectory

# ``solve`` est ré-exporté en lazy via __getattr__ du module pour que ``import src.solve`` nu
# n'attire PAS transitivement ``src.solve.terms`` comme attribut d'effet secondaire (ce qui casserait
# l'assertion de garde d'import léger ``not hasattr(src.solve, "terms")``). Accéder à
# ``src.solve.solve`` ou ``from src.solve import solve`` fonctionne correctement.
def __getattr__(name: str):
    if name == "solve":
        from .runner import solve  # noqa: PLC0415
        globals()[name] = solve    # cache — subsequent access is O(1), no repeated import
        return solve
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ += ["solve", "SolveConfig", "SolveTrajectory", "FrameInfo", "FrameEval"]
