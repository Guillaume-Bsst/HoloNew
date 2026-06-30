"""Backends de résolution enfichables. Protocole ``SolveBackend`` + usine ``make_backend`` ; l'impl cvxpy est
``cvxpy.CvxpyBackend`` (cvxpy importé là seulement)."""
from .base import SolveBackend, make_backend

__all__ = ["SolveBackend", "make_backend"]
