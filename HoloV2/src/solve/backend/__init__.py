"""Pluggable solve backends. ``SolveBackend`` protocol + ``make_backend`` factory ; the cvxpy impl is
``cvxpy.CvxpyBackend`` (cvxpy imported there only)."""
from .base import SolveBackend, make_backend

__all__ = ["SolveBackend", "make_backend"]
