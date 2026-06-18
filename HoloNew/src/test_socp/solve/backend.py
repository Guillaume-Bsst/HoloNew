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
