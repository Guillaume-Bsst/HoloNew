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
