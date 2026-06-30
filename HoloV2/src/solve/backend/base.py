"""Protocole SolveBackend + usine. Un backend transforme un ``Problem`` en ``Step``. Le Problem est
agnostique du solveur ; chaque backend (cvxpy, proxqp plus tard) l'interprète. ``base`` est numpy-only — l'import
lourd du solveur vit dans le module backend concret."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..contracts import Problem, Step


@runtime_checkable
class SolveBackend(Protocol):
    def solve(self, problem: Problem) -> Step:
        """Résout le sous-problème linéarisé -> Step optimal (avec état)."""


def make_backend(name: str) -> SolveBackend:
    """Usine : ``'cvxpy'`` -> ``CvxpyBackend``. L'import cvxpy se fait ici (en lazy), gardant l'import
    du package torch/cvxpy-free jusqu'à ce qu'un backend soit réellement construit."""
    if name == "cvxpy":
        from .cvxpy import CvxpyBackend
        return CvxpyBackend()
    raise ValueError(f"unknown solve backend {name!r} (known: 'cvxpy')")
