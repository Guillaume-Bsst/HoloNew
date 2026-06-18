"""Solver-agnostic representation of one linearised TEST-SOCP subproblem.

A subproblem optimises dqa (nv_a actuated-tangent step) and optionally dxi (n_obj=6
object-tangent step). The objective is a sum of squared residual blocks; constraints
are linear (incl. box / freeze) plus per-variable L2 trust regions. Builders fill these
with numpy arrays; a SolveBackend turns the spec into a concrete solve.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ResidualBlock:
    """Cost ‖A·dqa + A_obj·dxi + c‖² (weights pre-folded into A and c)."""
    A: np.ndarray                      # (m, nv_a)
    c: np.ndarray                      # (m,)
    A_obj: np.ndarray | None = None    # (m, n_obj) or None
    name: str = ""


@dataclass
class LinearConstraint:
    """lb ≤ A·dqa (+ A_obj·dxi) ≤ ub. None side = one-sided; lb==ub = equality."""
    A: np.ndarray                      # (m, nv_a)
    lb: np.ndarray | None = None       # (m,)
    ub: np.ndarray | None = None       # (m,)
    A_obj: np.ndarray | None = None    # (m, n_obj) or None
    name: str = ""


@dataclass
class TrustRegion:
    """‖var‖₂ ≤ radius for var in {'dqa','dxi'}."""
    var: str
    radius: float


@dataclass
class ProblemSpec:
    nv_a: int
    n_obj: int
    residuals: list[ResidualBlock] = field(default_factory=list)
    constraints: list[LinearConstraint] = field(default_factory=list)
    trust_regions: list[TrustRegion] = field(default_factory=list)

    def __post_init__(self):
        for blk in list(self.residuals) + list(self.constraints):
            if blk.A.shape[1] != self.nv_a:
                raise ValueError(f"{type(blk).__name__} {blk.name!r}: A has {blk.A.shape[1]} "
                                 f"cols, expected nv_a={self.nv_a}")
            if blk.A_obj is not None:
                if self.n_obj == 0:
                    raise ValueError(f"{type(blk).__name__} {blk.name!r}: A_obj set but n_obj=0")
                if blk.A_obj.shape[1] != self.n_obj:
                    raise ValueError(f"{type(blk).__name__} {blk.name!r}: A_obj has "
                                     f"{blk.A_obj.shape[1]} cols, expected n_obj={self.n_obj}")
        for tr in self.trust_regions:
            if tr.var not in ("dqa", "dxi"):
                raise ValueError(f"TrustRegion.var must be 'dqa'|'dxi', got {tr.var!r}")


@dataclass
class SolveResult:
    dqa: np.ndarray
    dxi: np.ndarray | None
    value: float
    status: str
