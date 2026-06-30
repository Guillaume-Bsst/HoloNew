"""Data contracts of the ``solve`` stage — the solver-AGNOSTIC representation of ONE linearised
subproblem + the backend output. FROZEN dataclasses of numpy arrays, numpy-only (no cvxpy, no logic),
importable everywhere.

A subproblem optimises ``dv`` (nv robot free-flyer tangent step) and optionally ``dxi`` (n_obj object
SE(3) tangent steps). Objective = Σ squared residual blocks (a QP objective); constraints = linear
(incl. box / joint limits) + per-DOF trust regions. Builders (``solve/terms``) fill these; a
``SolveBackend`` turns the ``Problem`` into a ``Step``."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResidualBlock:
    """Cost ``‖A·dv + A_obj·dxi + c‖²`` — weights ALREADY folded into A and c. ``m`` rows."""

    A: np.ndarray            # (m, nv)
    c: np.ndarray            # (m,)
    A_obj: np.ndarray | None  # (m, n_obj*6) or None  (robot<->object coupling)
    name: str                # "C-D", "S-rot"… (diagnostic + per-term cost breakdown)


@dataclass(frozen=True)
class LinearConstraint:
    """``lb ≤ A·dv (+ A_obj·dxi) ≤ ub``. ``None`` side = one-sided ; ``lb == ub`` = equality."""

    A: np.ndarray             # (m, nv)
    lb: np.ndarray | None     # (m,)
    ub: np.ndarray | None     # (m,)
    A_obj: np.ndarray | None  # (m, n_obj*6) or None
    name: str


@dataclass(frozen=True)
class TrustRegion:
    """``‖var‖_p ≤ radius`` (PER-DOF radius — handles the m/rad/joint unit heterogeneity).
    ``norm = -1`` => box ``|var| ≤ radius`` (∞-norm → QP, v1) ; ``norm = 2`` => L2 ellipsoid
    ``‖var ⊘ radius‖₂ ≤ 1`` (SOC → SOCP, future)."""

    var: str                  # 'dv' | 'dxi'
    radius: np.ndarray        # (nv,) or (n_obj*6,)
    norm: int                 # -1 (box) | 2 (L2)

    def __post_init__(self) -> None:
        if self.var not in ("dv", "dxi"):
            raise ValueError(f"TrustRegion.var must be 'dv'|'dxi', got {self.var!r}")
        if self.norm not in (-1, 2):
            raise ValueError(f"TrustRegion.norm must be -1 (box) or 2 (L2), got {self.norm}")
        if np.any(np.asarray(self.radius) <= 0.0):
            raise ValueError("TrustRegion.radius must be > 0 (per-DOF)")


@dataclass(frozen=True)
class Problem:
    """One linearised subproblem: Σ ``ResidualBlock`` (objective) + ``LinearConstraint`` + ``TrustRegion``."""

    nv: int
    n_obj: int
    residuals: tuple[ResidualBlock, ...]
    constraints: tuple[LinearConstraint, ...]
    trust_regions: tuple[TrustRegion, ...]

    def __post_init__(self) -> None:
        for blk in list(self.residuals) + list(self.constraints):
            if blk.A.ndim != 2 or blk.A.shape[1] != self.nv:
                raise ValueError(f"{type(blk).__name__} {blk.name!r}: A has shape {blk.A.shape}, "
                                 f"expected (m, nv={self.nv})")
            m = blk.A.shape[0]
            vecs = [blk.c] if isinstance(blk, ResidualBlock) else [blk.lb, blk.ub]
            for vec in vecs:
                if vec is not None and vec.shape[0] != m:
                    raise ValueError(f"{type(blk).__name__} {blk.name!r}: A has {m} rows but a "
                                     f"vector has {vec.shape[0]}")
            if blk.A_obj is not None:
                if self.n_obj == 0:
                    raise ValueError(f"{type(blk).__name__} {blk.name!r}: A_obj set but n_obj=0")
                if blk.A_obj.shape != (m, self.n_obj * 6):
                    raise ValueError(f"{type(blk).__name__} {blk.name!r}: A_obj has shape "
                                     f"{blk.A_obj.shape}, expected ({m}, {self.n_obj * 6})")
        for tr in self.trust_regions:
            k = self.nv if tr.var == "dv" else self.n_obj * 6
            if np.asarray(tr.radius).shape != (k,):
                raise ValueError(f"TrustRegion {tr.var!r}: radius shape {np.asarray(tr.radius).shape} "
                                 f"!= ({k},)")


@dataclass(frozen=True)
class Step:
    """Backend output: the optimal step + solver status."""

    dv: np.ndarray            # (nv,)
    dxi: np.ndarray | None    # (n_obj, 6)
    value: float
    status: str
