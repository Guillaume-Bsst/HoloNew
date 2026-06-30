"""Contrats de données de l'étage ``solve`` — la représentation AGNOSTIQUE du solveur d'UN sous-problème
linéarisé + la sortie backend. Classes gelées (``frozen``) de tableaux numpy, numpy-only (pas cvxpy,
pas de logique), importables partout.

Un sous-problème optimise ``dv`` (pas tangent du free-flyer du robot nv-dimensionnel) et optionnellement
``dxi`` (pas tangents SE(3) des n_obj objets). Objectif = Σ blocs résiduels au carré (objectif QP) ;
contraintes = linéaires (incl. box / limites articulaires) + régions de confiance par-DOF. Les builders
(``solve/terms``) les remplissent ; un ``SolveBackend`` transforme le ``Problem`` en ``Step``."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResidualBlock:
    """Coût ``‖A·dv + A_obj·dxi + c‖²`` — les poids DÉJÀ repliés dans A et c. ``m`` lignes."""

    A: np.ndarray            # (m, nv)
    c: np.ndarray            # (m,)
    A_obj: np.ndarray | None  # (m, n_obj*6) or None  (robot<->object coupling)
    name: str                # "C-D", "S-rot"… (diagnostic + per-term cost breakdown)


@dataclass(frozen=True)
class LinearConstraint:
    """``lb ≤ A·dv (+ A_obj·dxi) ≤ ub``. Côté ``None`` = unilatéral ; ``lb == ub`` = égalité."""

    A: np.ndarray             # (m, nv)
    lb: np.ndarray | None     # (m,)
    ub: np.ndarray | None     # (m,)
    A_obj: np.ndarray | None  # (m, n_obj*6) or None
    name: str


@dataclass(frozen=True)
class TrustRegion:
    """``‖var‖_p ≤ radius`` (rayon par-DOF — gère l'hétérogénéité des unités m/rad/articulaire).
    ``norm = -1`` => box ``|var| ≤ radius`` (∞-norm → QP, v1) ; ``norm = 2`` => ellipsoïde L2
    ``‖var ⊘ radius‖₂ ≤ 1`` (SOC → SOCP, futur)."""

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
    """Un sous-problème linéarisé : Σ ``ResidualBlock`` (objectif) + ``LinearConstraint`` + ``TrustRegion``."""

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
    """Sortie backend : le pas optimal + l'état du solveur."""

    dv: np.ndarray            # (nv,)
    dxi: np.ndarray | None    # (n_obj, 6)
    value: float
    status: str


from typing import TYPE_CHECKING

if TYPE_CHECKING:                       # annotations seulement -> contracts reste numpy-only au runtime
    from ..targets import StyleEval, ContactEval


@dataclass(frozen=True)
class FrameEval:
    """Sortie combinée d'évaluateur par trame : la FK de style + le champ/Jacobiennes de contact
    aux ``(q, object_poses)`` courants. Produit par le wrapper ``evaluate`` (``solve/loop.py``), consumé
    par ``assemble``. Un simple conteneur (pas de logique de forme) — les deux membres se valident eux-mêmes."""

    style: "StyleEval"
    contact: "ContactEval"


@dataclass(frozen=True)
class FrameInfo:
    """Diagnostic par-trame de résolution (tuning de poids + benchmark). ``cost_by_term`` est la norme
    résiduelle au carré par terme (``S-pos`` / ``C-D`` / …) au pas convergé — l'outil #1 de tuning."""

    n_iters: int
    status: str
    cost: float
    cost_by_term: dict[str, float]


@dataclass(frozen=True)
class SolveTrajectory:
    """Sortie du runner : la trajectoire ``qpos`` reorientée + les poses d'objets par-trame + diagnostics.
    ``object_poses`` est ``(T, N, 7)`` (pos + quat wxyz) ; ``N = 0`` conserve la forme ``(T, 0, 7)``."""

    qpos: np.ndarray          # (T, nq)
    object_poses: np.ndarray  # (T, N, 7)  pos + quat wxyz
    info: tuple[FrameInfo, ...]

    def __post_init__(self) -> None:
        if self.qpos.ndim != 2:
            raise ValueError(f"qpos must be 2-D (T, nq), got shape {self.qpos.shape}")
        T = self.qpos.shape[0]
        if self.object_poses.ndim != 3 or self.object_poses.shape[2] != 7:
            raise ValueError(
                f"object_poses must be (T, N, 7), got shape {self.object_poses.shape}")
        if self.object_poses.shape[0] != T:
            raise ValueError(
                f"object_poses has {self.object_poses.shape[0]} frames but qpos has {T}")
        if len(self.info) != T:
            raise ValueError(f"info has {len(self.info)} entries but qpos has {T} frames")

    @property
    def n_frames(self) -> int:
        return self.qpos.shape[0]
