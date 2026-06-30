"""reg — amortissement de pas ``‖w_reg·δv‖²`` (une QP bien-conditionnée, pas borné). ``A = w_reg·I(nv)``,
``c = 0``, ``A_obj = None`` (l'objet a son propre ancrage, le terme O). La régularisation de posture vers
une pose nominale (``q_nominal`` Holosoma) est un variant futur noté — v1 est un amortissement pur."""
from __future__ import annotations

import numpy as np

from ..contracts import ResidualBlock
from ..config import SolveConfig


def build_reg(nv: int, cfg: SolveConfig) -> list[ResidualBlock]:
    """Bloc ``reg`` unique : ``A = w_reg·I(nv)``, ``c = 0(nv)``."""
    return [ResidualBlock(A=cfg.w_reg * np.eye(nv), c=np.zeros(nv), A_obj=None, name="reg")]
