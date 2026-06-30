"""reg — step damping ``‖w_reg·δv‖²`` (a well-conditioned QP, bounded step). ``A = w_reg·I(nv)``,
``c = 0``, ``A_obj = None`` (the object has its own anchor, the O term). Posture regularisation toward
a nominal pose (Holosoma ``q_nominal``) is a noted future variant — v1 is plain damping."""
from __future__ import annotations

import numpy as np

from ..contracts import ResidualBlock
from ..config import SolveConfig


def build_reg(nv: int, cfg: SolveConfig) -> list[ResidualBlock]:
    """Single ``reg`` block: ``A = w_reg·I(nv)``, ``c = 0(nv)``."""
    return [ResidualBlock(A=cfg.w_reg * np.eye(nv), c=np.zeros(nv), A_obj=None, name="reg")]
