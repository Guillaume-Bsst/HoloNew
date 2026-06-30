"""build_reg : damping de pas A = w_reg·I, c = 0."""
import numpy as np

from src.solve.config import SolveConfig
from src.solve.terms.reg import build_reg


def test_reg_block():
    nv = 9
    cfg = SolveConfig(w_reg=0.05)
    blocks = build_reg(nv, cfg)
    assert len(blocks) == 1
    b = blocks[0]
    assert b.name == "reg" and b.A_obj is None
    assert b.A.shape == (nv, nv) and b.c.shape == (nv,)
    assert np.allclose(b.A, 0.05 * np.eye(nv)) and np.allclose(b.c, 0.0)
