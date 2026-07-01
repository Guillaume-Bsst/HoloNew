"""stack_cost_terms : empilement pur du cost_by_term sur la séquence (union des termes, frames None
-> ligne nulle). Pas de matplotlib/viser ici (logique pure du panel)."""
import numpy as np

from src.viz.panels.cost_dashboard import stack_cost_terms


class _Solved:                                          # double minimal d'un SolvedFrame
    def __init__(self, cost, by, status, n):
        self.cost, self.cost_by_term, self.status, self.n_iters = cost, by, status, n


class _Frame:
    def __init__(self, solved):
        self.solved = solved


def test_stack_cost_terms_union_and_none():
    frames = [
        _Frame(_Solved(3.0, {"S-pos": 1.0, "C-D": 2.0}, "optimal", 4)),
        _Frame(None),                                   # frame non résolue -> ligne nulle
        _Frame(_Solved(5.0, {"C-D": 5.0}, "max_iter", 10)),
    ]
    names, mat, total, status, n_iters = stack_cost_terms(frames)
    assert set(names) == {"S-pos", "C-D"}               # union des termes
    assert mat.shape == (3, len(names))
    j_cd = names.index("C-D")
    assert mat[0, j_cd] == 2.0 and mat[1, j_cd] == 0.0 and mat[2, j_cd] == 5.0
    assert np.array_equal(total, np.array([3.0, 0.0, 5.0]))
    assert status[0] == "optimal" and status[1] == "" and status[2] == "max_iter"
    assert n_iters[2] == 10 and n_iters[1] == 0


def test_stack_cost_terms_all_none():
    names, mat, total, status, n_iters = stack_cost_terms([_Frame(None), _Frame(None)])
    assert names == () and mat.shape == (2, 0)
    assert np.array_equal(total, np.zeros(2)) and status == ("", "")
