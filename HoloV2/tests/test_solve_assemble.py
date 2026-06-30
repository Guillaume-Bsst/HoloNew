"""assemble : concatène build_style/contact/object/reg + build_constraints en UN Problem bien formé
(nv/n_obj corrects, comptes de blocs). Les builders Plan B sont monkeypatchés -> on teste la logique
d'orchestration d'assemble, isolée des internes Plan B."""
import types

import numpy as np

from src.solve.contracts import FrameEval, LinearConstraint, Problem, ResidualBlock, TrustRegion
import src.solve.assemble as A


class _StubRobot:
    def __init__(self, nv):
        self.nv = nv
        self.nq = nv + 1


def _blocks(nv, names):
    return [ResidualBlock(A=np.zeros((1, nv)), c=np.zeros(1), A_obj=None, name=n) for n in names]


def test_assemble_concatenates_into_well_formed_problem(monkeypatch):
    nv, N = 5, 2
    monkeypatch.setattr(A, "_geo_field", lambda ch, orot, opos: object())   # isolate from GeoField internals
    monkeypatch.setattr(A, "build_style", lambda se, st, cfg: _blocks(nv, ["S-pos", "S-rot"]))
    monkeypatch.setattr(A, "build_contact", lambda ce, rf, geo, cfg: _blocks(nv, ["C-D", "C-X"]))
    monkeypatch.setattr(A, "build_object", lambda ce, ev, orot, opos, cfg: _blocks(nv, ["O"]))
    monkeypatch.setattr(A, "build_reg", lambda nv_, cfg: _blocks(nv, ["reg"]))
    monkeypatch.setattr(A, "build_constraints", lambda robot, cfg: (
        [LinearConstraint(A=np.zeros((1, nv)), lb=np.zeros(1), ub=None, A_obj=None, name="jl")],
        [TrustRegion(var="dv", radius=np.ones(nv), norm=-1)]))

    evals = FrameEval(style=object(), contact=types.SimpleNamespace())
    ft = types.SimpleNamespace(
        style=object(),
        robot_interaction=object(),
        env_interaction=object(),
        object_rot=np.zeros((N, 3, 3)),
        object_pos=np.zeros((N, 3)))
    cfg = types.SimpleNamespace(tr_object_pos=0.05, tr_object_rot=0.10)

    prob = A.assemble(evals, ft, geo=("chan0",), robot=_StubRobot(nv), cfg=cfg)
    assert isinstance(prob, Problem)
    assert prob.nv == nv and prob.n_obj == N
    assert len(prob.residuals) == 6                       # 2 style + 2 contact + 1 object + 1 reg
    assert {b.name for b in prob.residuals} == {"S-pos", "S-rot", "C-D", "C-X", "O", "reg"}
    assert len(prob.constraints) == 1 and len(prob.trust_regions) == 2  # dv (build_constraints) + dxi (assemble)
    dxi_tr = next(tr for tr in prob.trust_regions if tr.var == "dxi")
    assert dxi_tr.radius.shape == (N * 6,)


def test_assemble_no_objects_sets_n_obj_zero(monkeypatch):
    nv = 3
    monkeypatch.setattr(A, "_geo_field", lambda ch, orot, opos: object())   # isolate from GeoField internals
    monkeypatch.setattr(A, "build_style", lambda se, st, cfg: _blocks(nv, ["S-pos"]))
    monkeypatch.setattr(A, "build_contact", lambda ce, rf, geo, cfg: [])
    monkeypatch.setattr(A, "build_object", lambda ce, ev, orot, opos, cfg: [])
    monkeypatch.setattr(A, "build_reg", lambda nv_, cfg: _blocks(nv, ["reg"]))
    monkeypatch.setattr(A, "build_constraints", lambda robot, cfg: ([], []))

    evals = FrameEval(style=object(), contact=object())
    ft = types.SimpleNamespace(style=object(), robot_interaction=object(), env_interaction=object(),
                               object_rot=np.zeros((0, 3, 3)), object_pos=np.zeros((0, 3)))
    prob = A.assemble(evals, ft, geo=(), robot=_StubRobot(nv), cfg=object())
    assert prob.n_obj == 0 and len(prob.residuals) == 2 and len(prob.trust_regions) == 0  # no dxi when n_obj=0
