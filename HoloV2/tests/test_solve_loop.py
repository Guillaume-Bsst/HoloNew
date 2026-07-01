"""solve_frame : un itéré SQP synthétique (tracking quadratique sous box) converge — q -> target,
‖dv‖ -> 0 (arrêt sur step_tol avant n_iter), et le coût final décroît avec le budget d'itérations.
evaluate/assemble sont monkeypatchés ; le backend CVXPY (Plan A) résout réellement chaque QP."""
import types

import numpy as np

from src.solve.backend.cvxpy import CvxpyBackend
from src.solve.contracts import Problem, ResidualBlock, TrustRegion
import src.solve.loop as L

_TARGET = np.array([1.0])


class _StubRobot:
    def __init__(self):
        self.nq = self.nv = 1

    def integrate(self, q, v):
        return np.asarray(q, np.float64) + np.asarray(v, np.float64)


def _patch(monkeypatch):
    # evaluate renvoie q tel quel ("evals" = la config courante) ; assemble construit le QP de tracking.
    monkeypatch.setattr(L, "evaluate", lambda ev, q, poses: q)

    def fake_assemble(evals, ft, geo, robot, cfg, **kw):
        q = np.asarray(evals, np.float64)
        r = ResidualBlock(A=np.eye(1), c=(q - _TARGET), A_obj=None, name="track")
        tr = TrustRegion(var="dv", radius=np.array([0.3]), norm=-1)
        return Problem(nv=1, n_obj=0, residuals=(r,), constraints=(), trust_regions=(tr,))

    monkeypatch.setattr(L, "assemble", fake_assemble)


def _run(n_iter):
    cfg = types.SimpleNamespace(step_tol=1e-6, n_iter_per_frame=n_iter)
    return L.solve_frame(evaluator=None, frame_targets_f=None, geo=None, robot=_StubRobot(),
                         backend=CvxpyBackend(), cfg=cfg, q0=np.array([0.0]),
                         poses0=np.zeros((0, 7)), n_iter=n_iter)


def test_solve_frame_converges_to_target(monkeypatch):
    _patch(monkeypatch)
    q, poses, info = _run(n_iter=50)
    assert abs(q[0] - 1.0) < 1e-3                          # convergence vers la cible (‖dv‖ -> 0)
    assert info.n_iters < 50                               # arrêt anticipé sur step_tol
    assert poses.shape == (0, 7)
    assert "track" in info.cost_by_term


def test_solve_frame_cost_decreases_with_budget(monkeypatch):
    _patch(monkeypatch)
    costs = [_run(n_iter=k)[2].cost for k in (1, 2, 3)]
    assert costs[0] >= costs[1] >= costs[2] - 1e-9         # coût final non croissant avec le budget
    assert costs[2] < costs[0]                             # strictement amélioré


def test_evaluate_converts_object_poses_wxyz():
    # evaluate (op pur) : poses (N,7) -> object_pos (N,3) + object_rot (N,3,3) via wxyz->mat, puis appelle
    # evaluator.style(q) et evaluator.contacts(q, object_rot, object_pos). Vérifié sur des quats connus.
    captured = {}

    class _FakeEval:
        def style(self, q):
            captured["q_style"] = q
            return "STYLE"

        def contacts(self, q, object_rot, object_pos):
            captured["q_contacts"], captured["rot"], captured["pos"] = q, object_rot, object_pos
            return "CONTACT"

    q = np.array([0.1, 0.2, 0.3])
    c = np.cos(np.pi / 4)                                  # obj1 = 90° autour de z : wxyz [c,0,0,c]
    poses = np.array([[1, 2, 3, 1, 0, 0, 0],              # obj0 = identité
                      [4, 5, 6, c, 0, 0, c]], np.float64)
    fe = L.evaluate(_FakeEval(), q, poses)

    assert fe.style == "STYLE" and fe.contact == "CONTACT"
    assert captured["q_style"] is q and captured["q_contacts"] is q
    assert captured["rot"].shape == (2, 3, 3) and captured["pos"].shape == (2, 3)
    assert np.allclose(captured["pos"], [[1, 2, 3], [4, 5, 6]])
    assert np.allclose(captured["rot"][0], np.eye(3))
    assert np.allclose(captured["rot"][1], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-9)
    for R in captured["rot"]:                              # orthonormales
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
