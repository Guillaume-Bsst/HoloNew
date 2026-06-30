"""loop — l'itéré SQP/trust-region par frame. Flux LINÉAIRE explicite (le point que V1 ratait) :
``evaluate -> assemble -> backend.solve -> retract -> converge``. Une seule passe, pas de classe-dieu.

``evaluate`` est le wrapper du seam : l'``Evaluator`` (targets) expose ``.style(q)`` + ``.contacts(q,
object_rot, object_pos)`` mais pas un appel combiné — on convertit ``object_poses (N,7) -> (object_rot,
object_pos)`` puis on assemble un ``FrameEval``. ``prof.span`` vit ici (orchestrateur), jamais dans les
ops pures."""
from __future__ import annotations

import numpy as np

from ..obs import NULL
from .assemble import assemble
from .contracts import FrameEval, FrameInfo, Problem, Step
from .retract import quat_wxyz_to_mat, retract


def evaluate(evaluator, q: np.ndarray, object_poses: np.ndarray) -> FrameEval:
    """Géométrie courante au ``(q, object_poses)`` : style FK + champ de contact. Convertit les poses
    objet ``(N,7)`` (pos + quat wxyz) en ``(object_rot (N,3,3), object_pos (N,3))`` attendus par
    ``Evaluator.contacts``."""
    poses = np.asarray(object_poses, np.float64)
    n = poses.shape[0]
    object_rot = np.empty((n, 3, 3))
    object_pos = np.empty((n, 3))
    for i in range(n):
        object_pos[i] = poses[i, :3]
        object_rot[i] = quat_wxyz_to_mat(poses[i, 3:7])
    return FrameEval(style=evaluator.style(q),
                     contact=evaluator.contacts(q, object_rot, object_pos))


def cost_breakdown(problem: Problem, step: Step) -> dict[str, float]:
    """‖A·dv + A_obj·dξ + c‖² par NOM de terme au pas résolu (l'outil n°1 de tuning des poids).
    Les blocs de même nom s'agrègent."""
    dv = np.asarray(step.dv, np.float64)
    dxi = None if step.dxi is None else np.asarray(step.dxi, np.float64).reshape(-1)
    out: dict[str, float] = {}
    for b in problem.residuals:
        e = b.A @ dv + b.c
        if b.A_obj is not None and dxi is not None:
            e = e + b.A_obj @ dxi
        out[b.name] = out.get(b.name, 0.0) + float(e @ e)
    return out


def solve_frame(evaluator, frame_targets_f, geo, robot, backend, cfg, q0, poses0,
                n_iter: int | None = None, prof=NULL) -> tuple[np.ndarray, np.ndarray, FrameInfo]:
    """Un itéré SQP sur UNE frame depuis le seed ``(q0, poses0)``. Trust-region FIXE (adaptatif =
    incrément futur). Convergence : ``max|dv| < cfg.step_tol`` ou ``n_iter`` atteint (par défaut
    ``cfg.n_iter_per_frame`` ; le runner passe ``cfg.n_iter_first`` au cold start f=0)."""
    max_iter = cfg.n_iter_per_frame if n_iter is None else n_iter
    q = np.array(q0, np.float64, copy=True)
    poses = np.array(poses0, np.float64, copy=True)
    status, cost, cost_by_term, it = "no_iter", float("nan"), {}, 0
    with prof.span("frame_solve"):
        for it in range(1, max_iter + 1):
            evals = evaluate(evaluator, q, poses)
            problem = assemble(evals, frame_targets_f, geo, robot, cfg)
            step = backend.solve(problem)
            status, cost = step.status, step.value
            if step.dv is None or not np.all(np.isfinite(step.dv)):
                break                                       # backend non-optimal -> arrêt (avant cost_breakdown,
                                                            # qui ferait np.asarray(None) sur un dv manquant)
            cost_by_term = cost_breakdown(problem, step)    # seulement sur un pas valide
            q, poses = retract(q, poses, step, robot)
            if float(np.max(np.abs(step.dv))) < cfg.step_tol:
                break
    return q, poses, FrameInfo(n_iters=it, status=status, cost=cost, cost_by_term=cost_by_term)
