"""loop — the SQP/trust-region iterate per frame. Explicit LINEAR flow (the point V1 missed):
``evaluate -> assemble -> backend.solve -> retract -> converge``. Single pass, no god class.

``evaluate`` is the seam wrapper: the ``Evaluator`` (targets) exposes ``.style(q)`` + ``.contacts(q,
object_rot, object_pos)`` but not a combined call — we convert ``object_poses (N,7) -> (object_rot,
object_pos)`` then assemble a ``FrameEval``. ``prof.span`` lives here (orchestrator), never in pure
ops."""
from __future__ import annotations

import numpy as np

from ..obs import NULL
from .assemble import assemble
from .contracts import FrameEval, FrameInfo, Problem, Step
from .retract import quat_wxyz_to_mat, retract


def evaluate(evaluator, q: np.ndarray, object_poses: np.ndarray) -> FrameEval:
    """Current geometry at ``(q, object_poses)``: style FK + contact field. Converts object poses
    ``(N,7)`` (pos + quat wxyz) to ``(object_rot (N,3,3), object_pos (N,3))`` expected by
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
    """‖A·dv + A_obj·dξ + c‖² by TERM NAME at solved step (tool #1 for tuning weights).
    Blocks with same name aggregate."""
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
    """One SQP iterate on ONE frame from seed ``(q0, poses0)``. FIXED trust-region (adaptive =
    future increment). Convergence: ``max|dv| < cfg.step_tol`` or ``n_iter`` reached (default
    ``cfg.n_iter_per_frame``; runner passes ``cfg.n_iter_first`` at cold start f=0)."""
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
                break                                       # backend non-optimal -> stop (before cost_breakdown,
                                                            # which would np.asarray(None) on missing dv)
            cost_by_term = cost_breakdown(problem, step)    # seulement sur un pas valide
            q, poses = retract(q, poses, step, robot)
            if float(np.max(np.abs(step.dv))) < cfg.step_tol:
                break
    return q, poses, FrameInfo(n_iters=it, status=status, cost=cost, cost_by_term=cost_by_term)
