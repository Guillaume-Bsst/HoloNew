"""runner — ENTRÉE PUBLIQUE de l'étage solve : ``solve(grounded, ctx, frame_targets, config)`` ->
``SolveTrajectory``. Construit l'``Evaluator`` (targets) 1× et le backend (Plan A) 1×, puis boucle les
frames : f=0 -> ``compute_q_init`` (seed Holosoma) + budget ``n_iter_first`` ; f>0 -> ``warm_start``
(carry de f-1) + budget ``n_iter_per_frame``. ``prof.span`` (séquence) vit ici. Reste pinocchio/torch-free
(cinématique cachée dans ``ctx.robot``) ; cvxpy n'arrive que via le backend (lazy).

``robot_name`` clé la recette de style (links suivis) de l'``Evaluator`` : il est lu de l'argument
explicite, sinon de ``config.robot_name`` (cf. assomption d'intégration Plan B)."""
from __future__ import annotations

import numpy as np

from ..obs import NULL
from ..targets import Evaluator
from .backend import make_backend
from .config import SolveConfig
from .contracts import SolveTrajectory
from .init import compute_q_init, warm_start
from .loop import solve_frame


def solve(grounded, ctx, frame_targets, config: SolveConfig, *, robot_name: str | None = None,
          prof=NULL) -> SolveTrajectory:
    """Boucle online sur les frames -> ``SolveTrajectory``. ``frame_targets`` = ``list[FrameTargets]``
    (sortie ``targets.pipeline``). ``grounded`` est accepté pour la cohérence du seam public (provenance /
    futurs targets centroïdaux) ; la boucle n'en dépend pas directement (tout passe par
    ``frame_targets`` + ``ctx``)."""
    name = robot_name if robot_name is not None else getattr(config, "robot_name", None)
    if name is None:
        raise ValueError("robot_name requis (argument explicite ou config.robot_name) pour l'Evaluator")

    evaluator = Evaluator(ctx, name)
    backend = make_backend(config.backend)
    robot = ctx.robot
    geo = ctx.channels                                     # contexte géodésique/SDF par canal (build_contact)

    q = None
    poses = None
    qpos_rows: list[np.ndarray] = []
    poses_rows: list[np.ndarray] = []
    info_rows = []
    with prof.span("sequence", T=len(frame_targets)):
        for f, ft in enumerate(frame_targets):
            if f == 0:
                q, poses = compute_q_init(ft, robot)
                n_iter = config.n_iter_first
            else:
                q, poses = warm_start(q, poses)
                n_iter = config.n_iter_per_frame
            q, poses, fi = solve_frame(evaluator, ft, geo, robot, backend, config, q, poses,
                                       n_iter=n_iter, prof=prof)
            qpos_rows.append(q)
            poses_rows.append(poses)
            info_rows.append(fi)

    T = len(frame_targets)
    qpos = np.asarray(qpos_rows, np.float64) if qpos_rows else np.zeros((0, robot.nq))
    n_obj = poses_rows[0].shape[0] if poses_rows else 0
    object_poses = (np.asarray(poses_rows, np.float64) if n_obj
                    else np.zeros((T, 0, 7), np.float64))
    return SolveTrajectory(qpos=qpos, object_poses=object_poses, info=tuple(info_rows))
