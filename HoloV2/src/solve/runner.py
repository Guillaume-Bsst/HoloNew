"""runner — PUBLIC ENTRY POINT of the solve stage: ``solve(grounded, ctx, frame_targets, config)`` ->
``SolveTrajectory``. Builds the ``Evaluator`` (targets) 1× and backend (Plan A) 1×, then loops
frames: f=0 -> ``compute_q_init`` (Holosoma seed) + budget ``n_iter_first``; f>0 -> ``warm_start``
(carry from f-1) + budget ``n_iter_per_frame``. ``prof.span`` (sequence) lives here. Remains
pinocchio/torch-free (kinematics hidden in ``ctx.robot``); cvxpy arrives only via backend (lazy).

``robot_name`` keys the style recipe (tracked links) of the ``Evaluator``: read from explicit
argument, else from ``config.robot_name`` (cf. Plan B integration assumption)."""
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
    """Online loop over frames -> ``SolveTrajectory``. ``frame_targets`` = ``list[FrameTargets]``
    (output of ``targets.pipeline``). ``grounded`` is accepted for public seam consistency (sourcing /
    future centroidal targets); loop does not depend on it directly (all passes through
    ``frame_targets`` + ``ctx``)."""
    name = robot_name if robot_name is not None else getattr(config, "robot_name", None)
    if name is None:
        raise ValueError("robot_name required (explicit argument or config.robot_name) for Evaluator")

    evaluator = Evaluator(ctx, name)
    backend = make_backend(config.backend)
    robot = ctx.robot
    geo = ctx.channels                                     # geodesic/SDF context per channel (build_contact)

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
