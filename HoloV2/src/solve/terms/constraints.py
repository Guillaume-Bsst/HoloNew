"""Joint limits + box trust region — the LINEAR/box side of the subproblem (the residuals are the
quadratic side). Joint limits: ``lower ≤ q_joint + δv_joint ≤ upper`` -> a box on the joint DOFs of
``δv`` (the selection ``v[6:6+dof]``). ``build_constraints(robot, cfg)`` has no live ``q``, so v1
linearises at the NEUTRAL joints ``q0 = robot.neutral()[7:]`` (EXACT at the cold-start frame; an
approximate static box afterwards — Plan C should re-base on the live ``q`` each SQP iterate; see plan
Assumption 2). Trust region: per-DOF box radii from the config (heterogeneous units handled per-DOF)."""
from __future__ import annotations

import numpy as np

from ..contracts import LinearConstraint, TrustRegion
from ..config import SolveConfig


def build_constraints(robot, cfg: SolveConfig) -> tuple[list[LinearConstraint], list[TrustRegion]]:
    """Box joint-limit ``LinearConstraint`` on ``δv[6:6+dof]`` (linearised at neutral) + the per-DOF box
    ``TrustRegion`` for ``δv``. The object trust region (``δξ``) is added by Plan C's assemble when
    ``n_obj > 0`` (it needs ``n_obj``, not available here)."""
    nv, dof = robot.nv, robot.dof
    lower, upper = robot.joint_pos_limits()
    q0 = np.asarray(robot.neutral(), np.float64)[7:7 + dof]      # neutral joint angles (Assumption 2)

    S = np.zeros((dof, nv), np.float64)                          # select δv joint DOFs (v[6:6+dof])
    S[np.arange(dof), 6 + np.arange(dof)] = 1.0
    joint_limits = LinearConstraint(A=S, lb=np.asarray(lower, np.float64) - q0,
                                    ub=np.asarray(upper, np.float64) - q0,
                                    A_obj=None, name="joint_limits")

    radius = np.concatenate([np.full(3, cfg.tr_base_pos), np.full(3, cfg.tr_base_rot),
                             np.full(dof, cfg.tr_joints)])       # (nv,) per-DOF box radius
    trust = TrustRegion(var="dv", radius=radius, norm=-1)
    return [joint_limits], [trust]
