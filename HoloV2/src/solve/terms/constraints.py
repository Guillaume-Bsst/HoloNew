"""Limites articulaires + région de confiance box — le côté LINEAR/box du sous-problème (les résiduels sont le
côté quadratique). Limites articulaires : ``lower ≤ q_joint + δv_joint ≤ upper`` -> une boîte sur les DOFs
articulaires de ``δv`` (la sélection ``v[6:6+dof]``). ``build_constraints(robot, cfg)`` n'a pas de ``q``
en direct, donc v1 linéarise aux articulations NEUTRES ``q0 = robot.neutral()[7:]`` (EXACT au démarrage
à froid ; une boîte statique approximative après — Plan C devrait se rebaser sur le ``q`` direct à chaque
itération SQP ; voir Assumption 2 du plan). Région de confiance : rayons de boîte par-DOF depuis la config
(unités hétérogènes gérées par-DOF)."""
from __future__ import annotations

import numpy as np

from ..contracts import LinearConstraint, TrustRegion
from ..config import SolveConfig


def build_constraints(robot, cfg: SolveConfig) -> tuple[list[LinearConstraint], list[TrustRegion]]:
    """Contrainte de limite articulaire box ``LinearConstraint`` sur ``δv[6:6+dof]`` (linéarisée au neutre)
    + la région de confiance box par-DOF ``TrustRegion`` pour ``δv``. La région de confiance d'objet (``δξ``)
    est ajoutée par l'assemble du Plan C quand ``n_obj > 0`` (elle a besoin de ``n_obj``, pas disponible ici)."""
    nv, dof = robot.nv, robot.dof
    lower, upper = robot.joint_pos_limits()
    q0 = np.asarray(robot.neutral(), np.float64)[7:7 + dof]      # angles articulaires neutres (Assumption 2)

    S = np.zeros((dof, nv), np.float64)                          # sélectionner les DOFs articulaires δv (v[6:6+dof])
    S[np.arange(dof), 6 + np.arange(dof)] = 1.0
    joint_limits = LinearConstraint(A=S, lb=np.asarray(lower, np.float64) - q0,
                                    ub=np.asarray(upper, np.float64) - q0,
                                    A_obj=None, name="joint_limits")

    radius = np.concatenate([np.full(3, cfg.tr_base_pos), np.full(3, cfg.tr_base_rot),
                             np.full(dof, cfg.tr_joints)])       # (nv,) rayon de boîte par-DOF
    trust = TrustRegion(var="dv", radius=radius, norm=-1)
    return [joint_limits], [trust]
