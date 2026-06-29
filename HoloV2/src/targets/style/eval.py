"""style.eval — FK courant + jacobiennes géométriques des links suivis -> ``StyleEval`` (q-dépendant).

Le pendant ÉVALUATION de ``style.build`` (réf, q-indép.) : ``build`` pose la démo humaine en cible de
style ; ``style_eval`` lit l'état COURANT du robot à ``q``. CONFIG-FREE : le SCALE/OFFSET de la recette
(``targets/config``) ne sert qu'à la référence (où placer la cible), pas à l'éval du robot — ici on
lit la cinématique brute (``RobotModel.link_jacobians``) et on gather les links suivis dans l'ordre de
la recette (= ``StyleTargets.link_names``), pour que ``solve`` aligne cur vs réf canal-par-canal.

``link_jacobians`` rend les transforms monde ET les jacobiennes de frame LOCAL_WORLD_ALIGNED (axes
monde) pour TOUS les links ; on en extrait les links suivis par leur NOM. Pur, float64, torch-free
(la cinématique lourde est cachée dans l'instance ``RobotModel``). Reference-free, cost-free.
"""
from __future__ import annotations

import numpy as np

from ...prepare.contracts import RobotModel
from ..contracts import StyleEval


def style_eval(robot: RobotModel, q: np.ndarray, link_names: tuple[str, ...]) -> StyleEval:
    """État courant (FK) + jacobiennes des links ``link_names`` à ``q`` -> ``StyleEval``.

    ``robot.link_jacobians(q)`` rend ``(rot (L_all,3,3), pos (L_all,3), jac_lin (L_all,3,nv),
    jac_ang (L_all,3,nv))`` en repère monde, aligné sur ``robot.link_names`` ; on gather les links
    suivis (par nom). ``jac_pos`` = jac translationnelle, ``jac_rot`` = jac angulaire géométrique."""
    missing = [n for n in link_names if n not in robot.link_names]
    if missing:
        raise ValueError(f"style links absent from robot.link_names: {missing}")

    rot, pos, jac_lin, jac_ang = robot.link_jacobians(q)        # (L_all, ...) repère monde
    gather = np.array([robot.link_names.index(n) for n in link_names], np.int64)  # (L,) into FK order
    return StyleEval(
        position=np.ascontiguousarray(pos[gather]),            # (L, 3)
        rotation=np.ascontiguousarray(rot[gather]),            # (L, 3, 3)
        jac_pos=np.ascontiguousarray(jac_lin[gather]),         # (L, 3, nv)
        jac_rot=np.ascontiguousarray(jac_ang[gather]),         # (L, 3, nv)
        link_names=tuple(link_names),
    )
