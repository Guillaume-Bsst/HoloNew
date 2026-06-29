"""evaluator — orchestrateur q-DÉPENDANT du seam ``targets -> solve`` (objet ``Evaluator``).

Réunit les deux concerns (qui ne se connaissent pas) derrière l'API solveur : ``style_eval`` (FK des
links suivis) et ``contact_eval`` (champ + jacobiennes de contact). Construit UNE fois (séquence-wide)
à partir des assets STATIQUES de l'``InteractionContext`` (``robot_cloud``, ``channels``,
``object_clouds``, ``margin``, ``robot``) ; les seules entrées par itération sont les variables de
décision ``(q, object_poses)``. Les références par frame (``list[FrameTargets]``) sont une sortie
PARALLÈLE (``pipeline``), pas une entrée de l'évaluateur.

Les links suivis sont dérivés de la recette de style robot-keyée — exactement la clé qu'utilise
``style/build.py`` (``config.style_table(robot.name).keys()``) ; ``ctx.robot`` étant un ``RobotModel``
sans nom, le nom du robot est passé à la construction. Pur, torch-free (cinématique cachée dans
``ctx.robot``).
"""
from __future__ import annotations

import numpy as np

from ..prepare.contracts import InteractionContext
from .config import style_table
from .contracts import ContactEval, StyleEval
from .interaction import contact_eval
from .style import style_eval


class Evaluator:
    """Évaluateur q-dépendant construit une fois à partir de ``ctx``. ``robot_name`` clé la recette de
    style (links suivis). Expose ``.style(q)`` et ``.contacts(q, object_rot, object_pos)``."""

    def __init__(self, ctx: InteractionContext, robot_name: str) -> None:
        self._ctx = ctx
        self._robot = ctx.robot
        self._style_links: tuple[str, ...] = tuple(style_table(robot_name).keys())  # = StyleTargets order

    def style(self, q: np.ndarray) -> StyleEval:
        """État courant (FK) + jacobiennes des links suivis à ``q``."""
        return style_eval(self._robot, q, self._style_links)

    def contacts(self, q: np.ndarray, object_rot: np.ndarray, object_pos: np.ndarray) -> ContactEval:
        """Champ de contact courant (robot) + jacobiennes pour ``(q, object_poses)``."""
        return contact_eval(self._ctx, q, object_rot, object_pos)
