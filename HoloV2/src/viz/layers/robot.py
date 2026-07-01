"""Couche ``robot`` — le robot G1 résolu rendu via ViserUrdf (meshes complets), plan de route #1.
Lit ``frame.solved.q`` (config free-flyer pinocchio). Sans-opération/masquée quand
``frame.solved is None`` (pré-solve / solve désactivé). Logique portée de V1
``viewer._add_robot`` / ``draw_q`` ; viser/ViserUrdf sont confinés à cette couche
(consommateur, règle d'or 6).

Attention quaternion (piège #1) : q[3:7] est **xyzw** (convention pinocchio) ; viser
attend **wxyz** sur le frame de base ⇒ réordonnance ``q[[6, 3, 4, 5]]``."""
from __future__ import annotations

import numpy as np

from ..core.layer import UiState
from ..model import VizContext, VizFrame

_FOLDER = "Robot (solved)"
_ROOT = "/world/robot_solved"


class RobotLayer:
    """Couche ``Layer`` pour le robot résolu. ``setup`` charge l'URDF une fois (ViserUrdf) + un
    toggle de visibilité ; ``update`` pilote ``update_cfg`` (joints) et la pose de la base
    (pinocchio xyzw → viser wxyz). No-op / masquée quand ``frame.solved is None``."""

    folder = _FOLDER

    def __init__(self) -> None:
        self._urdf = None      # handle ViserUrdf
        self._base = None      # frame de base (scène viser)
        self._dof = 0
        self._toggle = None    # checkbox GUI

    def setup(self, server, gui, ctx: VizContext) -> None:
        """Charge l'URDF G1 avec ViserUrdf (meshes), crée le frame de base dans la scène et
        ajoute le toggle de visibilité dans le dossier GUI. Imports lourds confinés ici
        (viser/yourdfpy inutiles hors serveur)."""
        import yourdfpy
        from viser.extras import ViserUrdf

        self._base = server.scene.add_frame(_ROOT, show_axes=False)
        urdf = yourdfpy.URDF.load(str(ctx.robot_urdf_path), load_meshes=True,
                                  build_scene_graph=True)
        self._urdf = ViserUrdf(server, urdf_or_path=urdf, root_node_name=_ROOT)
        self._dof = len(self._urdf.get_actuated_joint_limits())
        self._urdf.update_cfg(np.zeros(self._dof))
        with gui.add_folder(_FOLDER):
            self._toggle = gui.add_checkbox("Show solved G1", bool(ctx.has_solve))

    def update(self, frame: VizFrame, ui: UiState) -> None:
        """Rafraîchit la pose du robot depuis le frame résolu courant.

        Quand ``frame.solved is None`` (pas encore résolu, ou solve désactivé), la couche
        masque le robot et retourne immédiatement. Quand présent, la visibilité suit le
        toggle GUI — pattern explicitement ré-appliqué ici (et pas seulement dans le callback
        on_update) pour que le robot réapparaisse dès que les données reviennent.
        """
        solved = frame.solved
        # Masquage explicite : le robot ne peut pas être positionné sans q résolu
        show = bool(self._toggle.value) and solved is not None
        self._urdf.show_visual = show
        self._base.visible = show
        if not show:
            return
        q = np.asarray(solved.q, np.float64)
        # Joints actionnés : indices [7 : 7+dof]
        self._urdf.update_cfg(q[7:7 + self._dof])
        # Position de la base (free-flyer translation)
        self._base.position = q[:3]
        # Réordonnance quaternion : pinocchio xyzw (q[3:7]) → viser wxyz
        # q[6]=qw, q[3]=qx, q[4]=qy, q[5]=qz
        self._base.wxyz = q[[6, 3, 4, 5]]
