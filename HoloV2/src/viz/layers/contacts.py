"""Couche contacts (roadmap #3) — contact CIBLE vs ATTEINT sur les M points de contrôle robot.

Les M points de correspondance robot portent deux champs de contact alignés (canal-first ``(C, M)``)
sur les MÊMES positions monde ``solved.robot_points_world`` :
  - CIBLE   : ``frame.targets.robot_interaction.field`` (le champ humain transporté sur le robot) ;
  - ATTEINT : ``frame.solved.contact_achieved.field``   (réévalué à la config ``q`` résolue).
On colore les points par le canal sélectionné (heatmap distance / masque actif / uniforme), un nuage
pour chaque champ (toggle indépendant). Couche SOLVE-GATED : sans ``solved`` les M points n'ont pas de
position monde -> masquée. Pure consommatrice ; viser confiné ici."""
from __future__ import annotations

import numpy as np

from ..core import colors
from ..core.layer import UiState
from ..model import VizContext, VizFrame
from ...targets.contracts import MultiChannelField

# Teintes uniformes (uint8 RGB) quand mode == "uniform"
_TARGET_RGB = np.array([255, 170, 0], np.uint8)     # cible : orange
_ACHIEVED_RGB = np.array([0, 200, 120], np.uint8)   # atteint : vert


def contact_colors(field: MultiChannelField, channel_idx: int, mode: str, margin: float,
                   *, uniform_rgb: np.ndarray = _TARGET_RGB) -> np.ndarray:
    """(M, 3) uint8 : colore les M points par le canal ``channel_idx`` de ``field``.

    ``mode`` : 'distance' (heatmap signée bleu→rouge), 'active' (masque booléen vert/gris),
    sinon 'uniform' (couleur ``uniform_rgb`` uniforme)."""
    if mode == "distance":
        return colors.heat_distance(field.distance[channel_idx], margin)
    if mode == "active":
        return colors.active_mask(field.active[channel_idx])
    M = field.distance.shape[1]
    return np.tile(np.asarray(uniform_rgb, np.uint8), (M, 1))


class ContactsLayer:
    """Deux nuages de points (cible / atteint) sur ``solved.robot_points_world``, colorés par le
    canal+mode globaux.

    Couche SOLVE-GATED : no-op + masquage si ``frame.solved is None`` (sans ``q`` résolu les
    M points n'ont pas de position monde). Gardes données manquantes : masquage silencieux si
    ``targets``/``robot_interaction`` est None ou si le canal sélectionné est inconnu."""

    folder = "Contacts (robot)"

    def setup(self, server, gui, ctx: VizContext) -> None:
        """Crée les deux poignées de nuage persistantes et les checkboxes GUI."""
        self._ctx = ctx
        with gui.add_folder(self.folder):
            self._cb_target = gui.add_checkbox("contact cible", True)
            self._cb_achieved = gui.add_checkbox("contact atteint", True)
        zero = np.zeros((1, 3), np.float32)
        self._h_target = server.scene.add_point_cloud(
            "/contacts/target", zero, np.zeros((1, 3), np.uint8), point_size=0.014)
        self._h_achieved = server.scene.add_point_cloud(
            "/contacts/achieved", zero, np.zeros((1, 3), np.uint8), point_size=0.014)

    def update(self, frame: VizFrame, ui: UiState) -> None:
        """Rafraîchit les deux nuages de contact pour le frame courant.

        Gardes de sortie anticipée (masquage silencieux sans levée) :
          - ``frame.solved is None``                           (couche solve-gated)
          - ``frame.targets`` ou ``robot_interaction`` est None
          - ``ui.channel`` absent de ``ctx.channel_names``
        Le nuage atteint est masqué seul si ``contact_achieved`` est None (résolution partielle)."""
        # --- Garde 1 : solve-gated — pas de position monde sans q résolu ---
        if frame.solved is None:
            self._h_target.visible = False
            self._h_achieved.visible = False
            return

        # --- Garde 2 : cible robot manquante ---
        if frame.targets is None or frame.targets.robot_interaction is None:
            self._h_target.visible = False
            self._h_achieved.visible = False
            return

        # --- Garde 3 : canal inconnu (UI en transition) ---
        if ui.channel not in self._ctx.channel_names:
            self._h_target.visible = False
            self._h_achieved.visible = False
            return

        c = self._ctx.channel_names.index(ui.channel)
        pts = np.asarray(frame.solved.robot_points_world, np.float32)   # (M, 3) monde
        margin = float(self._ctx.margin)
        sz = float(ui.point_size)

        # --- Nuage CIBLE (champ transporté humain→robot) ---
        tgt = frame.targets.robot_interaction.field                    # (C, M)
        self._h_target.points = pts
        self._h_target.colors = contact_colors(tgt, c, ui.color_mode, margin,
                                               uniform_rgb=_TARGET_RGB)
        self._h_target.point_size = sz
        self._h_target.visible = bool(self._cb_target.value)

        # --- Nuage ATTEINT (réévalué à q résolu) ---
        if frame.solved.contact_achieved is None:
            # Résolution partielle : éval contact absente -> masquer atteint uniquement
            self._h_achieved.visible = False
            return

        ach = frame.solved.contact_achieved.field                      # (C, M)
        self._h_achieved.points = pts
        self._h_achieved.colors = contact_colors(ach, c, ui.color_mode, margin,
                                                  uniform_rgb=_ACHIEVED_RGB)
        self._h_achieved.point_size = sz
        self._h_achieved.visible = bool(self._cb_achieved.value)
