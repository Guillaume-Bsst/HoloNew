"""Couche contacts (roadmap #3) — contact CIBLE vs ATTEINT sur les M points de contrôle robot.

Les M points de correspondance robot portent deux champs de contact alignés (canal-first ``(C, M)``)
sur les MÊMES positions monde ``solved.robot_points_world`` :
  - CIBLE   : ``frame.targets.robot_interaction.field`` (le champ humain transporté sur le robot) ;
  - ATTEINT : ``frame.solved.contact_achieved.field``   (réévalué à la config ``q`` résolue).
On colore les points par le canal sélectionné (heatmap distance / masque actif / uniforme), un nuage
pour chaque champ (toggle indépendant).

Ajout : lignes witness (sonde → point surface le plus proche) pour les probes ACTIVES du canal
sélectionné, dessinées via ``_contact_ops.witness_segments``.
  - Witness CIBLE  : mappé via la pose objet RÉSOLUE (``solved.object_poses[oi]``) lorsque le
    canal est un canal objet, sinon identité (canal sol déjà en monde).
  - Witness ATTEINT : mappé via la pose objet RÉSOLUE (``solved.object_poses[oi]``).

Couche SOLVE-GATED : sans ``solved`` les M points n'ont pas de position monde -> masquée. Pure
consommatrice ; viser confiné ici."""
from __future__ import annotations

import numpy as np

from ..core import colors
from ..core import viser_ops
from ..core.layer import UiState
from ..core.viser_ops import quat_wxyz_to_R
from ..model import VizContext, VizFrame
from ...targets.contracts import MultiChannelField
from ._contact_ops import witness_segments

# Teintes uniformes (uint8 RGB) quand mode == "uniform"
_TARGET_RGB = np.array([255, 170, 0], np.uint8)     # cible : orange
_ACHIEVED_RGB = np.array([0, 200, 120], np.uint8)   # atteint : vert

# Couleurs des lignes witness (uint8 RGB) — mêmes teintes que les nuages
_WIT_TARGET_RGB = np.array([255, 170, 0], np.uint8)    # witness cible : orange
_WIT_ACHIEVED_RGB = np.array([0, 200, 120], np.uint8)  # witness atteint : vert


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
    """Deux nuages de points (cible / atteint) + lignes witness sur ``solved.robot_points_world``,
    colorés par le canal+mode globaux.

    Couche SOLVE-GATED : no-op + masquage si ``frame.solved is None`` (sans ``q`` résolu les
    M points n'ont pas de position monde). Gardes données manquantes : masquage silencieux si
    ``targets``/``robot_interaction`` est None ou si le canal sélectionné est inconnu.

    Witness : segments (sonde_monde → witness_monde) pour les probes actives du canal courant.
    Canal objet → mapping par la pose objet RÉSOLUE (cible et atteint).
    Canal sol (object_idx=None) → witness déjà en monde (passage identité).
    """

    folder = "Contacts (robot)"

    def setup(self, server, gui, ctx: VizContext) -> None:
        """Crée les quatre poignées persistantes (2 nuages + 2 witness) et les checkboxes GUI."""
        self._ctx = ctx
        self._last_frame = None   # dernier VizFrame reçu (re-rendu en pause)
        self._last_ui = None      # dernière UiState reçue (re-rendu en pause)

        def _on_change(_) -> None:
            """Re-rend le frame courant immédiatement quand un toggle change en pause."""
            if self._last_frame is not None and self._last_ui is not None:
                self.update(self._last_frame, self._last_ui)

        with gui.add_folder(self.folder):
            self._cb_target = gui.add_checkbox("contact cible", True)
            self._cb_achieved = gui.add_checkbox("contact atteint", True)
            self._cb_wit_target = gui.add_checkbox("witness cible", False)
            self._cb_wit_achieved = gui.add_checkbox("witness atteint", False)

        self._cb_target.on_update(_on_change)
        self._cb_achieved.on_update(_on_change)
        self._cb_wit_target.on_update(_on_change)
        self._cb_wit_achieved.on_update(_on_change)

        # Poignées nuages de points (cible + atteint)
        zero = np.zeros((1, 3), np.float32)
        self._h_target = server.scene.add_point_cloud(
            "/contacts/target", zero, np.zeros((1, 3), np.uint8), point_size=0.014)
        self._h_achieved = server.scene.add_point_cloud(
            "/contacts/achieved", zero, np.zeros((1, 3), np.uint8), point_size=0.014)

        # Poignées lignes witness (cible + atteint)
        zero_seg = np.zeros((1, 2, 3), np.float32)
        zero_seg_col = np.zeros((1, 2, 3), np.uint8)
        self._h_wit_target = viser_ops.add_line_segments(
            server, "/contacts/witness_target", zero_seg, zero_seg_col, line_width=1.5)
        self._h_wit_achieved = viser_ops.add_line_segments(
            server, "/contacts/witness_achieved", zero_seg, zero_seg_col, line_width=1.5)

    def update(self, frame: VizFrame, ui: UiState) -> None:
        """Rafraîchit les nuages et les lignes witness pour le frame courant.

        Gardes de sortie anticipée (masquage silencieux sans levée) :
          - ``frame.solved is None``                           (couche solve-gated)
          - ``frame.targets`` ou ``robot_interaction`` est None
          - ``ui.channel`` absent de ``ctx.channel_names``
        Le nuage atteint et le witness atteint sont masqués seuls si ``contact_achieved`` est
        None (résolution partielle)."""
        # Mémorise le frame et l'état UI pour permettre le re-rendu en pause (bascule toggle)
        self._last_frame = frame
        self._last_ui = ui

        # --- Garde 1 : solve-gated — pas de position monde sans q résolu ---
        if frame.solved is None:
            self._h_target.visible = False
            self._h_achieved.visible = False
            self._h_wit_target.visible = False
            self._h_wit_achieved.visible = False
            return

        # --- Garde 2 : cible robot manquante ---
        if frame.targets is None or frame.targets.robot_interaction is None:
            self._h_target.visible = False
            self._h_achieved.visible = False
            self._h_wit_target.visible = False
            self._h_wit_achieved.visible = False
            return

        # --- Garde 3 : canal inconnu (UI en transition) ---
        if ui.channel not in self._ctx.channel_names:
            self._h_target.visible = False
            self._h_achieved.visible = False
            self._h_wit_target.visible = False
            self._h_wit_achieved.visible = False
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

        # --- Pose canal-objet RÉSOLUE pour le mapping witness cible et atteint ---
        oi = self._ctx.channels[c].object_idx    # None si canal sol
        if oi is not None:
            pose7 = np.asarray(frame.solved.object_poses[oi], np.float64)  # (7,)
            t_sol = pose7[:3]                                                # (3,)
            R_sol = quat_wxyz_to_R(pose7[3:7][np.newaxis])[0]              # (3, 3)
        else:
            R_sol = np.eye(3, dtype=np.float64)
            t_sol = np.zeros(3, np.float64)

        segs_tgt = witness_segments(pts, tgt.witness[c], tgt.active[c], R_sol, t_sol)
        if bool(self._cb_wit_target.value) and len(segs_tgt):
            self._h_wit_target.points = segs_tgt
            self._h_wit_target.colors = np.tile(
                np.array([[[255, 170, 0]]], np.uint8), (len(segs_tgt), 2, 1))
        self._h_wit_target.visible = bool(self._cb_wit_target.value) and len(segs_tgt) > 0

        # --- Nuage ATTEINT (réévalué à q résolu) ---
        if frame.solved.contact_achieved is None:
            # Résolution partielle : éval contact absente -> masquer atteint + witness atteint
            self._h_achieved.visible = False
            self._h_wit_achieved.visible = False
            return

        ach = frame.solved.contact_achieved.field                      # (C, M)
        self._h_achieved.points = pts
        self._h_achieved.colors = contact_colors(ach, c, ui.color_mode, margin,
                                                  uniform_rgb=_ACHIEVED_RGB)
        self._h_achieved.point_size = sz
        self._h_achieved.visible = bool(self._cb_achieved.value)

        # --- Witness ATTEINT (même pose RÉSOLUE que la cible) ---
        segs_ach = witness_segments(pts, ach.witness[c], ach.active[c], R_sol, t_sol)
        if bool(self._cb_wit_achieved.value) and len(segs_ach):
            self._h_wit_achieved.points = segs_ach
            self._h_wit_achieved.colors = np.tile(
                np.array([[[0, 200, 120]]], np.uint8), (len(segs_ach), 2, 1))
        self._h_wit_achieved.visible = bool(self._cb_wit_achieved.value) and len(segs_ach) > 0
