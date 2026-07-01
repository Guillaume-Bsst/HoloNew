"""Couche contacts objets (roadmap — composant 2) — contact CIBLE vs ATTEINT sur les nuages objets.

Pour chaque objet ``k`` (0..n_objects-1), on affiche les probes du nuage objet source ET résolu,
colorés par le canal sélectionné, plus les lignes witness correspondantes.

Convention « en ref objet » (source vs résolu) :
  - **CIBLE**   : nuage objet SOURCE (``frame.object_clouds_world[k]``), coloré par
    ``frame.targets.env_interaction.per_object[k]`` sur le canal sélectionné.
  - **ATTEINT** : nuage objet RÉSOLU (``object_cloud_solved(k)``), coloré par
    ``frame.solved.contact_achieved.env[k].field`` sur le canal sélectionné.

Les lignes witness (sonde → point de surface le plus proche) sont mappées via la pose de
l'objet DU CANAL c (pas de l'objet k) :
  - Witness CIBLE  : pose SOURCE du canal-objet (``frame.pose.object_rot/pos[oi]``).
  - Witness ATTEINT : pose RÉSOLUE du canal-objet (``frame.solved.object_poses[oi]``).
  - Canal sol (``object_idx=None``) : identité (witness déjà en monde).

Couche SOLVE-GATED : sans ``solved``, tous les handles sont masqués.
Consommatrice pure ; viser confiné ici."""
from __future__ import annotations

import numpy as np

from ..core import colors
from ..core import viser_ops
from ..core.layer import UiState
from ..core.viser_ops import quat_wxyz_to_R
from ..model import VizContext, VizFrame
from ...targets.contracts import MultiChannelField
from ._contact_ops import object_cloud_solved, witness_segments

# Teintes uniformes (uint8 RGB) pour les deux états
_TARGET_RGB = np.array([255, 170, 0], np.uint8)    # cible : orange
_ACHIEVED_RGB = np.array([0, 200, 120], np.uint8)  # atteint : vert


def _obj_contact_colors(
    field: MultiChannelField,
    channel_idx: int,
    mode: str,
    margin: float,
    *,
    uniform_rgb: np.ndarray,
) -> np.ndarray:
    """(P, 3) uint8 : colore les P points du nuage objet par le canal ``channel_idx`` du champ.

    ``mode`` : 'distance' (heatmap signée bleu→rouge), 'active' (masque booléen),
    sinon 'uniform' (couleur ``uniform_rgb`` uniforme)."""
    if mode == "distance":
        return colors.heat_distance(field.distance[channel_idx], margin)
    if mode == "active":
        return colors.active_mask(field.active[channel_idx])
    P = field.distance.shape[1]
    return np.tile(np.asarray(uniform_rgb, np.uint8), (P, 1))


class ObjectContactsLayer:
    """Contact cible ET atteint (nuages + lignes witness) pour chaque objet k, « en ref objet ».

    Pour chaque objet k, quatre handles sont créés :
      - nuage CIBLE   : probes du cloud source, colorés distance/active/uniform sur le canal ui.
      - nuage ATTEINT : probes du cloud résolu (transfo relative T_résolu ∘ T_source⁻¹).
      - witness CIBLE   : segments (probe_source → witness_monde) via pose SOURCE du canal.
      - witness ATTEINT : segments (probe_résolu → witness_monde) via pose RÉSOLUE du canal.

    Les quatre checkboxes GUI contrôlent la visibilité de tous les handles d'un même type.

    Couche SOLVE-GATED : no-op + masquage si ``frame.solved is None``. Gardes données manquantes :
    masquage silencieux si ``targets``/``env_interaction`` est None, canal inconnu, ou nb objets
    incohérent entre frame et handles.
    """

    folder = "Contacts objets"

    def setup(self, server, gui, ctx: VizContext) -> None:
        """Crée les handles (4 × n_objects) et les checkboxes GUI."""
        self._ctx = ctx
        self._last_frame = None   # dernier VizFrame reçu (re-rendu en pause)
        self._last_ui = None      # dernière UiState reçue (re-rendu en pause)

        def _on_change(_) -> None:
            """Re-rend le frame courant immédiatement quand un toggle change en pause."""
            if self._last_frame is not None and self._last_ui is not None:
                self.update(self._last_frame, self._last_ui)

        with gui.add_folder(self.folder):
            self._cb_target = gui.add_checkbox("cloud cible", True)
            self._cb_achieved = gui.add_checkbox("cloud atteint", True)
            self._cb_wit_target = gui.add_checkbox("witness cible", False)
            self._cb_wit_achieved = gui.add_checkbox("witness atteint", False)

        # Câblage toggle-en-pause : ré-invocation update() lors d'un clic en pause
        self._cb_target.on_update(_on_change)
        self._cb_achieved.on_update(_on_change)
        self._cb_wit_target.on_update(_on_change)
        self._cb_wit_achieved.on_update(_on_change)

        # Poignées par objet k (listes parallèles)
        n = ctx.n_objects
        zero = np.zeros((1, 3), np.float32)
        zero_seg = np.zeros((1, 2, 3), np.float32)
        zero_seg_col = np.zeros((1, 2, 3), np.uint8)

        self._h_targets: list = []         # nuages CIBLE
        self._h_achieved: list = []        # nuages ATTEINT
        self._h_wit_targets: list = []     # segments witness CIBLE
        self._h_wit_achieved: list = []    # segments witness ATTEINT

        for k in range(n):
            self._h_targets.append(
                server.scene.add_point_cloud(
                    f"/object_contacts/k{k}/target",
                    zero, np.zeros((1, 3), np.uint8), point_size=0.014,
                )
            )
            self._h_achieved.append(
                server.scene.add_point_cloud(
                    f"/object_contacts/k{k}/achieved",
                    zero, np.zeros((1, 3), np.uint8), point_size=0.014,
                )
            )
            self._h_wit_targets.append(
                viser_ops.add_line_segments(
                    server, f"/object_contacts/k{k}/witness_target",
                    zero_seg, zero_seg_col, line_width=1.5,
                )
            )
            self._h_wit_achieved.append(
                viser_ops.add_line_segments(
                    server, f"/object_contacts/k{k}/witness_achieved",
                    zero_seg, zero_seg_col, line_width=1.5,
                )
            )

    def _hide_all(self) -> None:
        """Cache tous les handles (toutes les couches de tous les objets)."""
        for k in range(len(self._h_targets)):
            self._h_targets[k].visible = False
            self._h_achieved[k].visible = False
            self._h_wit_targets[k].visible = False
            self._h_wit_achieved[k].visible = False

    def update(self, frame: VizFrame, ui: UiState) -> None:
        """Rafraîchit les nuages et witness pour le frame courant.

        Gardes de sortie anticipée (masquage silencieux sans levée) :
          - ``frame.solved is None``                            (couche solve-gated)
          - ``frame.targets`` ou ``env_interaction`` est None
          - ``frame.pose`` est None
          - ``ui.channel`` absent de ``ctx.channel_names``
        Le nuage atteint et le witness atteint sont masqués seuls si
        ``contact_achieved`` est None ou si le nombre d'objets est incohérent."""
        # Mémorise le frame et l'état UI pour permettre le re-rendu en pause (bascule toggle)
        self._last_frame = frame
        self._last_ui = ui

        # --- Garde 1 : solve-gated — sans q résolu pas de pose objet résolue ---
        if frame.solved is None:
            self._hide_all()
            return

        # --- Garde 2 : cibles objets manquantes ---
        if frame.targets is None or frame.targets.env_interaction is None:
            self._hide_all()
            return

        # --- Garde 3 : pose source manquante ---
        if frame.pose is None:
            self._hide_all()
            return

        # --- Garde 4 : canal inconnu (UI en transition) ---
        if ui.channel not in self._ctx.channel_names:
            self._hide_all()
            return

        c = self._ctx.channel_names.index(ui.channel)
        margin = float(self._ctx.margin)
        sz = float(ui.point_size)

        # Pose canal-objet SOURCE pour le mapping witness cible
        oi = self._ctx.channels[c].object_idx    # None si canal sol
        if oi is not None:
            R_src_ch = np.asarray(frame.pose.object_rot[oi], np.float64)   # (3, 3)
            t_src_ch = np.asarray(frame.pose.object_pos[oi], np.float64)   # (3,)
        else:
            R_src_ch = np.eye(3, dtype=np.float64)
            t_src_ch = np.zeros(3, np.float64)

        # Pose canal-objet RÉSOLUE pour le mapping witness atteint
        if oi is not None:
            pose7_ch = np.asarray(frame.solved.object_poses[oi], np.float64)  # (7,)
            t_sol_ch = pose7_ch[:3]                                            # (3,)
            R_sol_ch = quat_wxyz_to_R(pose7_ch[3:7][np.newaxis])[0]          # (3, 3)
        else:
            R_sol_ch = np.eye(3, dtype=np.float64)
            t_sol_ch = np.zeros(3, np.float64)

        n_obj = len(self._h_targets)   # nb d'objets connus à setup

        for k in range(n_obj):
            # --- Garde k : objet k hors de la plage des données frame ---
            if k >= len(frame.object_clouds_world):
                self._h_targets[k].visible = False
                self._h_achieved[k].visible = False
                self._h_wit_targets[k].visible = False
                self._h_wit_achieved[k].visible = False
                continue

            per_obj = frame.targets.env_interaction.per_object
            if k >= len(per_obj):
                self._h_targets[k].visible = False
                self._h_achieved[k].visible = False
                self._h_wit_targets[k].visible = False
                self._h_wit_achieved[k].visible = False
                continue

            # Cloud source de l'objet k
            cloud_src = np.asarray(frame.object_clouds_world[k], np.float32)  # (P, 3)

            # Pose source de l'objet k (pour object_cloud_solved)
            R_src_k = np.asarray(frame.pose.object_rot[k], np.float64)   # (3, 3)
            t_src_k = np.asarray(frame.pose.object_pos[k], np.float64)   # (3,)

            # --- Nuage CIBLE (cloud source coloré par le champ cible) ---
            tgt_field = per_obj[k]                                         # MultiChannelField
            self._h_targets[k].points = cloud_src
            self._h_targets[k].colors = _obj_contact_colors(
                tgt_field, c, ui.color_mode, margin, uniform_rgb=_TARGET_RGB,
            )
            self._h_targets[k].point_size = sz
            self._h_targets[k].visible = bool(self._cb_target.value)

            # --- Witness CIBLE (probe source → witness monde via pose canal SOURCE) ---
            segs_tgt = witness_segments(
                cloud_src, tgt_field.witness[c], tgt_field.active[c], R_src_ch, t_src_ch,
            )
            if bool(self._cb_wit_target.value) and len(segs_tgt):
                self._h_wit_targets[k].points = segs_tgt
                self._h_wit_targets[k].colors = np.tile(
                    np.array([[[255, 170, 0]]], np.uint8), (len(segs_tgt), 2, 1),
                )
            self._h_wit_targets[k].visible = (
                bool(self._cb_wit_target.value) and len(segs_tgt) > 0
            )

            # --- Nuage ATTEINT + witness atteint (masqués si contact_achieved absent) ---
            if (frame.solved.contact_achieved is None
                    or k >= len(frame.solved.contact_achieved.env)):
                # Résolution partielle ou nb objets incohérent : masquer atteint
                self._h_achieved[k].visible = False
                self._h_wit_achieved[k].visible = False
                continue

            # Pose résolue de l'objet k (pour object_cloud_solved)
            pose7_k = np.asarray(frame.solved.object_poses[k], np.float64)  # (7,)
            t_sol_k = pose7_k[:3]                                             # (3,)
            R_sol_k = quat_wxyz_to_R(pose7_k[3:7][np.newaxis])[0]           # (3, 3)

            # Cloud résolu = transfo relative T_résolu ∘ T_source⁻¹ appliquée au cloud source
            cloud_sol = object_cloud_solved(
                cloud_src, R_src_k, t_src_k, R_sol_k, t_sol_k,
            ).astype(np.float32)  # (P, 3)

            ach_field = frame.solved.contact_achieved.env[k].field          # MultiChannelField
            self._h_achieved[k].points = cloud_sol
            self._h_achieved[k].colors = _obj_contact_colors(
                ach_field, c, ui.color_mode, margin, uniform_rgb=_ACHIEVED_RGB,
            )
            self._h_achieved[k].point_size = sz
            self._h_achieved[k].visible = bool(self._cb_achieved.value)

            # --- Witness ATTEINT (probe résolu → witness monde via pose canal RÉSOLUE) ---
            segs_ach = witness_segments(
                cloud_sol, ach_field.witness[c], ach_field.active[c], R_sol_ch, t_sol_ch,
            )
            if bool(self._cb_wit_achieved.value) and len(segs_ach):
                self._h_wit_achieved[k].points = segs_ach
                self._h_wit_achieved[k].colors = np.tile(
                    np.array([[[0, 200, 120]]], np.uint8), (len(segs_ach), 2, 1),
                )
            self._h_wit_achieved[k].visible = (
                bool(self._cb_wit_achieved.value) and len(segs_ach) > 0
            )
