"""Fields layer — lignes witness (point -> surface la plus proche) et normales (court segment
le long de la direction de contact) pour les sondes ACTIVES du canal sélectionné. Les canaux
objet stockent witness/direction dans le référentiel OBJET-LOCAL, donc ils sont projetés en monde
via la paire (R, t) par-frame de l'objet ; le canal ground est déjà en monde.

Convention d'ordre des canaux (prepare.runner._validate) : canal 0 = ground (monde) ;
canal c >= 1 = objet c-1 (local objet)."""
from __future__ import annotations

import numpy as np

from ..core import viser_ops
from ..core.layer import UiState
from ..model import VizContext, VizFrame

# Nombre maximal de segments rendus (sous-échantillonnage aléatoire si dépassé)
_MAX_SEG = 400


class FieldsLayer:
    """Lignes witness et normales de contact pour le canal humain sélectionné."""

    folder = "Interaction - human"

    def setup(self, server, gui, ctx: VizContext) -> None:
        """Initialise les deux poignées de segments (witness + normales) et les cases à cocher GUI.
        Les deux handles sont initialement cachés (checkboxes à False)."""
        self._channel_names = ctx.channel_names
        z, zc = np.zeros((1, 2, 3), np.float32), np.zeros((1, 2, 3), np.uint8)
        self._wit = viser_ops.add_line_segments(server, "/witness", z, zc, line_width=1.5)
        self._nrm = viser_ops.add_line_segments(server, "/normals", z, zc, line_width=2.0)
        self._cb_w = gui.add_checkbox("witness lines", False)
        self._cb_n = gui.add_checkbox("normals", False)
        self._cb_w.on_update(lambda _: setattr(self._wit, "visible", self._cb_w.value))
        self._cb_n.on_update(lambda _: setattr(self._nrm, "visible", self._cb_n.value))

    def update(self, frame: VizFrame, ui: UiState) -> None:
        """Rafraîchit les segments witness et normales pour le frame courant.
        No-op (masque les deux handles) si les données sont absentes ou le canal inconnu."""
        # Garde no-op : champ manquant, nuage absent ou canal inconnu → masquer et sortir
        if (frame.human_field is None
                or frame.human_cloud_world is None
                or ui.channel not in self._channel_names):
            self._wit.visible = False
            self._nrm.visible = False
            return

        c = self._channel_names.index(ui.channel)
        field = frame.human_field
        idx = np.where(np.asarray(field.active[c], bool))[0]
        want = self._cb_w.value or self._cb_n.value

        if len(idx) and want:
            # Sous-échantillonnage déterministe si trop de sondes actives
            if len(idx) > _MAX_SEG:
                idx = np.random.default_rng(0).choice(idx, _MAX_SEG, replace=False)
            pts = np.asarray(frame.human_cloud_world, np.float64)[idx]   # (S, 3) monde
            wit = np.asarray(field.witness[c], np.float64)[idx]           # (S, 3) local canal
            dirn = np.asarray(field.direction[c], np.float64)[idx]        # (S, 3) local canal
            # Projection local objet -> monde (ground déjà en monde)
            object_idx = None if c == 0 else c - 1
            if object_idx is not None:
                R = np.asarray(frame.pose.object_rot[object_idx], np.float64)   # (3, 3)
                t = np.asarray(frame.pose.object_pos[object_idx], np.float64)   # (3,)
                wit = wit @ R.T + t
                dirn = dirn @ R.T
        else:
            # Aucune sonde active ou aucun toggle actif : tableaux vides
            pts = wit = dirn = np.zeros((0, 3), np.float64)

        # --- Lignes witness (point -> surface la plus proche) ---
        if self._cb_w.value and len(pts):
            seg = np.stack([pts, wit], axis=1).astype(np.float32)
            self._wit.points = seg
            self._wit.colors = np.tile([[[230, 230, 60]]], (len(pts), 2, 1)).astype(np.uint8)
        # Restaurer visible (chemin nominal) ou masquer (aucun point / checkbox off)
        self._wit.visible = self._cb_w.value and len(pts) > 0

        # --- Normales (court segment le long de la direction de contact) ---
        if self._cb_n.value and len(pts):
            seg = np.stack([pts, pts + dirn * 0.05], axis=1).astype(np.float32)
            self._nrm.points = seg
            self._nrm.colors = np.tile([[[60, 220, 200]]], (len(pts), 2, 1)).astype(np.uint8)
        # Restaurer visible (chemin nominal) ou masquer
        self._nrm.visible = self._cb_n.value and len(pts) > 0
