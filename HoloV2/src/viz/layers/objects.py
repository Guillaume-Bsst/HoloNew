"""Couche objets — les nuages de points posés, coloriés par leur PROPRE champ env
(``targets.env_interaction``) sur le canal sélectionné (choisir ``ground`` pour voir un objet
reposer sur le sol). Une poignée persistante par objet."""
from __future__ import annotations

import numpy as np

from ..core import colors, viser_ops
from ..core.layer import UiState
from ..model import VizContext, VizFrame


class ObjectsLayer:
    """Nuages de points objets, coloriés par le champ d'interaction d'env."""

    folder = "Static"

    def setup(self, server, gui, ctx: VizContext) -> None:
        """Initialise les poignées et construit l'arborescence GUI. Assemble une poignée
        persistante par objet pour les nuages de points."""
        self._channel_names = ctx.channel_names
        self._margin = ctx.margin
        self._handles = [
            viser_ops.add_point_cloud(server, f"/obj{k}", np.zeros((1, 3), np.float32),
                                      np.zeros((1, 3), np.uint8), point_size=0.012)
            for k in range(ctx.n_objects)]
        self._cb = gui.add_checkbox("object clouds", True)
        self._cb.on_update(lambda _: [setattr(h, "visible", self._cb.value) for h in self._handles])

    def update(self, frame: VizFrame, ui: UiState) -> None:
        """Rafraîchit les géométries et couleurs des nuages objets pour le frame courant,
        selon le mode couleur sélectionné (uniforme / distance / masque actif).
        No-op (masque tous les handles) si les données sont absentes ou le canal inconnu."""
        # Garde no-op : données manquantes ou canal inconnu → masquer tous les handles et sortir
        if (frame.targets is None
                or frame.object_clouds_world is None
                or ui.channel not in self._channel_names):
            for h in self._handles:
                h.visible = False
            return
        c = self._channel_names.index(ui.channel)
        env = frame.targets.env_interaction.per_object
        for k, h in enumerate(self._handles):
            # Aligner sur les données disponibles : handle sans nuage → masquer
            if k >= len(frame.object_clouds_world) or k >= len(env):
                h.visible = False
                continue
            pts = np.asarray(frame.object_clouds_world[k], np.float32)
            if ui.color_mode == "distance":
                col = colors.heat_distance(env[k].distance[c], self._margin)
            elif ui.color_mode == "active":
                col = colors.active_mask(env[k].active[c])
            else:                                                # uniforme orange
                col = np.tile(np.array([255, 140, 0], np.uint8), (pts.shape[0], 1))
            h.points = pts
            h.colors = col
            h.point_size = float(ui.point_size)
            h.visible = self._cb.value
