"""Couche nuage humain — le nuage de points posé colorié par le champ SÉLECTIONNÉ de ``human_field``
(uniforme / heatmap distance / masque actif). Poignée de nuage de points persistante, rafraîchie
par-frame."""
from __future__ import annotations

import numpy as np

from ..core import colors, viser_ops
from ..core.layer import UiState
from ..model import VizContext, VizFrame


class HumanCloudLayer:
    """Nuage de points humain colorié par le champ d'interaction sélectionné."""

    folder = "Interaction - human"

    def setup(self, server, gui, ctx: VizContext) -> None:
        """Initialise la poignée et construit l'arborescence GUI. Assemble une poignée persistante
        pour le nuage de points humain avec des données de départ vides."""
        self._channel_names = ctx.channel_names
        self._margin = ctx.margin
        self._handle = viser_ops.add_point_cloud(
            server, "/human", np.zeros((1, 3), np.float32), np.zeros((1, 3), np.uint8),
            point_size=0.012)
        self._cb = gui.add_checkbox("human cloud", True)
        self._cb.on_update(lambda _: setattr(self._handle, "visible", self._cb.value))

    def update(self, frame: VizFrame, ui: UiState) -> None:
        """Rafraîchit les géométries et couleurs du nuage humain pour le frame courant,
        selon le mode couleur sélectionné (uniforme / distance / masque actif)."""
        c = self._channel_names.index(ui.channel)
        field = frame.human_field
        if ui.color_mode == "distance":
            col = colors.heat_distance(field.distance[c], self._margin)
        elif ui.color_mode == "active":
            col = colors.active_mask(field.active[c])
        else:                                                    # uniforme
            col = np.tile(np.array([185, 185, 195], np.uint8),
                          (frame.human_cloud_world.shape[0], 1))
        self._handle.points = np.asarray(frame.human_cloud_world, np.float32)
        self._handle.colors = col
        self._handle.point_size = float(ui.point_size)
        self._handle.visible = self._cb.value
