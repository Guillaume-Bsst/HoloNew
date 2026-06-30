"""Couche fantôme — le mesh SMPL en arrière-plan translucide. Par-frame (les sommets changent à
chaque image), la couche le re-rajoute dans ``update`` (exception justifiée du design) ; la checkbox
le basculant."""
from __future__ import annotations

import numpy as np

from ..core.layer import UiState
from ..model import VizContext, VizFrame


class GhostLayer:
    """Mesh SMPL statique et translucide lisant depuis ``VizFrame.smpl_verts_world``."""

    folder = "Static"

    def setup(self, server, gui, ctx: VizContext) -> None:
        """Initialise la poignée et construit l'arborescence GUI. Les faces SMPL sont cachées
        (statiques) et assemblées une fois."""
        self._server = server
        self._faces = np.asarray(ctx.smpl_faces)
        self._handle = None
        self._cb = gui.add_checkbox("SMPL ghost", True)
        self._cb.on_update(lambda _: self._set_visible())

    def _set_visible(self) -> None:
        """Bascule la visibilité du mesh fantôme selon la checkbox."""
        if self._handle is not None:
            self._handle.visible = self._cb.value

    def update(self, frame: VizFrame, ui: UiState) -> None:
        """Rafraîchit le mesh SMPL depuis les sommets du frame courant. Source non-paramétrique
        (smpl_verts_world=None) → pas de mesh."""
        if frame.smpl_verts_world is None:
            # Masquer le handle existant si les données disparaissent
            if self._handle is not None:
                self._handle.visible = False
            return
        self._handle = self._server.scene.add_mesh_simple(
            "/ghost", np.asarray(frame.smpl_verts_world, np.float32), self._faces,
            color=(200, 200, 210), opacity=0.45, side="double")
        self._handle.visible = self._cb.value
