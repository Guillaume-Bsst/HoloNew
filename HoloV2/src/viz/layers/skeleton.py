"""Couche squelette — segments d'arête SMPL (parent→enfant) depuis ``pose.bone_pos`` (poignée
persistante, rafraîchie par-frame — pas de re-ajout)."""
from __future__ import annotations

import numpy as np

from ..core import viser_ops
from ..core.layer import UiState
from ..model import VizContext, VizFrame


class SkeletonLayer:
    """Squelette SMPL dessiné comme segments de lignes depuis les positions d'os."""

    folder = "Skeleton"

    def setup(self, server, gui, ctx: VizContext) -> None:
        """Initialise la poignée et construit l'arborescence GUI. Assemble les paires
        (parent→enfant) une fois depuis ``ctx.smpl_parents``."""
        parents = np.asarray(ctx.smpl_parents)
        self._pairs = [(int(parents[j]), j) for j in range(len(parents)) if parents[j] >= 0]
        n = max(len(self._pairs), 1)
        seg0 = np.zeros((n, 2, 3), np.float32)
        col = np.tile([[[0, 120, 255]]], (n, 2, 1)).astype(np.uint8)
        self._handle = viser_ops.add_line_segments(server, "/skeleton", seg0, col, line_width=3.0)
        self._cb = gui.add_checkbox("skeleton", True)
        self._cb.on_update(lambda _: setattr(self._handle, "visible", self._cb.value))

    def update(self, frame: VizFrame, ui: UiState) -> None:
        """Rafraîchit les points des segments depuis les positions d'os du frame courant."""
        if self._pairs:
            bp = np.asarray(frame.pose.bone_pos, np.float32)
            seg = np.stack([np.stack([bp[a], bp[b]]) for a, b in self._pairs]).astype(np.float32)
            self._handle.points = seg
        self._handle.visible = self._cb.value
