"""Style layer — les L cibles de style de lien (StyleTargets) : points à ``style.position``
(orange uniforme) + repères d'orientation par lien (3 courtes arêtes xyz depuis
``style.orientation`` wxyz) + étiquettes de nom de lien. Poignées persistantes ;
repères re-poussés par frame. C'est LA couche de validation de style clé."""
from __future__ import annotations

import numpy as np

from ..core import colors, viser_ops
from ..core.layer import UiState
from ..model import VizContext, VizFrame

# Longueur des axes du repère d'orientation (mètres)
_AXIS_LEN = 0.08


class StyleLayer:
    """Cibles de style : points + repères d'orientation + étiquettes de lien."""

    folder = "Style targets"

    def setup(self, server, gui, ctx: VizContext) -> None:
        """Initialise les poignées de nuage de points, de segments de repère et d'étiquettes
        textuelles. Trois cases à cocher indépendantes par sous-ensemble de rendu."""
        self._links = ctx.style_link_names
        L = len(self._links)
        # Nuage de points (un point par lien, orange uniforme)
        self._pts = viser_ops.add_point_cloud(
            server, "/style_pts",
            np.zeros((max(L, 1), 3), np.float32),
            np.tile(np.array([255, 170, 0], np.uint8), (max(L, 1), 1)),
            point_size=0.03)
        # Segments de repère (3 axes par lien -> 3L segments au total)
        self._frames = viser_ops.add_line_segments(
            server, "/style_frames",
            np.zeros((1, 2, 3), np.float32),
            np.zeros((1, 2, 3), np.uint8),
            line_width=2.5)
        # Étiquettes textuelles (une par lien)
        self._labels = [
            viser_ops.add_label(server, f"/style_label/{name}", name, (0.0, 0.0, 0.0))
            for name in self._links
        ]
        self._cb_p = gui.add_checkbox("link points", True)
        self._cb_f = gui.add_checkbox("orientation frames", True)
        self._cb_l = gui.add_checkbox("link labels", False)
        self._cb_p.on_update(lambda _: setattr(self._pts, "visible", self._cb_p.value))
        self._cb_f.on_update(lambda _: setattr(self._frames, "visible", self._cb_f.value))
        self._cb_l.on_update(lambda _: [setattr(h, "visible", self._cb_l.value)
                                         for h in self._labels])

    def update(self, frame: VizFrame, ui: UiState) -> None:
        """Rafraîchit points, repères et étiquettes pour le frame courant.
        No-op (masque tous les handles) si les cibles de style sont absentes."""
        # Garde no-op : targets manquant ou style absent → masquer et sortir
        if frame.targets is None or frame.targets.style is None:
            self._pts.visible = False
            self._frames.visible = False
            for h in self._labels:
                h.visible = False
            return

        style = frame.targets.style
        pos = np.asarray(style.position, np.float32)             # (L, 3)

        # --- Points de position ---
        self._pts.points = pos
        self._pts.colors = np.tile(np.array([255, 170, 0], np.uint8), (len(pos), 1))
        self._pts.point_size = max(float(ui.point_size) * 2.0, 0.02)
        # Restaurer visible selon checkbox (chemin nominal)
        self._pts.visible = self._cb_p.value

        # --- Repères d'orientation (3 axes xyz, couleurs AXIS_COLORS) ---
        if self._cb_f.value and style.orientation is not None:
            rots = viser_ops.quat_wxyz_to_R(style.orientation)   # (L, 3, 3)
            segs, cols = [], []
            for i in range(len(self._links)):
                for a in range(3):
                    d = rots[i][:, a]                             # direction monde de l'axe a
                    segs.append([pos[i], pos[i] + d * _AXIS_LEN])
                    cols.append([colors.AXIS_COLORS[a], colors.AXIS_COLORS[a]])
            self._frames.points = np.asarray(segs, np.float32)
            self._frames.colors = np.asarray(cols, np.uint8)
            self._frames.visible = True
        else:
            self._frames.visible = False

        # --- Étiquettes textuelles ---
        for i, h in enumerate(self._labels):
            h.position = tuple(float(v) for v in pos[i])
            h.visible = self._cb_l.value
