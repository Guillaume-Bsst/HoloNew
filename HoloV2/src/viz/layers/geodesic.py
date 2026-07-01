"""Couche geodesic (roadmap #7) — champ géodésique des canaux objets/terrain.

Chaque canal non-plan porte une ``GeodesicTable`` (``ctx.channels[c].geodesic``) : ses ``points`` de
surface + ``normals``, et la matrice ``geo`` dont la ligne ``geo[src]`` EST le champ géodésique
mono-source depuis le point ``src``. On affiche les points (élevés en monde par la pose du canal),
colorés par ce champ mono-source normalisé en heatmap (proche = bleu, loin = rouge), source choisie par
un index GUI ; les ``normals`` sont optionnellement tracées en courts segments. Canal sol-plan
(``geodesic is None``) -> no-op. Pré-solve OK. Pure consommatrice ; viser confiné ici."""
from __future__ import annotations

import numpy as np

from ..core import colors
from ..core.layer import UiState
from ..model import VizContext, VizFrame

_NORMAL_LEN = 0.03                                   # longueur des segments de normale (m)
_NORMAL_RGB = np.array([60, 220, 200], np.uint8)


def geo_normalized(geo_row: np.ndarray) -> np.ndarray:
    """(P,) f64 dans [0,1] : distance géodésique mono-source normalisée par son max (source -> 0).

    Ligne constante (max ~ 0) -> tout 0 (pas de division par zéro).
    """
    d = np.asarray(geo_row, np.float64)
    hi = float(d.max()) if d.size else 0.0
    return d / hi if hi > 1e-12 else np.zeros_like(d)


def geo_heat_colors(geo_row: np.ndarray) -> np.ndarray:
    """(P,3) uint8 : heatmap de la géodésique mono-source normalisée (proche = bleu, loin = rouge)."""
    return colors.heat_distance(geo_normalized(geo_row), 1.0)


class GeodesicLayer:
    """Couche visu : par canal géodésique, un nuage de points (heat mono-source) + des normales.

    La source est un index GUI ; canaux sans table (``geodesic is None``) -> ignorés. Pré-solve OK.
    Gardes données manquantes : ``frame.pose`` absent ou ``object_idx`` hors bornes -> masque le
    handle concerné et continue (pas de levée). Ensemble de points vide -> masque aussi.
    """

    folder = "Champ géodésique"

    def setup(self, server, gui, ctx: VizContext) -> None:
        """Crée les contrôles GUI et un nuage + des segments de normale par canal géodésique."""
        self._ctx = ctx
        # Seuls les canaux portant une GeodesicTable sont rendus ; le sol-plan (None) est ignoré
        self._geo = [(ch.geodesic, ch.object_idx, ch.name)
                     for ch in ctx.channels if ch.geodesic is not None]
        max_src = max((g.n_points - 1 for g, _, _ in self._geo), default=0)
        with gui.add_folder(self.folder):
            self._cb_pts = gui.add_checkbox("points (heat géodésique)", False)
            self._cb_nrm = gui.add_checkbox("normales", False)
            self._src = gui.add_slider("point source", 0, max(max_src, 0), 1, 0)
        self._h_pts, self._h_nrm = [], []
        for _g, _oi, name in self._geo:
            self._h_pts.append(server.scene.add_point_cloud(
                f"/geodesic/{name}/pts",
                np.zeros((1, 3), np.float32),
                np.zeros((1, 3), np.uint8),
                point_size=0.008,
            ))
            self._h_nrm.append(server.scene.add_line_segments(
                f"/geodesic/{name}/nrm",
                np.zeros((1, 2, 3), np.float32),
                np.zeros((1, 2, 3), np.uint8),
                line_width=1.5,
            ))

    def update(self, frame: VizFrame, ui: UiState) -> None:
        """Rafraîchit les nuages géodésiques et les normales pour le frame courant.

        - ``show_pts`` / ``show_nrm`` : état des deux checkboxes GUI.
        - Canal ground (``object_idx is None``) : coordonnées locales = monde.
        - Canal objet : élève en monde via ``frame.pose.object_rot/pos[object_idx]``.
        - Gardes : ``frame.pose`` absent, ``object_idx`` hors bornes, ensemble vide ->
          masque les deux handles du canal concerné et passe au suivant.
        """
        show_pts = bool(self._cb_pts.value)
        show_nrm = bool(self._cb_nrm.value)
        src = int(self._src.value)
        for (geo, oi, _name), h_pts, h_nrm in zip(self._geo, self._h_pts, self._h_nrm):
            # --- Résolution de la pose (monde vs objet) ----------------------------------------
            if oi is None:
                # Canal en monde (sol/terrain) : transformation identité
                R: np.ndarray = np.eye(3)
                t: np.ndarray = np.zeros(3)
            else:
                # Canal objet : pose per-frame requise
                if frame.pose is None or oi >= len(frame.pose.object_rot):
                    h_pts.visible = False
                    h_nrm.visible = False
                    continue
                R = np.asarray(frame.pose.object_rot[oi], np.float64)
                t = np.asarray(frame.pose.object_pos[oi], np.float64)
            # --- Garde ensemble vide -----------------------------------------------------------
            if geo.n_points == 0:
                h_pts.visible = False
                h_nrm.visible = False
                continue
            # --- Points de surface en monde ---------------------------------------------------
            pw = np.asarray(geo.points, np.float64) @ R.T + t          # (P,3) monde
            # --- Nuage heat géodésique --------------------------------------------------------
            if show_pts:
                s = min(src, geo.n_points - 1)
                h_pts.points = pw.astype(np.float32)
                h_pts.colors = geo_heat_colors(geo.geo[s])             # heat mono-source
                h_pts.visible = True
            else:
                h_pts.visible = False
            # --- Normales (courts segments) ---------------------------------------------------
            if show_nrm:
                nw = np.asarray(geo.normals, np.float64) @ R.T          # normales (rotation seule)
                seg = np.stack([pw, pw + nw * _NORMAL_LEN], axis=1).astype(np.float32)  # (P,2,3)
                h_nrm.points = seg
                h_nrm.colors = np.broadcast_to(
                    _NORMAL_RGB, (seg.shape[0], 2, 3)
                ).astype(np.uint8)
                h_nrm.visible = True
            else:
                h_nrm.visible = False
