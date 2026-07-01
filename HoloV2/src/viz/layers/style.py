"""Style layer — les L cibles de style de lien (StyleTargets) : points à ``style.position``
(orange uniforme) + repères d'orientation par lien (3 courtes arêtes xyz depuis
``style.orientation`` wxyz) + étiquettes de nom de lien. Poignées persistantes ;
repères re-poussés par frame. C'est LA couche de validation de style clé.

En MIROIR de la référence, la couche trace aussi la version RÉSOLUE (eval) des mêmes cibles :
là où les liens suivis du robot ont réellement atterri après optimisation (``frame.solved.
style_achieved``, un ``StyleEval``). Points VERTS + repères d'orientation résolus, solve-gatés
(masqués tant qu'aucun résolu n'est disponible). Pas d'étiquettes résolues : les noms sont
identiques à la référence."""
from __future__ import annotations

import numpy as np

from ..core import colors, viser_ops
from ..core.layer import UiState
from ..model import VizContext, VizFrame

# Longueur des axes du repère d'orientation (mètres)
_AXIS_LEN = 0.08
# Couleur des points de style résolus — vert distinctif (référence = orange [255, 170, 0])
_GREEN_SOLVED = np.array([0, 200, 80], np.uint8)


class StyleLayer:
    """Cibles de style : points + repères d'orientation + étiquettes de lien.

    Deux jeux de poignées : RÉFÉRENCE (cible, orange) et RÉSOLU (atteint par l'optimiseur, vert).
    Le rendu résolu est solve-gaté et piloté par ses propres sous-toggles indépendants."""

    folder = "Style targets"

    def setup(self, server, gui, ctx: VizContext) -> None:
        """Initialise les poignées de nuage de points, de segments de repère et d'étiquettes
        textuelles, pour la référence ET le résolu. Cases à cocher indépendantes par sous-ensemble
        de rendu, câblées sur ``_on_change`` (re-rendu immédiat en pause)."""
        self._links = ctx.style_link_names
        L = len(self._links)
        self._last_frame = None   # dernier VizFrame reçu (re-rendu en pause)
        self._last_ui = None      # dernière UiState reçue (re-rendu en pause)

        # --- RÉFÉRENCE (cible) ---
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

        # --- RÉSOLU (atteint) — miroir vert, sans étiquettes ---
        # Nuage de points résolu (un point par lien, vert uniforme)
        self._pts_sol = viser_ops.add_point_cloud(
            server, "/style_pts_solved",
            np.zeros((max(L, 1), 3), np.float32),
            np.tile(_GREEN_SOLVED, (max(L, 1), 1)),
            point_size=0.03)
        # Segments de repère résolus (3 axes par lien)
        self._frames_sol = viser_ops.add_line_segments(
            server, "/style_frames_solved",
            np.zeros((1, 2, 3), np.float32),
            np.zeros((1, 2, 3), np.uint8),
            line_width=2.5)

        # --- Cases à cocher : référence puis résolu ---
        self._cb_p = gui.add_checkbox("link points", True)
        self._cb_f = gui.add_checkbox("orientation frames", True)
        self._cb_l = gui.add_checkbox("link labels", False)
        # Résolu : points ON par défaut ; repères OFF par défaut (évite le fouillis 3 axes × L
        # affichés deux fois quand référence et résolu se superposent).
        self._cb_p_sol = gui.add_checkbox("link points (résolu)", True)
        self._cb_f_sol = gui.add_checkbox("orientation frames (résolu)", False)

        def _on_change(_) -> None:
            """Re-rend le frame courant immédiatement quand une case change en pause."""
            if self._last_frame is not None and self._last_ui is not None:
                self.update(self._last_frame, self._last_ui)

        self._on_change = _on_change
        for cb in (self._cb_p, self._cb_f, self._cb_l, self._cb_p_sol, self._cb_f_sol):
            cb.on_update(_on_change)

    def update(self, frame: VizFrame, ui: UiState) -> None:
        """Rafraîchit points, repères et étiquettes (référence + résolu) pour le frame courant.
        La partie référence est un no-op (masque ses handles) si les cibles de style sont absentes ;
        la partie résolue est un no-op si aucun résolu n'est disponible. Les deux sont indépendantes."""
        # Mémorise le frame et l'état UI pour permettre le re-rendu en pause (bascule toggle)
        self._last_frame = frame
        self._last_ui = ui

        # --- RÉSOLU (solve-gaté), indépendant de la partie référence ---
        self._update_solved(frame, ui)

        # --- RÉFÉRENCE (comportement historique inchangé) ---
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

    def _update_solved(self, frame: VizFrame, ui: UiState) -> None:
        """Rend la version RÉSOLUE (eval) des cibles de style — points verts + repères d'orientation,
        depuis ``frame.solved.style_achieved`` (un ``StyleEval``). Solve-gaté : masque les handles
        résolus si aucun ``SolvedFrame`` ou aucun ``style_achieved`` n'est disponible. Ne touche PAS
        aux handles de référence."""
        solved = frame.solved
        if solved is None or solved.style_achieved is None:
            self._pts_sol.visible = False
            self._frames_sol.visible = False
            return

        se = solved.style_achieved
        pos = np.asarray(se.position, np.float32)                # (L, 3) position monde résolue

        # --- Points résolus (vert) — même taille que les points de référence ---
        self._pts_sol.points = pos
        self._pts_sol.colors = np.tile(_GREEN_SOLVED, (len(pos), 1))
        self._pts_sol.point_size = max(float(ui.point_size) * 2.0, 0.02)
        self._pts_sol.visible = self._cb_p_sol.value

        # --- Repères d'orientation résolus (3 axes xyz depuis se.rotation) ---
        # ``se.rotation`` est DÉJÀ une matrice de rotation monde (L, 3, 3) : pas de conversion
        # quaternion. Les COLONNES de la matrice sont les axes monde x/y/z du lien.
        if self._cb_f_sol.value:
            rots = np.asarray(se.rotation, np.float64)           # (L, 3, 3)
            segs, cols = [], []
            for i in range(len(pos)):
                for a in range(3):
                    d = rots[i][:, a]                            # direction monde de l'axe a (colonne a)
                    segs.append([pos[i], pos[i] + d * _AXIS_LEN])
                    cols.append([colors.AXIS_COLORS[a], colors.AXIS_COLORS[a]])
            self._frames_sol.points = np.asarray(segs, np.float32)
            self._frames_sol.colors = np.asarray(cols, np.uint8)
            self._frames_sol.visible = True
        else:
            self._frames_sol.visible = False
