"""Couche objets — les nuages de points posés, coloriés par leur PROPRE champ env
(``targets.env_interaction``) sur le canal sélectionné (choisir ``ground`` pour voir un objet
reposer sur le sol). Une poignée persistante par objet.

Sous-toggle **«cloud objet résolu»** : quand activé et ``frame.solved`` présent, trace un
nuage supplémentaire par objet à la pose résolue par l'optimiseur (vert distinctif). Permet
de visualiser de combien chaque objet a été déplacé depuis sa pose source (orange).

Sous-toggle **«object mesh»** : le mesh objet SOURCE translucide (un par objet, tan chaud), posé
par la MÊME transformation rigide que le nuage source (``world = verts_local @ R.T + t`` avec la pose
NON-scalée de ``frame.pose``), donc il s'aligne EXACTEMENT sur le nuage source. Purement géométrie
source : PAS solve-gaté (visible avec ou sans ``--solve``) et indépendant du canal/nuage sélectionné.
Re-ajouté par frame (les sommets monde bougent), à la manière de la couche fantôme SMPL."""
from __future__ import annotations

import numpy as np

from ..core import colors, viser_ops
from ..core.layer import UiState
from ..model import VizContext, VizFrame
from ._contact_ops import object_cloud_solved

# Couleur du cloud résolu — vert distinctif (source = orange [255, 140, 0])
_GREEN_SOLVED = np.array([0, 200, 80], np.uint8)
# Couleur du mesh objet source — tan chaud translucide, DISTINCT du gris fantôme SMPL (200,200,210)
_TAN_MESH = (230, 190, 140)


class ObjectsLayer:
    """Nuages de points objets, coloriés par le champ d'interaction d'env.

    Deux jeux de poignées par objet : source (observé, orange) et résolu (optimiseur, vert).
    Le cloud résolu est solve-gaté et contrôlé par un sous-toggle indépendant."""

    folder = "Static"

    def setup(self, server, gui, ctx: VizContext) -> None:
        """Initialise les poignées et construit l'arborescence GUI.

        Crée une poignée source et une poignée résolue par objet.  Le toggle principal
        «object clouds» pilote les sources ; «cloud objet résolu» pilote les handles résolus
        et déclenche un re-rendu immédiat en pause via ``_on_change``."""
        self._server = server     # requis pour re-ajouter le mesh objet source par frame
        self._channel_names = ctx.channel_names
        self._margin = ctx.margin
        self._object_meshes = ctx.object_meshes  # géométrie mesh source (local + faces), un par objet
        self._last_frame = None   # dernier VizFrame reçu (re-rendu en pause)
        self._last_ui = None      # dernière UiState reçue (re-rendu en pause)
        # Poignées mesh source (re-ajoutées par frame comme le fantôme SMPL) — None jusqu'au 1er rendu
        self._mesh_handles = [None] * len(ctx.object_meshes)

        # Poignées source (pose observée)
        self._handles = [
            viser_ops.add_point_cloud(server, f"/obj{k}", np.zeros((1, 3), np.float32),
                                      np.zeros((1, 3), np.uint8), point_size=0.012)
            for k in range(ctx.n_objects)]
        # Poignées résolues (pose après optimisation)
        self._handles_sol = [
            viser_ops.add_point_cloud(server, f"/obj{k}_sol", np.zeros((1, 3), np.float32),
                                      np.zeros((1, 3), np.uint8), point_size=0.012)
            for k in range(ctx.n_objects)]

        self._cb = gui.add_checkbox("object clouds", True)
        self._cb.on_update(lambda _: [setattr(h, "visible", self._cb.value)
                                       for h in self._handles])
        self._cb_solved = gui.add_checkbox("cloud objet résolu", False)
        self._cb_mesh = gui.add_checkbox("object mesh", True)

        def _on_change(_) -> None:
            """Re-rend le frame courant immédiatement quand un sous-toggle change en pause."""
            if self._last_frame is not None and self._last_ui is not None:
                self.update(self._last_frame, self._last_ui)

        self._cb_solved.on_update(_on_change)
        self._cb_mesh.on_update(_on_change)

    def update(self, frame: VizFrame, ui: UiState) -> None:
        """Rafraîchit les géométries et couleurs des nuages objets pour le frame courant,
        selon le mode couleur sélectionné (uniforme / distance / masque actif).

        No-op (masque tous les handles source ET résolus) si les données sont absentes ou le
        canal inconnu.  Le cloud résolu est solve-gaté : masqué quand ``frame.solved is None``
        ou quand le toggle «cloud objet résolu» est désactivé.  Des gardes d'index protègent
        contre les désalignements entre n_objects et les longueurs des listes dans le frame."""
        # Mémorise le frame et l'état UI pour permettre le re-rendu en pause (bascule toggle)
        self._last_frame = frame
        self._last_ui = ui

        # Mesh objet source : rendu AVANT le guard nuage/canal, car il ne dépend QUE de frame.pose
        # + géométrie mesh (ni targets ni ui.channel). Pas solve-gaté (géométrie source).
        self._update_meshes(frame)

        # Garde no-op : données manquantes ou canal inconnu → masquer tous les handles et sortir
        if (frame.targets is None
                or frame.object_clouds_world is None
                or ui.channel not in self._channel_names):
            for h in self._handles:
                h.visible = False
            for h in self._handles_sol:
                h.visible = False
            return

        c = self._channel_names.index(ui.channel)
        env = frame.targets.env_interaction.per_object
        # Le cloud résolu n'est tracé que si le toggle est ON et que solved est présent
        show_solved = bool(self._cb_solved.value) and frame.solved is not None

        for k, (h, h_sol) in enumerate(zip(self._handles, self._handles_sol)):
            # Aligner sur les données disponibles : handle sans nuage → masquer les deux
            if k >= len(frame.object_clouds_world) or k >= len(env):
                h.visible = False
                h_sol.visible = False
                continue

            # --- Cloud source (pose observée) -------------------------------------------
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

            # --- Cloud résolu (pose optimiseur) — solve-gaté ----------------------------
            if not show_solved:
                h_sol.visible = False
                continue
            # Garde : pose source absente ou objet hors borne dans les poses résolues
            if (frame.pose is None
                    or k >= len(frame.pose.object_rot)
                    or k >= len(frame.solved.object_poses)):
                h_sol.visible = False
                continue

            # Extraction de la pose source (monde)
            R_src = np.asarray(frame.pose.object_rot[k], np.float64)   # (3, 3)
            t_src = np.asarray(frame.pose.object_pos[k], np.float64)   # (3,)
            # Extraction de la pose résolue depuis le bundle solved
            pose7 = frame.solved.object_poses[k]                        # (7,) xyz qwxyz
            t_sol = np.asarray(pose7[:3], np.float64)                   # (3,)
            R_sol = viser_ops.quat_wxyz_to_R(pose7[3:7][np.newaxis])[0]  # (3, 3)

            # Transfo T_résolu ∘ T_source⁻¹ appliquée au cloud source
            pts_sol = object_cloud_solved(pts, R_src, t_src, R_sol, t_sol).astype(np.float32)
            col_sol = np.tile(_GREEN_SOLVED, (pts_sol.shape[0], 1))
            h_sol.points = pts_sol
            h_sol.colors = col_sol
            h_sol.point_size = float(ui.point_size)
            h_sol.visible = True

    def _update_meshes(self, frame: VizFrame) -> None:
        """Rend le mesh objet SOURCE translucide, un par objet, INDÉPENDAMMENT du guard nuage/canal.

        Aligné EXACTEMENT sur le nuage source : posé par la MÊME transformation rigide que le nuage
        (``world = verts_local @ R.T + t`` avec ``R = frame.pose.object_rot[k]``,
        ``t = frame.pose.object_pos[k]`` — poses NON-scalées, aucun scale à appliquer). Re-ajouté par
        frame (les sommets monde changent), à la manière de la couche fantôme SMPL. Pas solve-gaté.

        Gardes : ``frame.pose is None`` ou objet hors borne des poses source → masque le handle s'il
        existe. ``object_meshes`` vide → boucle vide, aucun mesh."""
        show = bool(self._cb_mesh.value)
        for k, mesh in enumerate(self._object_meshes):
            h = self._mesh_handles[k]
            # Pas de pose disponible ou objet hors borne des poses source → masquer
            if frame.pose is None or k >= len(frame.pose.object_rot):
                if h is not None:
                    h.visible = False
                continue
            R = np.asarray(frame.pose.object_rot[k], np.float64)   # (3, 3)
            t = np.asarray(frame.pose.object_pos[k], np.float64)   # (3,)
            verts_local = np.asarray(mesh.vertices, np.float64)    # (V, 3) repère local objet
            world = (verts_local @ R.T + t).astype(np.float32)     # (V, 3) monde (transform rigide)
            faces = np.asarray(mesh.faces)                         # (F, 3) int
            h = self._server.scene.add_mesh_simple(
                f"/obj_mesh/{k}", world, faces,
                color=_TAN_MESH, opacity=0.4, side="double")
            self._mesh_handles[k] = h
            h.visible = show
