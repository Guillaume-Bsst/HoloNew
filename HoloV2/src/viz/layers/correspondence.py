"""Couche correspondance (roadmap #4) — lignes SMPL↔G1 à la config résolue.

L'appariement OT figé (``ctx.correspondence``) associe à chaque point de contrôle robot ``m`` un point
de la surface SMPL (``smpl_idx[m]``). On trace le segment ``human_cloud_world[smpl_idx[m]] ->
solved.robot_points_world[m]`` : la carte humain→robot rendue visible (lignes courtes, non croisées =
bonne carte). Couche SOLVE-GATED : sans ``solved`` le côté robot n'a pas de position monde -> masquée.
Pure consommatrice ; viser confiné ici."""
from __future__ import annotations

import numpy as np

from ..core.layer import UiState
from ..model import VizContext, VizFrame

_LINE_RGB = np.array([120, 220, 230], np.uint8)     # cyan doux


def correspondence_segments(human_cloud_world: np.ndarray, robot_points_world: np.ndarray,
                            smpl_idx: np.ndarray) -> np.ndarray:
    """(M, 2, 3) f32 : segment[m] = (human_cloud_world[smpl_idx[m]], robot_points_world[m]).

    Lève si un ``smpl_idx`` sort du nuage humain (appariement incohérent)."""
    human = np.asarray(human_cloud_world, np.float64)
    robot = np.asarray(robot_points_world, np.float64)
    idx = np.asarray(smpl_idx, np.int64)
    if idx.size and (idx.min() < 0 or idx.max() >= human.shape[0]):
        raise ValueError(f"smpl_idx hors plage [0, {human.shape[0]})")
    a = human[idx]                                   # (M, 3) extrémité humaine
    b = robot                                        # (M, 3) extrémité robot
    return np.stack([a, b], axis=1).astype(np.float32)


class CorrespondenceLayer:
    """``Layer`` : un handle de segments persistant, MAJ par frame depuis l'appariement OT. No-op si
    ``frame.solved is None`` (couche solve-gated : sans ``q`` résolu les M points robot n'ont pas de
    position monde)."""

    folder = "Correspondance SMPL↔G1"

    def setup(self, server, gui, ctx: VizContext) -> None:
        """Crée la poignée de segments persistante et le dossier GUI une fois."""
        self._smpl_idx = np.asarray(ctx.correspondence.smpl_idx, np.int64)
        with gui.add_folder(self.folder):
            self._cb = gui.add_checkbox("lignes SMPL↔G1", True)
        # Segment factice initial (1, 2, 3) : sera remplacé à la première MAJ
        self._h = server.scene.add_line_segments(
            "/correspondence", np.zeros((1, 2, 3), np.float32), np.zeros((1, 2, 3), np.uint8),
            line_width=1.5)

    def update(self, frame: VizFrame, ui: UiState) -> None:
        """Rafraîchit les segments de correspondance pour le frame courant.

        Gardes de sortie anticipée (masquage silencieux, aucune levée) :
          - ``frame.solved is None``              (couche solve-gated)
          - ``frame.human_cloud_world is None``   (nuage humain absent)
          - ``not self._cb.value``                (couche désactivée par l'utilisateur)
        """
        if frame.solved is None or frame.human_cloud_world is None or not bool(self._cb.value):
            self._h.visible = False
            return
        seg = correspondence_segments(frame.human_cloud_world, frame.solved.robot_points_world,
                                      self._smpl_idx)                      # (M, 2, 3)
        col = np.broadcast_to(_LINE_RGB, (seg.shape[0], 2, 3)).astype(np.uint8)
        self._h.points = seg
        self._h.colors = col
        self._h.visible = self._cb.value
