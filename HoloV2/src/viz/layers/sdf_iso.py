"""Couche sdf_iso (roadmap #6) — iso-surface (≈ surface) des SDF de canaux dans le viewer prod.

Approche l'iso-surface par la BANDE proche-surface ``|d| < band`` des noeuds de grille, colorée par
distance signée (divergent : bleu intérieur / blanc surface / rouge extérieur) ; le passage par zero
trace la surface. Logique portée de ``viz/sdf.py`` (masque de bande + ``_diverging``, ici via
``core.colors.diverging`` ; coordonnées nœuds via ``core.geometry.node_coords``). Les points sont en
frame LOCALE du canal et élèves en monde par la pose per-frame (ground = monde ; objet =
``frame.pose.object_rot/pos[object_idx]``). Pré-solve OK (n'utilise pas ``solved``). Pure
consommatrice ; viser confiné dans ``setup``."""
from __future__ import annotations

import numpy as np

from ..core import colors
from ..core.geometry import node_coords
from ..core.layer import UiState
from ..model import VizContext, VizFrame
from ...prepare.contracts import SDF


def iso_band_points(sdf: SDF, band: float) -> tuple[np.ndarray, np.ndarray]:
    """Noeuds de la grille dont ``|d| < band`` (la coquille proche-surface ≈ iso-surface).

    Paramètres
    ----------
    sdf :
        Grille SDF portant ``grid``, ``origin``, ``spacing`` (frame LOCALE du canal).
    band :
        Demi-épaisseur de la bande en mètres ; seuls les nœuds avec ``|d| < band`` sont retenus.

    Retourne
    --------
    points : (K, 3) float64
        Coordonnées locales des K nœuds retenus.
    dist : (K,) float64
        Distance signée de chaque nœud retenu (négative = intérieur, positive = extérieur).
    """
    coords = node_coords(sdf)                          # (Nx, Ny, Nz, 3) coordonnées locales
    mask = np.abs(sdf.grid) < float(band)              # (Nx, Ny, Nz) booléen bande
    return coords[mask].astype(np.float64), sdf.grid[mask].astype(np.float64)


class SdfIsoLayer:
    """Couche visu : une coquille de bande SDF par canal, posée en monde par frame. Pré-solve OK.

    Crée un nuage de points par canal (``setup``). À chaque ``update``, sélectionne les nœuds
    dans la bande ``|d| < bande (m)`` (GUI), les élève en monde (identité pour le ground, pose
    per-frame pour les objets), et les colorie par distance signée (divergent bleu/blanc/rouge).
    """

    folder = "SDF iso (surface)"

    def setup(self, server, gui, ctx: VizContext) -> None:
        """Crée les contrôles GUI (dossier, checkbox, curseur bande) et un nuage par canal."""
        self._ctx = ctx
        with gui.add_folder(self.folder):
            self._cb = gui.add_checkbox("bande iso", False)
            self._band = gui.add_number(
                "bande (m)", float(ctx.margin), min=0.005, max=0.5, step=0.005)
        # Un handle viser par canal — persistant, points recalculés à chaque update
        self._handles = []
        for ch in ctx.channels:
            h = server.scene.add_point_cloud(
                f"/sdf_iso/{ch.name}",
                np.zeros((1, 3), np.float32),
                np.zeros((1, 3), np.uint8),
                point_size=0.006,
            )
            self._handles.append(h)

    def update(self, frame: VizFrame, ui: UiState) -> None:
        """Rafraîchit les coquilles iso pour le frame courant.

        - Si la checkbox est désactivée, masque tous les handles et sort.
        - Canal ground (``object_idx is None``) : coordonnées locales = monde.
        - Canal objet : élève en monde via ``frame.pose.object_rot/pos[object_idx]``.
        - Gardes données manquantes : ``frame.pose`` absent ou ``object_idx`` hors bornes →
          masque le handle concerné, continue (pas de levée d'exception).
        """
        show = bool(self._cb.value)
        band = float(self._band.value)
        for ch, h in zip(self._ctx.channels, self._handles):
            # Checkbox désactivée → masquer immédiatement
            if not show:
                h.visible = False
                continue
            pts_local, dist = iso_band_points(ch.sdf, band)     # (K, 3) locale, (K,)
            if ch.object_idx is None:
                # Canal ground : frame locale == monde (SDF posé en monde à la construction)
                pts_world = pts_local
            else:
                # Canal objet : pose per-frame requise
                if frame.pose is None or ch.object_idx >= len(frame.pose.object_rot):
                    h.visible = False
                    continue
                R = np.asarray(frame.pose.object_rot[ch.object_idx], np.float64)  # (3, 3)
                t = np.asarray(frame.pose.object_pos[ch.object_idx], np.float64)  # (3,)
                pts_world = pts_local @ R.T + t
            h.points = pts_world.astype(np.float32)
            h.colors = colors.diverging(dist, max(band, 1e-9))   # bleu/blanc/rouge signé
            h.visible = show                                       # = True ici (chemin nominal)
