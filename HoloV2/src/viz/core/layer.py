"""Protocole de couche et sélecteurs UI partagés — la jointure entre Player et chaque couche visu.
Une couche possède son propre dossier GUI + des poignées de scène persistantes (crées une fois à
``setup``) et, à chaque ``update``, ne rafraîchit que ces poignées depuis le modèle visu — jamais
touche une autre couche. ``@runtime_checkable`` pour que l'app/tests puissent assurer qu'une classe
est une Couche (membres présents)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..model import VizContext, VizFrame


@dataclass(frozen=True)
class UiState:
    """Sélecteurs cross-couche, gelés, assemblés par le Player et passés à chaque ``update``."""

    channel: str        # Nom du canal sélectionné (sol / obj0 / …)
    color_mode: str     # Mode de couleur : 'uniform' | 'distance' | 'active'
    point_size: float   # Taille des points du nuage (m)


@runtime_checkable
class Layer(Protocol):
    """Protocole structural pour une couche visu.

    Une couche rend un sous-ensemble persistant de la scène (dossier GUI + poignées viser). La
    vie d'une couche : (1) setup() une fois pour créer la poignée et initialiser l'arborescence
    GUI ; (2) update() appelée à chaque frame pour rafraîchir les données visualisées selon l'état
    UI et le frame visu courant. Les couches ne communiquent jamais entre elles."""

    folder: str
    """Chemin du dossier GUI au sein du serveur viser (ex. 'human', 'object_0')."""

    def setup(self, server, gui, ctx: VizContext) -> None:
        """Crée les poignées viser persistantes et construit l'arborescence GUI une fois.

        Args:
            server: serveur viser (ns.Server).
            gui: racine GUI viser (ns.gui.Folder).
            ctx: ressources statiques de la scène (VizContext).
        """
        ...

    def update(self, frame: VizFrame, ui: UiState) -> None:
        """Rafraîchit les géométries et propriétés pour le frame courant, selon les sélecteurs UI.

        Args:
            frame: vue montée et cible du frame (VizFrame).
            ui: sélecteurs UI courant (canal, mode couleur, taille point).
        """
        ...
