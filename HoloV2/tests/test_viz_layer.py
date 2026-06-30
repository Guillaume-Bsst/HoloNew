"""Protocole Layer + UiState — sélecteurs gelés et conformance structurelle runtime_checkable."""
import dataclasses

import numpy as np
import pytest

from src.viz.core.layer import Layer, UiState


def test_uistate_fields_and_frozen():
    """Vérifie que UiState a les trois champs et est gelée."""
    ui = UiState(channel="ground", color_mode="distance", point_size=0.012)
    assert ui.channel == "ground" and ui.color_mode == "distance" and ui.point_size == 0.012
    # Vérifier que la modification échoue avec FrozenInstanceError
    with pytest.raises(dataclasses.FrozenInstanceError):
        ui.channel = "obj0"


def test_layer_isinstance_structural():
    """Vérifie que @runtime_checkable détecte structurellement les couches conformes."""
    class Good:
        """Couche valide avec tous les membres requis."""
        folder = "X"
        def setup(self, server, gui, ctx): ...
        def update(self, frame, ui): ...

    class Bad:
        """Couche invalide : pas de méthode update."""
        folder = "X"
        def setup(self, server, gui, ctx): ...
        # Pas de update

    # Good() doit passer isinstance grâce à @runtime_checkable
    assert isinstance(Good(), Layer)
    # Bad() ne doit pas passer (update manquant)
    assert not isinstance(Bad(), Layer)
