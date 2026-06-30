"""Couches minces — conformité structurelle (chacune est une Layer avec le bon dossier). La
conformité par-pixel du rendu est vérifiée par la comparaison manuelle (Task 12) ; ``update()`` est
quasi-pur assignement de poignée donc il n'y a pas de test unitaire significatif par couche (cf.
section tests du design)."""
import pytest

from src.viz.core.layer import Layer
from src.viz.layers.ghost import GhostLayer
from src.viz.layers.skeleton import SkeletonLayer


@pytest.mark.parametrize("cls, folder", [
    (GhostLayer, "Static"),
    (SkeletonLayer, "Skeleton"),
])
def test_layer_structural(cls, folder):
    """Vérifie que chaque couche est une Layer avec le dossier GUI correct."""
    layer = cls()
    assert isinstance(layer, Layer)
    assert layer.folder == folder
