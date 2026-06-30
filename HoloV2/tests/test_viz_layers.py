"""Couches minces — conformité structurelle (chacune est une Layer avec le bon dossier). La
conformité par-pixel du rendu est vérifiée par la comparaison manuelle (Task 12) ; ``update()`` est
quasi-pur assignement de poignée donc il n'y a pas de test unitaire significatif par couche (cf.
section tests du design)."""
import numpy as np
import pytest

from src.viz.core.layer import Layer, UiState
from src.viz.layers.ghost import GhostLayer
from src.viz.layers.skeleton import SkeletonLayer


class FakeHandle:
    """Fake viser handle avec attributs modifiables."""
    def __init__(self):
        self.visible = True
        self.points = None
        self.colors = None


class FakeCheckbox:
    """Fake GUI checkbox."""
    def __init__(self):
        self.value = True

    def on_update(self, callback):
        """Enregistre le callback (pas d'implémentation pour test)."""
        pass


class FakeScene:
    """Fake viser scene."""
    def add_line_segments(self, *args, **kwargs):
        """Retourne un fake handle pour les segments."""
        return FakeHandle()

    def add_mesh_simple(self, *args, **kwargs):
        """Retourne un fake handle pour le mesh."""
        return FakeHandle()


class FakeServer:
    """Fake viser server."""
    def __init__(self):
        self.scene = FakeScene()


class FakeGui:
    """Fake viser GUI."""
    def add_checkbox(self, *args, **kwargs):
        """Retourne un fake checkbox."""
        return FakeCheckbox()


class FakeContext:
    """Fake VizContext avec données minimales."""
    smpl_parents = np.array([0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 7, 8, 9, 10, 11, 12, 13, 13, 14, 15, 16, 17, 17, 18], dtype=np.int32)
    smpl_faces = np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int32)


class FakePose:
    """Fake pose avec bone_pos optionnel."""
    def __init__(self, bone_pos=None):
        self.bone_pos = bone_pos


class FakeFrame:
    """Fake VizFrame avec pose et vertices."""
    def __init__(self, bone_pos=None, smpl_verts_world=None):
        self.pose = FakePose(bone_pos)
        self.smpl_verts_world = smpl_verts_world


class FakeUiState:
    """Fake UiState (vide pour test)."""
    pass


@pytest.mark.parametrize("cls, folder", [
    (GhostLayer, "Static"),
    (SkeletonLayer, "Skeleton"),
])
def test_layer_structural(cls, folder):
    """Vérifie que chaque couche est une Layer avec le dossier GUI correct."""
    layer = cls()
    assert isinstance(layer, Layer)
    assert layer.folder == folder


def test_skeleton_layer_update_with_none_bone_pos():
    """Vérifie que SkeletonLayer.update() ne plante pas quand bone_pos est None et masque le handle."""
    layer = SkeletonLayer()
    layer.setup(FakeServer(), FakeGui(), FakeContext())

    # update() avec bone_pos=None ne doit pas planter
    frame = FakeFrame(bone_pos=None)
    layer.update(frame, FakeUiState())

    # Le handle doit être caché
    assert layer._handle.visible == False


def test_skeleton_layer_update_with_valid_bone_pos():
    """Vérifie que SkeletonLayer.update() fonctionne avec des données valides."""
    layer = SkeletonLayer()
    layer.setup(FakeServer(), FakeGui(), FakeContext())

    # Créer des données bone_pos valides (24 joints SMPL)
    bone_pos = np.random.randn(24, 3).astype(np.float32)
    frame = FakeFrame(bone_pos=bone_pos)
    layer.update(frame, FakeUiState())

    # Le handle doit avoir les points mis à jour et être visible
    assert layer._handle.points is not None
    assert layer._handle.visible == True


def test_ghost_layer_update_with_none_verts():
    """Vérifie que GhostLayer.update() masque le handle quand smpl_verts_world est None."""
    layer = GhostLayer()
    layer.setup(FakeServer(), FakeGui(), FakeContext())

    # Première fois : pas encore de handle
    frame = FakeFrame(smpl_verts_world=None)
    layer.update(frame, FakeUiState())

    # Pas d'exception et handle reste None
    assert layer._handle is None

    # Simuler un handle existant et appeler update() avec None data
    layer._handle = FakeHandle()
    layer._handle.visible = True
    layer.update(frame, FakeUiState())

    # Le handle doit être caché
    assert layer._handle.visible == False


def test_ghost_layer_update_with_valid_verts():
    """Vérifie que GhostLayer.update() fonctionne avec des données valides."""
    layer = GhostLayer()
    layer.setup(FakeServer(), FakeGui(), FakeContext())

    # Créer des données vertices valides (6 vertices pour un simple mesh)
    verts = np.random.randn(6, 3).astype(np.float32)
    frame = FakeFrame(smpl_verts_world=verts)
    layer.update(frame, FakeUiState())

    # Le handle doit être créé et visible
    assert layer._handle is not None
    assert layer._handle.visible == True
