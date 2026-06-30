"""Couches minces — conformité structurelle (chacune est une Layer avec le bon dossier). La
conformité par-pixel du rendu est vérifiée par la comparaison manuelle (Task 12) ; ``update()`` est
quasi-pur assignement de poignée donc il n'y a pas de test unitaire significatif par couche (cf.
section tests du design)."""
import numpy as np
import pytest

from src.viz.core.layer import Layer, UiState
from src.viz.layers.ghost import GhostLayer
from src.viz.layers.human_cloud import HumanCloudLayer
from src.viz.layers.objects import ObjectsLayer
from src.viz.layers.skeleton import SkeletonLayer


class FakeHandle:
    """Fake viser handle avec attributs modifiables."""
    def __init__(self):
        self.visible = True
        self.points = None
        self.colors = None
        self.point_size = 0.012


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

    def add_point_cloud(self, *args, **kwargs):
        """Retourne un fake handle pour le nuage de points."""
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
    channel_names = ("ground", "obj0")
    margin = 0.1
    n_objects = 1


class FakeField:
    """Fake MultiChannelField avec distance et active."""
    def __init__(self, n_channels=2, n_points=10):
        self.distance = np.random.randn(n_channels, n_points).astype(np.float32)
        self.active = np.random.rand(n_channels, n_points) > 0.5


class FakeContactEval:
    """Fake ContactEval avec distance et active."""
    def __init__(self, n_channels=2, n_points=10):
        self.distance = np.random.randn(n_channels, n_points).astype(np.float32)
        self.active = np.random.rand(n_channels, n_points) > 0.5


class FakeEnvInteraction:
    """Fake env_interaction avec per_object."""
    def __init__(self, n_objects=1, n_channels=2, n_points=10):
        self.per_object = [FakeContactEval(n_channels, n_points) for _ in range(n_objects)]


class FakeTargets:
    """Fake FrameTargets avec env_interaction."""
    def __init__(self, n_objects=1, n_channels=2, n_points=10):
        self.env_interaction = FakeEnvInteraction(n_objects, n_channels, n_points)


class FakePose:
    """Fake pose avec bone_pos optionnel."""
    def __init__(self, bone_pos=None):
        self.bone_pos = bone_pos


class FakeFrame:
    """Fake VizFrame avec pose et vertices."""
    def __init__(self, bone_pos=None, smpl_verts_world=None, human_cloud_world=None,
                 object_clouds_world=None, human_field=None, targets=None):
        self.pose = FakePose(bone_pos)
        self.smpl_verts_world = smpl_verts_world
        self.human_cloud_world = human_cloud_world if human_cloud_world is not None else np.random.randn(10, 3).astype(np.float32)
        self.object_clouds_world = object_clouds_world if object_clouds_world is not None else [np.random.randn(10, 3).astype(np.float32)]
        self.human_field = human_field if human_field is not None else FakeField(2, 10)
        self.targets = targets if targets is not None else FakeTargets(1, 2, 10)


class FakeUiState:
    """Fake UiState avec défauts pour test."""
    def __init__(self, channel="ground", color_mode="uniform", point_size=0.012):
        self.channel = channel
        self.color_mode = color_mode
        self.point_size = point_size


@pytest.mark.parametrize("cls, folder", [
    (GhostLayer, "Static"),
    (SkeletonLayer, "Skeleton"),
    (HumanCloudLayer, "Interaction - human"),
    (ObjectsLayer, "Static"),
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
