"""Couches minces — conformité structurelle et comportement de ``update()`` en cas de données
manquantes. Chaque layer doit exposer le dossier GUI correct ET ne jamais lever en cas de données
absentes (masquage silencieux du handle)."""
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


# Sentinelle pour distinguer « non fourni » de « explicitement None »
_UNSET = object()


class FakeFrame:
    """Fake VizFrame avec pose et vertices. Passer ``None`` à ``human_field`` ou ``targets``
    produit vraiment ``None`` (utile pour tester les gardes no-op) ; omettre le paramètre
    fournit des données valides par défaut."""
    def __init__(self, bone_pos=None, smpl_verts_world=None, human_cloud_world=_UNSET,
                 object_clouds_world=_UNSET, human_field=_UNSET, targets=_UNSET):
        self.pose = FakePose(bone_pos)
        self.smpl_verts_world = smpl_verts_world
        self.human_cloud_world = (np.random.randn(10, 3).astype(np.float32)
                                  if human_cloud_world is _UNSET else human_cloud_world)
        self.object_clouds_world = ([np.random.randn(10, 3).astype(np.float32)]
                                    if object_clouds_world is _UNSET else object_clouds_world)
        self.human_field = FakeField(2, 10) if human_field is _UNSET else human_field
        self.targets = FakeTargets(1, 2, 10) if targets is _UNSET else targets


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


# ---------------------------------------------------------------------------
# HumanCloudLayer — gardes no-op + chemin nominal
# ---------------------------------------------------------------------------

def test_human_cloud_update_human_field_none():
    """Garde no-op : human_field=None → update() ne lève pas ET masque le handle."""
    layer = HumanCloudLayer()
    layer.setup(FakeServer(), FakeGui(), FakeContext())

    frame = FakeFrame(human_field=None)
    layer.update(frame, FakeUiState())

    assert layer._handle.visible == False


def test_human_cloud_update_human_cloud_none():
    """Garde no-op : human_cloud_world=None → update() ne lève pas ET masque le handle."""
    layer = HumanCloudLayer()
    layer.setup(FakeServer(), FakeGui(), FakeContext())

    frame = FakeFrame(human_cloud_world=None)
    layer.update(frame, FakeUiState())

    assert layer._handle.visible == False


def test_human_cloud_update_unknown_channel():
    """Garde no-op : canal inconnu → update() ne lève pas (pas de ValueError) ET masque le handle."""
    layer = HumanCloudLayer()
    layer.setup(FakeServer(), FakeGui(), FakeContext())

    frame = FakeFrame()
    ui = FakeUiState(channel="canal_inexistant")
    layer.update(frame, ui)

    assert layer._handle.visible == False


def test_human_cloud_update_happy_path():
    """Chemin nominal : données valides + canal connu → points/colors mis à jour, handle visible."""
    layer = HumanCloudLayer()
    layer.setup(FakeServer(), FakeGui(), FakeContext())

    pts = np.random.randn(10, 3).astype(np.float32)
    frame = FakeFrame(human_cloud_world=pts)
    ui = FakeUiState(channel="ground", color_mode="uniform")
    layer.update(frame, ui)

    assert layer._handle.points is not None
    assert layer._handle.colors is not None
    assert layer._handle.visible == True


# ---------------------------------------------------------------------------
# ObjectsLayer — gardes no-op + chemin nominal
# ---------------------------------------------------------------------------

def test_objects_update_targets_none():
    """Garde no-op : targets=None → update() ne lève pas ET masque tous les handles."""
    layer = ObjectsLayer()
    layer.setup(FakeServer(), FakeGui(), FakeContext())

    frame = FakeFrame(targets=None)
    layer.update(frame, FakeUiState())

    for h in layer._handles:
        assert h.visible == False


def test_objects_update_object_clouds_none():
    """Garde no-op : object_clouds_world=None → update() ne lève pas ET masque tous les handles."""
    layer = ObjectsLayer()
    layer.setup(FakeServer(), FakeGui(), FakeContext())

    frame = FakeFrame(object_clouds_world=None)
    layer.update(frame, FakeUiState())

    for h in layer._handles:
        assert h.visible == False


def test_objects_update_unknown_channel():
    """Garde no-op : canal inconnu → update() ne lève pas ET masque tous les handles."""
    layer = ObjectsLayer()
    layer.setup(FakeServer(), FakeGui(), FakeContext())

    frame = FakeFrame()
    ui = FakeUiState(channel="canal_inexistant")
    layer.update(frame, ui)

    for h in layer._handles:
        assert h.visible == False


class FakeContextTwoObjects(FakeContext):
    """Contexte avec 2 objets pour tester les gardes d'index."""
    n_objects = 2


def test_objects_update_fewer_clouds_than_handles():
    """Garde d'index : moins de nuages que de handles → pas d'IndexError, handles excédentaires masqués."""
    # Contexte avec 2 handles mais seulement 1 nuage dans le frame
    layer = ObjectsLayer()
    layer.setup(FakeServer(), FakeGui(), FakeContextTwoObjects())

    # Un seul nuage pour 2 handles
    single_cloud = [np.random.randn(10, 3).astype(np.float32)]
    frame = FakeFrame(
        object_clouds_world=single_cloud,
        targets=FakeTargets(n_objects=1, n_channels=2, n_points=10),
    )
    layer.update(frame, FakeUiState())

    # Le premier handle doit avoir des données, le second doit être masqué
    assert layer._handles[0].points is not None
    assert layer._handles[1].visible == False


def test_objects_update_happy_path():
    """Chemin nominal : données valides + canal connu → points/colors mis à jour, handle visible."""
    layer = ObjectsLayer()
    layer.setup(FakeServer(), FakeGui(), FakeContext())

    pts = np.random.randn(10, 3).astype(np.float32)
    frame = FakeFrame(
        object_clouds_world=[pts],
        targets=FakeTargets(n_objects=1, n_channels=2, n_points=10),
    )
    ui = FakeUiState(channel="ground", color_mode="uniform")
    layer.update(frame, ui)

    assert layer._handles[0].points is not None
    assert layer._handles[0].colors is not None
    assert layer._handles[0].visible == True
