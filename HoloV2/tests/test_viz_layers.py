"""Couches minces — conformité structurelle et comportement de ``update()`` en cas de données
manquantes. Chaque layer doit exposer le dossier GUI correct ET ne jamais lever en cas de données
absentes (masquage silencieux du handle)."""
import numpy as np
import pytest

from src.viz.core.layer import Layer, UiState
from src.viz.layers.fields import FieldsLayer
from src.viz.layers.ghost import GhostLayer
from src.viz.layers.human_cloud import HumanCloudLayer
from src.viz.layers.objects import ObjectsLayer
from src.viz.layers.skeleton import SkeletonLayer
from src.viz.layers.style import StyleLayer


class FakeHandle:
    """Fake viser handle avec attributs modifiables."""
    def __init__(self):
        self.visible = True
        self.points = None
        self.colors = None
        self.point_size = 0.012
        self.position = (0.0, 0.0, 0.0)


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

    def add_label(self, *args, **kwargs):
        """Retourne un fake handle pour une étiquette texte (StyleLayer)."""
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
    style_link_names = ("link_a", "link_b")   # pour StyleLayer


class FakeField:
    """Fake MultiChannelField avec distance, active, witness et direction (pour FieldsLayer)."""
    def __init__(self, n_channels=2, n_points=10):
        self.distance = np.random.randn(n_channels, n_points).astype(np.float32)
        self.active = np.random.rand(n_channels, n_points) > 0.5
        # Requis par FieldsLayer.update() : witness/direction en local canal
        self.witness = np.random.randn(n_channels, n_points, 3).astype(np.float32)
        self.direction = np.random.randn(n_channels, n_points, 3).astype(np.float32)


class FakeContactEval:
    """Fake ContactEval avec distance et active."""
    def __init__(self, n_channels=2, n_points=10):
        self.distance = np.random.randn(n_channels, n_points).astype(np.float32)
        self.active = np.random.rand(n_channels, n_points) > 0.5


class FakeEnvInteraction:
    """Fake env_interaction avec per_object."""
    def __init__(self, n_objects=1, n_channels=2, n_points=10):
        self.per_object = [FakeContactEval(n_channels, n_points) for _ in range(n_objects)]


class FakeStyleTargets:
    """Fake StyleTargets avec position et orientation (quaternions wxyz identité)."""
    def __init__(self, n_links=2):
        self.position = np.random.randn(n_links, 3).astype(np.float32)
        # Quaternions identité wxyz (w=1, x=y=z=0) pour éviter des rotations dégénérées
        self.orientation = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_links, 1)).astype(np.float64)


class FakeTargets:
    """Fake FrameTargets avec env_interaction et style (pour StyleLayer)."""
    def __init__(self, n_objects=1, n_channels=2, n_points=10, n_links=2):
        self.env_interaction = FakeEnvInteraction(n_objects, n_channels, n_points)
        self.style = FakeStyleTargets(n_links)


class FakePose:
    """Fake pose avec bone_pos optionnel + object_rot/object_pos pour FieldsLayer."""
    def __init__(self, bone_pos=None, n_objects=1):
        self.bone_pos = bone_pos
        # Requis par FieldsLayer.update() pour la projection local objet -> monde
        self.object_rot = np.tile(np.eye(3, dtype=np.float64), (n_objects, 1, 1))
        self.object_pos = np.zeros((n_objects, 3), dtype=np.float64)


# Sentinelle pour distinguer « non fourni » de « explicitement None »
_UNSET = object()


class FakeSolvedFrame:
    """Fake SolvedFrame avec object_poses (N, 7) — pose identité par défaut (xyz=0, qw=1).

    ``style_achieved`` : StyleEval résolu ou None (None par défaut, pour les couches qui n'en
    ont pas besoin comme ObjectsLayer ; renseigné pour tester le rendu résolu de StyleLayer)."""
    def __init__(self, n_objects=1, style_achieved=None):
        # (N, 7) : [x, y, z, qw, qx, qy, qz] — quaternion identité
        poses = np.zeros((n_objects, 7), dtype=np.float64)
        poses[:, 3] = 1.0   # qw=1
        self.object_poses = poses
        self.style_achieved = style_achieved


def _fake_style_eval(n_links=2, nv=6):
    """Construit un StyleEval réel (numpy-only, importable sans écran) pour tester le rendu
    résolu de StyleLayer. Rotations = identité (colonnes = axes monde canoniques), positions
    aléatoires. jac_pos/jac_rot sont des remplissages (non lus par la couche viz)."""
    from src.targets.contracts import StyleEval
    pos = np.random.randn(n_links, 3).astype(np.float64)                    # (L, 3)
    rot = np.tile(np.eye(3, dtype=np.float64), (n_links, 1, 1))             # (L, 3, 3) identité
    return StyleEval(
        position=pos, rotation=rot,
        jac_pos=np.zeros((n_links, 3, nv)), jac_rot=np.zeros((n_links, 3, nv)),
        link_names=tuple(f"link_{chr(ord('a') + i)}" for i in range(n_links)))


class FakeCheckboxCapture:
    """Fake checkbox qui capture les callbacks on_update (pour tests paused-toggle)."""
    def __init__(self, initial_value=True):
        self.value = initial_value
        self._callbacks = []

    def on_update(self, callback):
        """Capture le callback au lieu de l'ignorer."""
        self._callbacks.append(callback)

    def fire(self, event=None):
        """Déclenche tous les callbacks enregistrés (simule un changement de valeur GUI)."""
        for cb in self._callbacks:
            cb(event)


class FakeGuiCapturing:
    """Fake GUI qui retourne des FakeCheckboxCapture respectant la valeur initiale passée.
    Permet de capturer les callbacks on_update et de tester le paused-toggle."""
    def __init__(self):
        self.checkboxes = []

    def add_checkbox(self, *args, **kwargs):
        """Crée un FakeCheckboxCapture avec la valeur initiale passée en 2ème arg positionnel."""
        initial = args[1] if len(args) > 1 else kwargs.get("initial", True)
        cb = FakeCheckboxCapture(initial_value=initial)
        self.checkboxes.append(cb)
        return cb


class FakeFrame:
    """Fake VizFrame avec pose et vertices. Passer ``None`` à ``human_field`` ou ``targets``
    produit vraiment ``None`` (utile pour tester les gardes no-op) ; omettre le paramètre
    fournit des données valides par défaut.

    ``solved`` : FakeSolvedFrame ou None (None par défaut, simule l'absence de solve).
    ``pose``   : FakePose ou None ; si omis, construit FakePose(bone_pos).  Passer
                 explicitement None permet de tester les gardes ``frame.pose is None``."""
    def __init__(self, bone_pos=None, smpl_verts_world=None, human_cloud_world=_UNSET,
                 object_clouds_world=_UNSET, human_field=_UNSET, targets=_UNSET,
                 solved=None, pose=_UNSET):
        self.pose = FakePose(bone_pos) if pose is _UNSET else pose
        self.smpl_verts_world = smpl_verts_world
        self.human_cloud_world = (np.random.randn(10, 3).astype(np.float32)
                                  if human_cloud_world is _UNSET else human_cloud_world)
        self.object_clouds_world = ([np.random.randn(10, 3).astype(np.float32)]
                                    if object_clouds_world is _UNSET else object_clouds_world)
        self.human_field = FakeField(2, 10) if human_field is _UNSET else human_field
        self.targets = FakeTargets(1, 2, 10) if targets is _UNSET else targets
        self.solved = solved


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
    (FieldsLayer, "Interaction - human"),
    (StyleLayer, "Style targets"),
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


# ---------------------------------------------------------------------------
# FieldsLayer — gardes no-op + chemin nominal
# ---------------------------------------------------------------------------

def test_fields_update_human_field_none():
    """Garde no-op : human_field=None → update() ne lève pas ET masque les deux handles."""
    layer = FieldsLayer()
    layer.setup(FakeServer(), FakeGui(), FakeContext())

    frame = FakeFrame(human_field=None)
    layer.update(frame, FakeUiState())

    assert layer._wit.visible == False
    assert layer._nrm.visible == False


def test_fields_update_unknown_channel():
    """Garde no-op : canal inconnu → update() ne lève pas (pas de ValueError) ET masque les handles."""
    layer = FieldsLayer()
    layer.setup(FakeServer(), FakeGui(), FakeContext())

    frame = FakeFrame()
    ui = FakeUiState(channel="canal_inexistant")
    layer.update(frame, ui)

    assert layer._wit.visible == False
    assert layer._nrm.visible == False


def test_fields_update_happy_path_ground():
    """Chemin nominal canal ground : sondes actives forcées → segments mis à jour, handles visibles.
    Canal 0 = ground, witness/direction déjà en monde (pas de transformation locale->monde)."""
    layer = FieldsLayer()
    layer.setup(FakeServer(), FakeGui(), FakeContext())

    # Forcer quelques sondes actives pour garantir que len(idx) > 0
    rng = np.random.default_rng(42)
    n_points = 10
    field = FakeField(n_channels=2, n_points=n_points)
    field.active[0, :5] = True   # 5 sondes actives sur le canal ground

    pts_world = rng.random((n_points, 3)).astype(np.float32)
    frame = FakeFrame(human_cloud_world=pts_world, human_field=field)
    ui = FakeUiState(channel="ground")
    layer.update(frame, ui)

    # FakeCheckbox.value = True → les deux handles doivent être visibles et avoir des points
    assert layer._wit.points is not None
    assert layer._nrm.points is not None
    assert layer._wit.visible == True
    assert layer._nrm.visible == True


def test_fields_update_happy_path_object_channel():
    """Chemin nominal canal objet (c=1) : witness/direction projetés local->monde via (R, t)."""
    layer = FieldsLayer()
    layer.setup(FakeServer(), FakeGui(), FakeContext())

    rng = np.random.default_rng(7)
    n_points = 10
    field = FakeField(n_channels=2, n_points=n_points)
    field.active[1, :8] = True   # sondes actives sur le canal obj0

    pts_world = rng.random((n_points, 3)).astype(np.float32)
    frame = FakeFrame(human_cloud_world=pts_world, human_field=field)
    ui = FakeUiState(channel="obj0")
    layer.update(frame, ui)

    # Doit avoir mis à jour les segments sans lever
    assert layer._wit.points is not None
    assert layer._wit.visible == True


# ---------------------------------------------------------------------------
# StyleLayer — gardes no-op + chemin nominal
# ---------------------------------------------------------------------------

def test_style_update_targets_none():
    """Garde no-op : targets=None → update() ne lève pas ET masque points + repères + étiquettes."""
    layer = StyleLayer()
    layer.setup(FakeServer(), FakeGui(), FakeContext())

    frame = FakeFrame(targets=None)
    layer.update(frame, FakeUiState())

    assert layer._pts.visible == False
    assert layer._frames.visible == False
    for h in layer._labels:
        assert h.visible == False


def test_style_update_happy_path():
    """Chemin nominal : style.position valide → points mis à jour et visibles ; repères visibles
    avec orientation non-None ; étiquettes positionnées (cb_l=True dans FakeCheckbox)."""
    layer = StyleLayer()
    layer.setup(FakeServer(), FakeGui(), FakeContext())

    # FakeTargets inclut FakeStyleTargets (2 liens, quaternions identité)
    frame = FakeFrame(targets=FakeTargets(n_objects=1, n_channels=2, n_points=10, n_links=2))
    ui = FakeUiState()
    layer.update(frame, ui)

    # Points de position mis à jour et handle visible (FakeCheckbox.value = True)
    assert layer._pts.points is not None
    assert layer._pts.visible == True
    # Repères d'orientation calculés (orientation non-None + cb_f=True)
    assert layer._frames.points is not None
    assert layer._frames.visible == True
    # Étiquettes positionnées
    for h in layer._labels:
        assert h.position is not None


def test_style_update_style_none():
    """Garde no-op : frame.targets.style=None → masquer tous les handles sans lever."""
    layer = StyleLayer()
    layer.setup(FakeServer(), FakeGui(), FakeContext())

    # Construire un fake targets sans attribut style valide
    class FakeTargetsNoStyle:
        style = None
        env_interaction = FakeEnvInteraction(1, 2, 10)

    frame = FakeFrame(targets=FakeTargetsNoStyle())
    layer.update(frame, FakeUiState())

    assert layer._pts.visible == False
    assert layer._frames.visible == False
    for h in layer._labels:
        assert h.visible == False


# ---------------------------------------------------------------------------
# StyleLayer — cibles RÉSOLUES (solve-gaté, toggle repères, paused-toggle)
# ---------------------------------------------------------------------------

def test_style_solved_drawn_on_happy_frame():
    """Chemin nominal résolu : solved.style_achieved peuplé + toggles ON (FakeCheckbox=True)
    → points résolus == se.position, verts, visibles ; repères résolus = 3L segments visibles."""
    layer = StyleLayer()
    layer.setup(FakeServer(), FakeGui(), FakeContext())
    # FakeCheckbox.value = True → _cb_p_sol et _cb_f_sol valent True

    se = _fake_style_eval(n_links=2)
    frame = FakeFrame(
        targets=FakeTargets(n_objects=1, n_channels=2, n_points=10, n_links=2),
        solved=FakeSolvedFrame(n_objects=1, style_achieved=se),
    )
    layer.update(frame, FakeUiState())

    # Points résolus : positions == se.position, visibles, verts distincts de l'orange de réf
    assert np.allclose(layer._pts_sol.points, se.position)
    assert layer._pts_sol.visible == True
    assert not np.array_equal(layer._pts.colors[0], layer._pts_sol.colors[0])
    # Repères résolus : 3 axes × L liens = 3L segments (chacun (2, 3)), visibles
    assert layer._frames_sol.points.shape == (3 * 2, 2, 3)
    assert layer._frames_sol.visible == True
    # La référence reste rendue normalement (comportement historique intact)
    assert layer._pts.points is not None
    assert layer._pts.visible == True


def test_style_solved_hidden_when_solved_none():
    """Solve-gaté : frame.solved=None → handles résolus masqués, référence inchangée, sans lever."""
    layer = StyleLayer()
    layer.setup(FakeServer(), FakeGui(), FakeContext())

    frame = FakeFrame(
        targets=FakeTargets(n_objects=1, n_channels=2, n_points=10, n_links=2),
        solved=None,
    )
    layer.update(frame, FakeUiState())

    # Résolu masqué
    assert layer._pts_sol.visible == False
    assert layer._frames_sol.visible == False
    # Référence intacte
    assert layer._pts.points is not None
    assert layer._pts.visible == True
    assert layer._frames.visible == True


def test_style_solved_hidden_when_style_achieved_none():
    """Solve-gaté : solved présent mais style_achieved=None → handles résolus masqués sans lever."""
    layer = StyleLayer()
    layer.setup(FakeServer(), FakeGui(), FakeContext())

    frame = FakeFrame(
        targets=FakeTargets(n_objects=1, n_channels=2, n_points=10, n_links=2),
        solved=FakeSolvedFrame(n_objects=1, style_achieved=None),
    )
    layer.update(frame, FakeUiState())

    assert layer._pts_sol.visible == False
    assert layer._frames_sol.visible == False


def test_style_solved_paused_toggle_redraws_frames():
    """Paused-toggle : les repères résolus sont OFF par défaut (_cb_f_sol initial=False) ;
    leur activation en pause re-dessine les repères résolus sans nudge du slider."""
    gui = FakeGuiCapturing()
    layer = StyleLayer()
    layer.setup(FakeServer(), gui, FakeContext())
    # Ordre des cases : 0=_cb_p, 1=_cb_f, 2=_cb_l, 3=_cb_p_sol, 4=_cb_f_sol
    assert gui.checkboxes[4].value is False   # repères résolus OFF par défaut

    se = _fake_style_eval(n_links=2)
    frame = FakeFrame(
        targets=FakeTargets(n_objects=1, n_channels=2, n_points=10, n_links=2),
        solved=FakeSolvedFrame(n_objects=1, style_achieved=se),
    )
    ui = FakeUiState()

    # 1ère update : _cb_f_sol=False → repères résolus masqués ; points résolus visibles (_cb_p_sol=True)
    layer.update(frame, ui)
    assert layer._frames_sol.visible == False
    assert layer._pts_sol.visible == True

    # Activation du toggle repères résolus EN PAUSE (sans nudge du slider)
    gui.checkboxes[4].value = True
    gui.checkboxes[4].fire(None)   # déclenche _on_change → re-invoque update()

    # Les repères résolus doivent maintenant être visibles sans nouvel appel manuel à update()
    assert layer._frames_sol.visible == True
    assert layer._frames_sol.points.shape == (3 * 2, 2, 3)


# ---------------------------------------------------------------------------
# ObjectsLayer — cloud résolu (solve-gaté, toggle, gardes, paused-toggle)
# ---------------------------------------------------------------------------

def test_objects_solved_cloud_drawn_on_happy_frame():
    """Chemin nominal résolu : solved présent + toggle activé → cloud résolu visible, vert,
    distinct du cloud source orange (couleur distincte par inspection du 1er pixel)."""
    layer = ObjectsLayer()
    layer.setup(FakeServer(), FakeGui(), FakeContext())
    # FakeCheckbox.value = True par défaut → _cb_solved.value = True

    pts = np.ones((10, 3), np.float32)
    frame = FakeFrame(
        object_clouds_world=[pts],
        targets=FakeTargets(n_objects=1, n_channels=2, n_points=10),
        solved=FakeSolvedFrame(n_objects=1),
    )
    ui = FakeUiState(channel="ground", color_mode="uniform")
    layer.update(frame, ui)

    # Cloud résolu : handle renseigné et visible
    assert layer._handles_sol[0].visible == True
    assert layer._handles_sol[0].points is not None
    assert layer._handles_sol[0].colors is not None
    # Couleur résolue distincte de la couleur source (vert vs orange)
    assert not np.array_equal(layer._handles[0].colors[0], layer._handles_sol[0].colors[0])


def test_objects_solved_cloud_hidden_when_solved_none():
    """Solve-gaté : frame.solved=None → cloud résolu masqué sans lever."""
    layer = ObjectsLayer()
    layer.setup(FakeServer(), FakeGui(), FakeContext())

    pts = np.ones((10, 3), np.float32)
    frame = FakeFrame(
        object_clouds_world=[pts],
        targets=FakeTargets(n_objects=1, n_channels=2, n_points=10),
        solved=None,
    )
    layer.update(frame, FakeUiState(channel="ground"))

    assert layer._handles_sol[0].visible == False


def test_objects_solved_cloud_hidden_when_toggle_off():
    """Toggle off : _cb_solved.value=False → cloud résolu masqué même si solved présent."""
    layer = ObjectsLayer()
    layer.setup(FakeServer(), FakeGui(), FakeContext())
    layer._cb_solved.value = False   # désactiver manuellement le toggle résolu

    pts = np.ones((10, 3), np.float32)
    frame = FakeFrame(
        object_clouds_world=[pts],
        targets=FakeTargets(n_objects=1, n_channels=2, n_points=10),
        solved=FakeSolvedFrame(n_objects=1),
    )
    layer.update(frame, FakeUiState(channel="ground"))

    assert layer._handles_sol[0].visible == False


def test_objects_solved_cloud_guard_pose_none():
    """Garde : frame.pose=None → cloud résolu masqué, source cloud inchangé, pas de levée."""
    layer = ObjectsLayer()
    layer.setup(FakeServer(), FakeGui(), FakeContext())
    # _cb_solved.value = True (FakeCheckbox par défaut)

    pts = np.ones((10, 3), np.float32)
    frame = FakeFrame(
        object_clouds_world=[pts],
        targets=FakeTargets(n_objects=1, n_channels=2, n_points=10),
        solved=FakeSolvedFrame(n_objects=1),
        pose=None,
    )
    layer.update(frame, FakeUiState(channel="ground", color_mode="uniform"))

    # Cloud résolu doit être masqué (pas de pose pour calculer la transformation)
    assert layer._handles_sol[0].visible == False


def test_objects_solved_cloud_guard_fewer_object_poses():
    """Garde bornes : solved.object_poses plus court que _handles_sol → surplus masqué sans IndexError."""
    layer = ObjectsLayer()
    layer.setup(FakeServer(), FakeGui(), FakeContextTwoObjects())
    # _cb_solved.value = True (FakeCheckbox par défaut)

    pts0 = np.ones((10, 3), np.float32)
    pts1 = np.ones((10, 3), np.float32)
    frame = FakeFrame(
        object_clouds_world=[pts0, pts1],
        targets=FakeTargets(n_objects=2, n_channels=2, n_points=10),
        solved=FakeSolvedFrame(n_objects=1),   # 1 seule pose pour 2 objets
        pose=FakePose(n_objects=2),
    )
    layer.update(frame, FakeUiState(channel="ground", color_mode="uniform"))

    # 1er objet : pose résolue disponible (k=0 < 1)
    assert layer._handles_sol[0].visible == True
    # 2ème objet : k=1 >= len(solved.object_poses)=1 → masqué
    assert layer._handles_sol[1].visible == False


def test_objects_paused_toggle_redraws_solved_cloud():
    """Paused-toggle : activation de _cb_solved en pause re-dessine le cloud résolu
    sans nudge du slider (via _last_frame/_last_ui mémorisés)."""
    gui = FakeGuiCapturing()
    layer = ObjectsLayer()
    layer.setup(FakeServer(), gui, FakeContext())
    # gui.checkboxes[0] = _cb (value=True), gui.checkboxes[1] = _cb_solved (value=False)

    pts = np.ones((10, 3), np.float32)
    frame = FakeFrame(
        object_clouds_world=[pts],
        targets=FakeTargets(n_objects=1, n_channels=2, n_points=10),
        solved=FakeSolvedFrame(n_objects=1),
    )
    ui = FakeUiState(channel="ground")

    # 1ère update : _cb_solved.value = False → cloud résolu masqué
    layer.update(frame, ui)
    assert layer._handles_sol[0].visible == False

    # Simule l'activation du toggle résolu EN PAUSE (sans nudge du slider)
    gui.checkboxes[1].value = True
    gui.checkboxes[1].fire(None)   # déclenche _on_change → re-invoque update()

    # Le cloud résolu doit maintenant être visible sans nouvel appel à update()
    assert layer._handles_sol[0].visible == True
