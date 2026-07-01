"""Tests de la couche contacts (roadmap #3).

Deux familles :
  1. ``contact_colors`` — fonction pure, sans écran, sans torch. On vérifie forme/dtype,
     que 'distance' produit des couleurs distinctes pour des distances différentes, et que
     'active' sépare actifs/inactifs.
  2. ``ContactsLayer.update()`` — appelé sur des fakes duck-typés (SimpleNamespace + classes
     légères). Couvre le chemin nominal ET les gardes de données manquantes (solved=None,
     targets/robot_interaction absents, canal inconnu)."""
import types

import numpy as np

from src.targets.contracts import MultiChannelField
from src.viz.core.layer import UiState
from src.viz.layers.contacts import ContactsLayer, contact_colors


# =============================================================================
# Données de test communes
# =============================================================================

def _field(C: int = 2, M: int = 4) -> MultiChannelField:
    """Champ multi-canal minimal avec des distances et activations variées."""
    dist = np.zeros((C, M))
    dist[0] = np.array([0.0, 0.02, 0.05, 0.09])   # canal 0 varie
    direction = np.zeros((C, M, 3))
    direction[..., 2] = 1.0
    witness = np.zeros((C, M, 3))
    active = np.zeros((C, M), bool)
    active[0] = np.array([True, True, False, False])
    return MultiChannelField(
        distance=dist, direction=direction, witness=witness,
        active=active, channels=tuple(f"c{i}" for i in range(C)),
    )


# =============================================================================
# 1. Tests de la fonction pure contact_colors
# =============================================================================

def test_distance_mode_shape_dtype_and_varies():
    """Mode 'distance' : forme (M, 3) uint8 + au moins deux couleurs distinctes."""
    f = _field()
    cols = contact_colors(f, channel_idx=0, mode="distance", margin=0.1)
    assert cols.shape == (4, 3) and cols.dtype == np.uint8
    # distances differentes -> au moins deux couleurs distinctes
    assert len({tuple(c) for c in cols}) >= 2


def test_active_mode_splits_active_inactive():
    """Mode 'active' : les actifs et les inactifs reçoivent des couleurs distinctes."""
    f = _field()
    cols = contact_colors(f, channel_idx=0, mode="active", margin=0.1)
    assert cols.shape == (4, 3) and cols.dtype == np.uint8
    # les 2 actifs partagent une couleur, distincte de celle des 2 inactifs
    assert tuple(cols[0]) == tuple(cols[1])
    assert tuple(cols[2]) == tuple(cols[3])
    assert tuple(cols[0]) != tuple(cols[2])


def test_uniform_mode_single_color():
    """Mode 'uniform' : tous les points reçoivent exactement la même couleur."""
    f = _field()
    cols = contact_colors(f, channel_idx=1, mode="uniform", margin=0.1)
    assert cols.shape == (4, 3) and cols.dtype == np.uint8
    assert len({tuple(c) for c in cols}) == 1


# =============================================================================
# 2. Fakes duck-typés pour les tests d'update()
# =============================================================================

class _Handle:
    """Poignée de nuage de points factice — accepte les setters de ContactsLayer."""

    def __init__(self):
        self.visible = True
        self.points = None
        self.colors = None
        self.point_size = None


class _Cb:
    """Checkbox factice."""

    def __init__(self, val: bool = True):
        self.value = val

    def on_update(self, callback) -> None:
        """No-op : les tests existants n'exercent pas le chemin on_update."""
        pass


class _FolderCtx:
    """Contexte de dossier GUI factice (gestionnaire de contexte)."""

    def __init__(self, gui):
        self._gui = gui

    def __enter__(self):
        return self._gui

    def __exit__(self, *a):
        pass


class _FakeGui:
    """GUI factice : add_folder retourne un gestionnaire de contexte ; add_checkbox retourne _Cb."""

    def __init__(self, cb_val: bool = True):
        self._cb_val = cb_val

    def add_folder(self, name: str):
        return _FolderCtx(self)

    def add_checkbox(self, label: str, default: bool = True):
        return _Cb(self._cb_val)


class _FakeScene:
    """Scene factice : add_point_cloud retourne un _Handle."""

    def add_point_cloud(self, name, pts, cols, point_size=None):
        return _Handle()


class _FakeServer:
    """Serveur viser factice."""

    def __init__(self):
        self.scene = _FakeScene()


def _make_ctx(channel_names: tuple = ("ground", "obj0")):
    """Contexte viz minimal duck-typé."""
    return types.SimpleNamespace(channel_names=channel_names, margin=0.1)


def _make_frame(*, solved: bool = True, targets: bool = True,
                robot_interaction: bool = True,
                contact_achieved: bool = True, C: int = 2, M: int = 4):
    """VizFrame factice duck-typé couvrant les combinaisons de gardes.

    Paramètres :
        solved              : si False, ``frame.solved`` est None (solve-gated).
        targets             : si False, ``frame.targets`` est None.
        robot_interaction   : si False, ``frame.targets.robot_interaction`` est None.
        contact_achieved    : si False, ``frame.solved.contact_achieved`` est None.
    """
    # Champ partagé pour cible et atteint
    tgt_field = _field(C, M)
    ach_ns = types.SimpleNamespace(field=tgt_field) if contact_achieved else None
    solved_ns = (
        types.SimpleNamespace(
            robot_points_world=np.zeros((M, 3), np.float32),
            contact_achieved=ach_ns,
        )
        if solved else None
    )
    ri_ns = types.SimpleNamespace(field=tgt_field) if robot_interaction else None
    tgts_ns = types.SimpleNamespace(robot_interaction=ri_ns) if targets else None
    return types.SimpleNamespace(solved=solved_ns, targets=tgts_ns)


def _build_layer(cb_val: bool = True, channel_names: tuple = ("ground", "obj0")) -> ContactsLayer:
    """Construit et initialise une ContactsLayer sur des fakes."""
    layer = ContactsLayer()
    layer.setup(_FakeServer(), _FakeGui(cb_val=cb_val), _make_ctx(channel_names))
    return layer


# =============================================================================
# 3. Tests d'update() — chemin nominal + gardes
# =============================================================================

def test_update_happy_path_sets_points_and_visible():
    """Chemin nominal : solved + targets + canal connu -> points/couleurs définis, visible=True."""
    layer = _build_layer(cb_val=True)
    frame = _make_frame(solved=True, targets=True)
    ui = UiState(channel="ground", color_mode="distance", point_size=0.01)

    layer.update(frame, ui)

    # Les deux nuages sont rendus visibles et ont reçu leurs points
    assert layer._h_target.visible is True
    assert layer._h_achieved.visible is True
    assert layer._h_target.points is not None
    assert layer._h_achieved.points is not None
    assert layer._h_target.colors is not None
    assert layer._h_achieved.colors is not None


def test_update_cb_false_makes_visible_false():
    """Checkbox désactivée (cb_val=False) -> visible=False sur le chemin nominal."""
    layer = _build_layer(cb_val=False)
    frame = _make_frame(solved=True, targets=True)
    ui = UiState(channel="ground", color_mode="active", point_size=0.01)

    layer.update(frame, ui)

    assert layer._h_target.visible is False
    assert layer._h_achieved.visible is False


def test_update_solved_none_hides_both():
    """solved=None (solve-gated) -> les deux nuages masqués, aucune levée."""
    layer = _build_layer()
    frame = _make_frame(solved=False)
    ui = UiState(channel="ground", color_mode="distance", point_size=0.01)

    layer.update(frame, ui)

    assert layer._h_target.visible is False
    assert layer._h_achieved.visible is False


def test_update_targets_none_hides_both():
    """targets=None -> les deux nuages masqués, aucune levée."""
    layer = _build_layer()
    frame = _make_frame(targets=False)
    ui = UiState(channel="ground", color_mode="distance", point_size=0.01)

    layer.update(frame, ui)

    assert layer._h_target.visible is False
    assert layer._h_achieved.visible is False


def test_update_robot_interaction_none_hides_both():
    """robot_interaction=None -> les deux nuages masqués, aucune levée."""
    layer = _build_layer()
    frame = _make_frame(robot_interaction=False)
    ui = UiState(channel="ground", color_mode="distance", point_size=0.01)

    layer.update(frame, ui)

    assert layer._h_target.visible is False
    assert layer._h_achieved.visible is False


def test_update_unknown_channel_hides_both():
    """Canal inconnu (UI en transition) -> masqués, aucune levée."""
    layer = _build_layer()
    frame = _make_frame(solved=True, targets=True)
    ui = UiState(channel="channel_inconnu", color_mode="distance", point_size=0.01)

    layer.update(frame, ui)

    assert layer._h_target.visible is False
    assert layer._h_achieved.visible is False


def test_update_contact_achieved_none_hides_achieved_only():
    """contact_achieved=None (résolution partielle) -> cible visible, atteint masqué."""
    layer = _build_layer(cb_val=True)
    frame = _make_frame(solved=True, targets=True, contact_achieved=False)
    ui = UiState(channel="ground", color_mode="uniform", point_size=0.01)

    layer.update(frame, ui)

    assert layer._h_target.visible is True     # cible : données présentes -> visible
    assert layer._h_achieved.visible is False  # atteint : contact_achieved absent -> masqué
