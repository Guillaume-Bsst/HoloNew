"""Tests de la couche correspondance (roadmap #4).

Deux familles :
  1. ``correspondence_segments`` — fonction pure, sans écran. Indices connus → segments connus
     (extrémité humaine = human[smpl_idx[m]], extrémité robot = robot[m]) ; vide si M=0 ;
     lève sur indice hors-plage.
  2. ``CorrespondenceLayer.update()`` — appelé sur des fakes duck-typés. Couvre le chemin
     nominal (segments définis + visible=cb.value) et les gardes de données manquantes
     (solved=None, nuage humain=None, checkbox désactivée → masquage silencieux sans levée)."""
import types

import numpy as np
import pytest

from src.viz.core.layer import UiState
from src.viz.layers.correspondence import CorrespondenceLayer, correspondence_segments


# =============================================================================
# 1. Tests de la fonction pure correspondence_segments
# =============================================================================

def test_segments_pick_paired_endpoints():
    """Indices connus → extremités exactes : human[smpl_idx[m]] et robot[m]."""
    human = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [3., 0., 0.]])   # N=4
    robot = np.array([[0., 1., 0.], [0., 2., 0.]])                                # M=2
    smpl_idx = np.array([2, 0])           # robot0 <-> human2 ; robot1 <-> human0
    seg = correspondence_segments(human, robot, smpl_idx)
    assert seg.shape == (2, 2, 3) and seg.dtype == np.float32
    assert np.allclose(seg[0, 0], [2., 0., 0.]) and np.allclose(seg[0, 1], [0., 1., 0.])
    assert np.allclose(seg[1, 0], [0., 0., 0.]) and np.allclose(seg[1, 1], [0., 2., 0.])


def test_empty_when_no_points():
    """M=0 : tableau vide de forme (0, 2, 3)."""
    seg = correspondence_segments(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0,), np.int64))
    assert seg.shape == (0, 2, 3)


def test_out_of_range_index_raises():
    """smpl_idx hors plage → ValueError ou IndexError."""
    human = np.zeros((2, 3)); robot = np.zeros((1, 3)); smpl_idx = np.array([5])   # 5 >= N=2
    with pytest.raises((IndexError, ValueError)):
        correspondence_segments(human, robot, smpl_idx)


# =============================================================================
# 2. Fakes duck-typés pour les tests d'update()
# =============================================================================

class _Handle:
    """Poignée de segments factice — accepte les setters de CorrespondenceLayer."""

    def __init__(self):
        self.visible = True
        self.points = None
        self.colors = None


class _Cb:
    """Checkbox factice."""

    def __init__(self, val: bool = True):
        self.value = val


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
    """Scène factice : add_line_segments retourne un _Handle."""

    def add_line_segments(self, name, segs, cols, line_width=None):
        return _Handle()


class _FakeServer:
    """Serveur viser factice."""

    def __init__(self):
        self.scene = _FakeScene()


def _make_ctx(smpl_idx=None):
    """Contexte viz minimal duck-typé avec un CorrespondenceTable factice."""
    if smpl_idx is None:
        smpl_idx = np.array([0, 1])
    correspondence = types.SimpleNamespace(smpl_idx=smpl_idx)
    return types.SimpleNamespace(correspondence=correspondence)


def _make_frame(*, solved: bool = True, human_cloud: bool = True, M: int = 2):
    """VizFrame factice duck-typé couvrant les gardes de CorrespondenceLayer.

    Paramètres :
        solved         : si False, ``frame.solved`` est None (couche solve-gated).
        human_cloud    : si False, ``frame.human_cloud_world`` est None.
        M              : nombre de points de correspondance.
    """
    solved_ns = (
        types.SimpleNamespace(robot_points_world=np.zeros((M, 3), np.float32))
        if solved else None
    )
    cloud = np.zeros((4, 3), np.float32) if human_cloud else None
    return types.SimpleNamespace(solved=solved_ns, human_cloud_world=cloud)


def _build_layer(cb_val: bool = True, smpl_idx=None) -> CorrespondenceLayer:
    """Construit et initialise une CorrespondenceLayer sur des fakes."""
    layer = CorrespondenceLayer()
    layer.setup(_FakeServer(), _FakeGui(cb_val=cb_val), _make_ctx(smpl_idx))
    return layer


# =============================================================================
# 3. Tests d'update() — chemin nominal + gardes
# =============================================================================

_UI = UiState(channel="ground", color_mode="uniform", point_size=0.012)


def test_update_happy_path_sets_segments_and_visible():
    """Chemin nominal : solved + nuage humain + cb=True → segments/couleurs définis, visible=True."""
    # smpl_idx=[0,1] : robot[0]<->human[0], robot[1]<->human[1] ; M=2, N=4
    layer = _build_layer(cb_val=True, smpl_idx=np.array([0, 1]))
    frame = _make_frame(solved=True, human_cloud=True, M=2)

    layer.update(frame, _UI)

    assert layer._h.visible is True
    assert layer._h.points is not None
    assert layer._h.colors is not None
    assert layer._h.points.shape == (2, 2, 3)   # (M, 2, 3)


def test_update_visible_equals_cb_value():
    """visible est bien self._cb.value (True ici) et non une constante True indépendante."""
    layer = _build_layer(cb_val=True, smpl_idx=np.array([0, 1]))
    frame = _make_frame(solved=True, human_cloud=True, M=2)
    layer.update(frame, _UI)
    assert layer._h.visible == layer._cb.value


def test_update_solved_none_hides_no_raise():
    """solved=None (solve-gated) → handle masqué, aucune levée."""
    layer = _build_layer(cb_val=True, smpl_idx=np.array([0, 1]))
    frame = _make_frame(solved=False)

    layer.update(frame, _UI)

    assert layer._h.visible is False


def test_update_human_cloud_none_hides_no_raise():
    """human_cloud_world=None → handle masqué, aucune levée."""
    layer = _build_layer(cb_val=True, smpl_idx=np.array([0, 1]))
    frame = _make_frame(solved=True, human_cloud=False, M=2)

    layer.update(frame, _UI)

    assert layer._h.visible is False


def test_update_cb_false_hides_no_raise():
    """Checkbox désactivée → handle masqué, aucune levée (même si données valides)."""
    layer = _build_layer(cb_val=False, smpl_idx=np.array([0, 1]))
    frame = _make_frame(solved=True, human_cloud=True, M=2)

    layer.update(frame, _UI)

    assert layer._h.visible is False
