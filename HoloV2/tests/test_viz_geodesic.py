"""Tests de la couche geodesic (roadmap #7).

Deux familles :
  1. Fonctions pures ``geo_normalized`` / ``geo_heat_colors`` — numpy-only, sans écran.
     Source (distance 0) -> 0 ; max -> 1 ; formes/dtype vérifiés ; ligne constante -> 0.
  2. ``GeodesicLayer.update()`` — appelé sur des fakes duck-typés.
     Couvre :
     - chemin nominal canal objet (geodesic + pose valide) : nuage visible, couleurs définies ;
     - chemin nominal canal ground (object_idx=None, world coords) : visible sans pose ;
     - canal sans GeodesicTable (geodesic=None) : ignoré au setup, aucun handle, aucun crash ;
     - show_pts=False, show_nrm=False -> les deux handles masqués ;
     - ``frame.pose=None`` sur un canal objet -> masqué, aucune levée ;
     - ``object_idx`` hors bornes -> masqué, aucune levée ;
     - ensemble de points vide (n_points=0) -> masqué, aucune levée.
"""
import types

import numpy as np

from src.prepare.contracts import GeodesicTable
from src.viz.core.layer import UiState
from src.viz.layers.geodesic import GeodesicLayer, geo_heat_colors, geo_normalized


# =============================================================================
# 1. Tests des fonctions pures
# =============================================================================

def test_normalized_source_zero_max_one():
    """Source (distance 0) -> 0 ; point le plus loin -> 1 ; valeurs dans [0,1]."""
    row = np.array([0.0, 1.0, 2.0, 4.0])              # geo[src] : src lui-même = 0
    n = geo_normalized(row)
    assert n.shape == (4,)
    assert np.isclose(n[0], 0.0) and np.isclose(n[3], 1.0)
    assert np.all((n >= 0.0) & (n <= 1.0))


def test_normalized_constant_row_is_zero():
    """Ligne constante (max ~ 0) -> tout 0 (pas de division par zéro)."""
    n = geo_normalized(np.zeros(5))
    assert np.allclose(n, 0.0)


def test_heat_colors_shape_dtype():
    """geo_heat_colors : (P,3) uint8 quelle que soit la valeur max de la ligne."""
    cols = geo_heat_colors(np.array([0.0, 0.5, 1.0, 2.0]))
    assert cols.shape == (4, 3) and cols.dtype == np.uint8


# =============================================================================
# 2. Fakes duck-typés pour les tests d'update()
# =============================================================================

class _PtsHandle:
    """Poignée de nuage de points factice."""

    def __init__(self) -> None:
        self.visible: bool = True
        self.points = None
        self.colors = None


class _NrmHandle:
    """Poignée de segments factice (normales)."""

    def __init__(self) -> None:
        self.visible: bool = True
        self.points = None
        self.colors = None


class _Cb:
    """Checkbox factice."""

    def __init__(self, val: bool = True) -> None:
        self.value = val

    def on_update(self, callback) -> None:
        """No-op : les tests existants n'exercent pas le chemin on_update."""
        pass


class _Slider:
    """Curseur entier factice."""

    def __init__(self, val: int = 0) -> None:
        self.value = val

    def on_update(self, callback) -> None:
        """No-op : les tests existants n'exercent pas le chemin on_update."""
        pass


class _FolderCtx:
    """Contexte de dossier GUI factice (gestionnaire de contexte)."""

    def __init__(self, gui) -> None:
        self._gui = gui

    def __enter__(self):
        return self._gui

    def __exit__(self, *a):
        pass


class _FakeGui:
    """GUI factice : add_folder retourne un gestionnaire de contexte ; widgets retournent des fakes.

    ``cb_pts_val`` / ``cb_nrm_val`` pilotent les checkboxes ; ``src_val`` le curseur source.
    """

    def __init__(
        self,
        cb_pts_val: bool = True,
        cb_nrm_val: bool = False,
        src_val: int = 0,
    ) -> None:
        self._cb_pts_val = cb_pts_val
        self._cb_nrm_val = cb_nrm_val
        self._src_val = src_val
        self._cb_count = 0         # pour distinguer les deux add_checkbox successifs

    def add_folder(self, name: str):
        return _FolderCtx(self)

    def add_checkbox(self, label: str, default: bool = False):
        self._cb_count += 1
        # Premier appel -> cb_pts, second -> cb_nrm
        return _Cb(self._cb_pts_val if self._cb_count == 1 else self._cb_nrm_val)

    def add_slider(self, label: str, min_val, max_val, step, default):
        return _Slider(self._src_val)


class _FakeScene:
    """Scène factice : add_point_cloud / add_line_segments retournent des handles factices."""

    def add_point_cloud(self, name, pts, cols, point_size=None):
        return _PtsHandle()

    def add_line_segments(self, name, segs, cols, line_width=None):
        return _NrmHandle()


class _FakeServer:
    """Serveur viser factice."""

    def __init__(self) -> None:
        self.scene = _FakeScene()


# =============================================================================
# Helpers
# =============================================================================

def _make_geo(n_points: int = 4) -> GeodesicTable:
    """GeodesicTable synthétique de n_points points (géodésiques aléatoires)."""
    rng = np.random.default_rng(42)
    pts = rng.random((n_points, 3)).astype(np.float32)
    nrm = rng.random((n_points, 3)).astype(np.float32)
    geo = rng.random((n_points, n_points)).astype(np.float32)
    return GeodesicTable(points=pts, normals=nrm, geo=geo, name="obj0")


def _make_empty_geo() -> GeodesicTable:
    """GeodesicTable vide (0 points) pour tester la garde ensemble vide."""
    return GeodesicTable(
        points=np.zeros((0, 3), np.float32),
        normals=np.zeros((0, 3), np.float32),
        geo=np.zeros((0, 0), np.float32),
        name="empty",
    )


def _make_chan(name: str, object_idx, geodesic) -> types.SimpleNamespace:
    """Canal factice avec les attributs consommés par GeodesicLayer."""
    return types.SimpleNamespace(name=name, object_idx=object_idx, geodesic=geodesic)


def _make_ctx(channels) -> types.SimpleNamespace:
    """VizContext minimal duck-typé."""
    return types.SimpleNamespace(channels=channels)


def _make_pose(n_objects: int = 1) -> types.SimpleNamespace:
    """FramePose factice : rotations identité, translations nulles pour n_objects objets."""
    return types.SimpleNamespace(
        object_rot=np.eye(3)[None].repeat(n_objects, axis=0),  # (N,3,3)
        object_pos=np.zeros((n_objects, 3)),                    # (N,3)
    )


def _make_frame(pose=None, use_pose: bool = True) -> types.SimpleNamespace:
    """VizFrame factice. use_pose=False -> pose=None."""
    p = None if not use_pose else (pose if pose is not None else _make_pose())
    return types.SimpleNamespace(pose=p)


def _build_layer(
    channels,
    cb_pts_val: bool = True,
    cb_nrm_val: bool = False,
    src_val: int = 0,
) -> GeodesicLayer:
    """Construit et initialise une GeodesicLayer sur des fakes."""
    layer = GeodesicLayer()
    ctx = _make_ctx(channels)
    gui = _FakeGui(cb_pts_val=cb_pts_val, cb_nrm_val=cb_nrm_val, src_val=src_val)
    layer.setup(_FakeServer(), gui, ctx)
    return layer


_UI = UiState(channel="obj0", color_mode="uniform", point_size=0.01)


# =============================================================================
# 3. Tests d'update() — chemin nominal + gardes données manquantes
# =============================================================================

def test_update_object_channel_happy_path():
    """Chemin nominal canal objet : geodesic + pose valide, show_pts=True -> visible + points/couleurs."""
    geo = _make_geo(4)
    ch = _make_chan("obj0", object_idx=0, geodesic=geo)
    layer = _build_layer((ch,), cb_pts_val=True, cb_nrm_val=False)
    frame = _make_frame(_make_pose(n_objects=1))

    layer.update(frame, _UI)

    h = layer._h_pts[0]
    assert h.visible is True
    assert h.points is not None and h.points.shape == (4, 3)
    assert h.colors is not None and h.colors.shape == (4, 3)
    # Handle normales masqué (cb_nrm=False)
    assert layer._h_nrm[0].visible is False


def test_update_ground_channel_happy_path():
    """Chemin nominal canal ground (object_idx=None) : coords locales = monde, show_pts=True."""
    geo = _make_geo(6)
    ch = _make_chan("ground", object_idx=None, geodesic=geo)
    layer = _build_layer((ch,), cb_pts_val=True, cb_nrm_val=False)
    # Pas de pose d'objet nécessaire pour le ground
    frame = _make_frame(use_pose=False)

    layer.update(frame, _UI)

    h = layer._h_pts[0]
    assert h.visible is True
    assert h.points is not None and h.points.shape == (6, 3)


def test_update_normals_shown_when_cb_nrm_true():
    """show_nrm=True -> handle normales visible avec segments (P,2,3)."""
    geo = _make_geo(5)
    ch = _make_chan("obj0", object_idx=0, geodesic=geo)
    layer = _build_layer((ch,), cb_pts_val=False, cb_nrm_val=True)
    frame = _make_frame(_make_pose(n_objects=1))

    layer.update(frame, _UI)

    h = layer._h_nrm[0]
    assert h.visible is True
    assert h.points is not None and h.points.shape == (5, 2, 3)
    assert h.colors is not None and h.colors.shape == (5, 2, 3)


def test_update_no_geodesic_channel_noop():
    """Canal avec geodesic=None : ignoré au setup -> aucun handle, update ne lève pas."""
    ch_no_geo = _make_chan("ground", object_idx=None, geodesic=None)
    layer = _build_layer((ch_no_geo,))
    # Aucun handle créé
    assert layer._h_pts == [] and layer._h_nrm == []
    # update ne doit pas lever
    layer.update(_make_frame(), _UI)


def test_update_show_pts_false_hides():
    """show_pts=False, show_nrm=False -> les deux handles masqués."""
    geo = _make_geo(4)
    ch = _make_chan("obj0", object_idx=0, geodesic=geo)
    layer = _build_layer((ch,), cb_pts_val=False, cb_nrm_val=False)
    frame = _make_frame(_make_pose(n_objects=1))

    layer.update(frame, _UI)

    assert layer._h_pts[0].visible is False
    assert layer._h_nrm[0].visible is False


def test_update_pose_none_hides_object_channel():
    """frame.pose=None sur un canal objet -> les deux handles masqués, aucune levée."""
    geo = _make_geo(4)
    ch = _make_chan("obj0", object_idx=0, geodesic=geo)
    layer = _build_layer((ch,), cb_pts_val=True, cb_nrm_val=True)
    frame = _make_frame(use_pose=False)    # pose=None

    layer.update(frame, _UI)

    assert layer._h_pts[0].visible is False
    assert layer._h_nrm[0].visible is False


def test_update_object_idx_out_of_range_hides():
    """object_idx hors bornes (object_idx=5, seulement 1 objet en pose) -> masqué, aucune levée."""
    geo = _make_geo(4)
    ch = _make_chan("obj5", object_idx=5, geodesic=geo)   # idx=5 mais pose n'a qu'1 objet
    layer = _build_layer((ch,), cb_pts_val=True, cb_nrm_val=True)
    frame = _make_frame(_make_pose(n_objects=1))

    layer.update(frame, _UI)

    assert layer._h_pts[0].visible is False
    assert layer._h_nrm[0].visible is False


def test_update_empty_points_hides():
    """GeodesicTable avec 0 points -> les deux handles masqués, aucune levée."""
    geo = _make_empty_geo()
    ch = _make_chan("obj0", object_idx=None, geodesic=geo)
    layer = _build_layer((ch,), cb_pts_val=True, cb_nrm_val=True)
    frame = _make_frame(use_pose=False)

    layer.update(frame, _UI)

    assert layer._h_pts[0].visible is False
    assert layer._h_nrm[0].visible is False
