"""Tests de la couche sdf_iso (roadmap #6).

Deux familles :
  1. ``iso_band_points`` — fonction pure numpy-only. Grille 3×3×3 connue (plan z=1) :
     - bande étroite (band=0.5) -> seule la tranche médiane (d=0) retenue (9 nœuds à z=1.0) ;
     - bande large (band=1.5) -> les 3 tranches (27 nœuds, d∈{-1, 0, 1}).
  2. ``SdfIsoLayer.update()`` — appelé sur des fakes duck-typés. Couvre :
     - chemin nominal canal ground (object_idx=None) : nuage visible + points/couleurs définis ;
     - chemin nominal canal objet (object_idx=0) : points élevés en monde (R=I, t=0) ;
     - checkbox désactivée -> tous les handles masqués ;
     - ``frame.pose`` absent (None) -> handle objet masqué, aucune levée ;
     - ``object_idx`` hors bornes -> handle masqué, aucune levée.
"""
import types

import numpy as np

from src.prepare.contracts import Channel, SDF
from src.viz.core.layer import UiState
from src.viz.layers.sdf_iso import SdfIsoLayer, iso_band_points


# =============================================================================
# Données synthétiques communes
# =============================================================================

def _plane_sdf() -> SDF:
    """Grille 3×3×3, spacing=1, origin=(0,0,0) ; distance signée = z-1 (plan à z=1, tranche médiane).

    Indices z : 0 -> d=-1, 1 -> d=0, 2 -> d=+1.
    """
    nx = ny = nz = 3
    grid = np.zeros((nx, ny, nz))
    for k in range(nz):
        grid[:, :, k] = (k - 1) * 1.0            # z indices 0,1,2 -> d = -1, 0, +1
    witness = np.zeros((nx, ny, nz, 3))           # contenu indifférent pour ce test
    return SDF(grid=grid, witness=witness, origin=np.zeros(3), spacing=1.0, name="plane")


# =============================================================================
# 1. Tests de la fonction pure iso_band_points
# =============================================================================

def test_band_keeps_only_zero_crossing_slice():
    """Bande étroite (band=0.5) : seule la tranche k=1 (d=0) est dans |d|<0.5 -> 9 nœuds à z=1."""
    sdf = _plane_sdf()
    pts, dist = iso_band_points(sdf, band=0.5)     # |d|<0.5 -> seulement la tranche k=1 (d=0)
    assert pts.shape == (9, 3) and dist.shape == (9,)
    assert np.allclose(dist, 0.0)
    # tous les points de la bande sont à z = origin_z + spacing*1 = 1.0
    assert np.allclose(pts[:, 2], 1.0)


def test_wider_band_keeps_more_nodes():
    """Bande large (band=1.5) : les 3 tranches (27 nœuds) sont dans |d|<1.5."""
    sdf = _plane_sdf()
    pts, dist = iso_band_points(sdf, band=1.5)     # |d|<1.5 -> les 3 tranches (27 nœuds)
    assert pts.shape == (27, 3)
    assert set(np.round(np.unique(dist), 6)) == {-1.0, 0.0, 1.0}


# =============================================================================
# 2. Fakes duck-typés pour les tests d'update()
# =============================================================================

class _Handle:
    """Poignée de nuage de points factice — accepte les setters de SdfIsoLayer."""

    def __init__(self) -> None:
        self.visible: bool = True
        self.points = None
        self.colors = None


class _Cb:
    """Checkbox factice."""

    def __init__(self, val: bool = True) -> None:
        self.value = val


class _Number:
    """Curseur numérique factice."""

    def __init__(self, val: float = 0.5) -> None:
        self.value = val


class _FolderCtx:
    """Contexte de dossier GUI factice (gestionnaire de contexte)."""

    def __init__(self, gui) -> None:
        self._gui = gui

    def __enter__(self):
        return self._gui

    def __exit__(self, *a):
        pass


class _FakeGui:
    """GUI factice : add_folder retourne un gestionnaire de contexte, add_checkbox/_number des fakes."""

    def __init__(self, cb_val: bool = True, band_val: float = 0.5) -> None:
        self._cb_val = cb_val
        self._band_val = band_val

    def add_folder(self, name: str):
        return _FolderCtx(self)

    def add_checkbox(self, label: str, default: bool = False):
        return _Cb(self._cb_val)

    def add_number(self, label: str, initial: float, **kwargs):
        return _Number(self._band_val)


class _FakeScene:
    """Scene factice : add_point_cloud retourne un _Handle."""

    def add_point_cloud(self, name, pts, cols, point_size=None):
        return _Handle()


class _FakeServer:
    """Serveur viser factice."""

    def __init__(self) -> None:
        self.scene = _FakeScene()


# =============================================================================
# Helpers de construction des frames/contextes factices
# =============================================================================

def _make_channel_ground() -> Channel:
    """Canal ground : object_idx=None, SDF plan 3×3×3."""
    return Channel(name="ground", object_idx=None, sdf=_plane_sdf())


def _make_channel_obj(idx: int = 0) -> Channel:
    """Canal objet : object_idx=idx, SDF plan 3×3×3."""
    return Channel(name=f"obj{idx}", object_idx=idx, sdf=_plane_sdf())


def _make_ctx(channels: tuple) -> types.SimpleNamespace:
    """VizContext minimal duck-typé."""
    return types.SimpleNamespace(channels=channels, margin=0.5)


def _make_pose(n_objects: int = 1) -> types.SimpleNamespace:
    """FramePose factice : rotations identité, translations nulles pour n_objects objets."""
    return types.SimpleNamespace(
        object_rot=np.eye(3)[None, :, :].repeat(n_objects, axis=0),  # (N, 3, 3)
        object_pos=np.zeros((n_objects, 3)),                           # (N, 3)
    )


def _make_frame(pose=None) -> types.SimpleNamespace:
    """VizFrame factice avec une pose donnée (ou pose=_make_pose() par défaut)."""
    return types.SimpleNamespace(pose=pose if pose is not None else _make_pose())


def _build_layer(channels: tuple, cb_val: bool = True, band_val: float = 0.5) -> SdfIsoLayer:
    """Construit et initialise une SdfIsoLayer sur des fakes."""
    layer = SdfIsoLayer()
    ctx = _make_ctx(channels)
    layer.setup(_FakeServer(), _FakeGui(cb_val=cb_val, band_val=band_val), ctx)
    return layer


# =============================================================================
# 3. Tests d'update() — chemin nominal + gardes
# =============================================================================

def test_update_ground_channel_happy_path():
    """Chemin nominal canal ground : checkbox=True, object_idx=None -> visible=True + points/couleurs définis."""
    channels = (_make_channel_ground(),)
    layer = _build_layer(channels, cb_val=True, band_val=0.5)
    frame = _make_frame()
    ui = UiState(channel="ground", color_mode="uniform", point_size=0.01)

    layer.update(frame, ui)

    h = layer._handles[0]
    assert h.visible is True
    assert h.points is not None
    assert h.colors is not None


def test_update_object_channel_happy_path():
    """Chemin nominal canal objet : checkbox=True, R=I, t=0 -> visible=True + points élevés en monde."""
    channels = (_make_channel_obj(0),)
    layer = _build_layer(channels, cb_val=True, band_val=0.5)
    frame = _make_frame(_make_pose(n_objects=1))
    ui = UiState(channel="obj0", color_mode="uniform", point_size=0.01)

    layer.update(frame, ui)

    h = layer._handles[0]
    assert h.visible is True
    assert h.points is not None
    assert h.colors is not None


def test_update_checkbox_false_hides_all():
    """Checkbox désactivée -> tous les handles masqués (ground + objet)."""
    channels = (_make_channel_ground(), _make_channel_obj(0))
    layer = _build_layer(channels, cb_val=False, band_val=0.5)
    frame = _make_frame(_make_pose(n_objects=1))
    ui = UiState(channel="ground", color_mode="uniform", point_size=0.01)

    layer.update(frame, ui)

    for h in layer._handles:
        assert h.visible is False


def test_update_pose_none_hides_object_channel():
    """frame.pose=None -> handle du canal objet masqué, aucune levée (ground non affecté)."""
    channels = (_make_channel_obj(0),)
    layer = _build_layer(channels, cb_val=True, band_val=0.5)
    frame = types.SimpleNamespace(pose=None)
    ui = UiState(channel="obj0", color_mode="uniform", point_size=0.01)

    layer.update(frame, ui)

    assert layer._handles[0].visible is False


def test_update_object_idx_out_of_range_hides():
    """object_idx hors bornes (object_idx=5, seulement 1 objet en pose) -> masqué, aucune levée."""
    channels = (_make_channel_obj(5),)              # object_idx=5 mais pose n'a qu'1 objet
    layer = _build_layer(channels, cb_val=True, band_val=0.5)
    frame = _make_frame(_make_pose(n_objects=1))    # seulement 1 objet -> indices valides : 0
    ui = UiState(channel="obj5", color_mode="uniform", point_size=0.01)

    layer.update(frame, ui)

    assert layer._handles[0].visible is False
