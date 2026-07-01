"""Régression : basculement des couches solve/interaction en pause (bug UX).

Avant le correctif, les 5 couches Phase-B (robot, contacts, correspondence, sdf_iso, geodesic)
câblaient leurs checkboxes SANS on_update → basculer en pause était sans effet jusqu'à ce que
le slider frame bouge. Ce test reproduit le chemin interactif PAUSED-TOGGLE sans serveur viser.

Structure :
  - Fakes capturant les callbacks on_update (``_CaptureCb``) + GUI capturante (``_CaptureGui``).
  - Un test par couche couverte (contacts + correspondence + robot).
  - Scénario : setup() → update() données présentes → changement checkbox → déclenchement du
    callback capturé → assertion que la visibilité reflète immédiatement le nouveau état.
"""
from __future__ import annotations

import sys
import types

import numpy as np

from src.targets.contracts import MultiChannelField
from src.viz.core.layer import UiState
from src.viz.layers.contacts import ContactsLayer
from src.viz.layers.correspondence import CorrespondenceLayer
from src.viz.layers.robot import RobotLayer


# =============================================================================
# Infrastructure : fakes capturant les callbacks on_update
# =============================================================================

class _CaptureCb:
    """Checkbox factice capturant les callbacks on_update (simulation d'un clic en pause)."""

    def __init__(self, val: bool = True) -> None:
        self.value = val
        self._callbacks: list = []

    def on_update(self, cb) -> None:
        """Enregistre le callback — capturé par le test pour invoquer manuellement."""
        self._callbacks.append(cb)

    def trigger(self) -> None:
        """Simule un clic utilisateur : invoque tous les callbacks enregistrés."""
        for cb in self._callbacks:
            cb(None)


class _FolderCtx:
    """Contexte de dossier GUI factice (gestionnaire de contexte)."""

    def __init__(self, gui) -> None:
        self._gui = gui

    def __enter__(self):
        return self._gui

    def __exit__(self, *a):
        pass


class _CaptureGui:
    """GUI factice capturant les checkboxes créées dans une liste ordonnée.

    Les couches appellent add_checkbox dans un ordre connu ;
    les tests accèdent aux contrôles via self.checkboxes[i].
    """

    def __init__(self, cb_val: bool = True) -> None:
        self._cb_val = cb_val
        self.checkboxes: list[_CaptureCb] = []

    def add_folder(self, name: str):
        return _FolderCtx(self)

    def add_checkbox(self, label: str, default: bool = True) -> _CaptureCb:
        cb = _CaptureCb(self._cb_val)
        self.checkboxes.append(cb)
        return cb


# =============================================================================
# Fakes de scène (points / segments)
# =============================================================================

class _PtHandle:
    """Poignée de nuage de points factice."""

    def __init__(self) -> None:
        self.visible: bool = True
        self.points = None
        self.colors = None
        self.point_size = None


class _SegHandle:
    """Poignée de segments factice."""

    def __init__(self) -> None:
        self.visible: bool = True
        self.points = None
        self.colors = None


class _FakeScene:
    """Scène factice renvoyant des handles (points / segments)."""

    def add_point_cloud(self, name, pts, cols, point_size=None):
        return _PtHandle()

    def add_line_segments(self, name, segs, cols, line_width=None):
        return _SegHandle()


class _FakeServer:
    def __init__(self) -> None:
        self.scene = _FakeScene()


# =============================================================================
# Helpers frames / contextes
# =============================================================================

def _make_field(C: int = 2, M: int = 4) -> MultiChannelField:
    """Champ multi-canal minimal."""
    dist = np.zeros((C, M))
    direction = np.zeros((C, M, 3))
    direction[..., 2] = 1.0
    witness = np.zeros((C, M, 3))
    active = np.zeros((C, M), bool)
    return MultiChannelField(
        distance=dist, direction=direction, witness=witness,
        active=active, channels=tuple(f"c{i}" for i in range(C)),
    )


def _contacts_frame(*, solved: bool = True, M: int = 4) -> types.SimpleNamespace:
    """Frame pour ContactsLayer : données valides si solved=True."""
    C = 2
    tgt_field = _make_field(C, M)
    ach_ns = types.SimpleNamespace(field=tgt_field)
    solved_ns = (
        types.SimpleNamespace(
            robot_points_world=np.zeros((M, 3), np.float32),
            contact_achieved=ach_ns,
        )
        if solved else None
    )
    ri_ns = types.SimpleNamespace(field=tgt_field)
    tgts_ns = types.SimpleNamespace(robot_interaction=ri_ns)
    return types.SimpleNamespace(solved=solved_ns, targets=tgts_ns)


def _contacts_ctx() -> types.SimpleNamespace:
    return types.SimpleNamespace(channel_names=("ground", "obj0"), margin=0.1)


def _corr_frame(*, solved: bool = True, M: int = 2) -> types.SimpleNamespace:
    """Frame pour CorrespondenceLayer : données valides si solved=True."""
    solved_ns = (
        types.SimpleNamespace(robot_points_world=np.zeros((M, 3), np.float32))
        if solved else None
    )
    cloud = np.zeros((4, 3), np.float32)
    return types.SimpleNamespace(solved=solved_ns, human_cloud_world=cloud)


def _corr_ctx(smpl_idx=None) -> types.SimpleNamespace:
    if smpl_idx is None:
        smpl_idx = np.array([0, 1])
    return types.SimpleNamespace(correspondence=types.SimpleNamespace(smpl_idx=smpl_idx))


_UI = UiState(channel="ground", color_mode="distance", point_size=0.01)


# =============================================================================
# Tests : ContactsLayer — bascule en pause
# =============================================================================

def test_contacts_target_toggle_paused():
    """Décocher 'contact cible' en pause -> masquage immédiat sans appel player.

    Reproduit le chemin interactif : setup() + update() données présentes + clic checkbox.
    Vérifie que le callback on_update capturé déclenche bien le re-rendu (visible=False).
    """
    gui = _CaptureGui(cb_val=True)
    layer = ContactsLayer()
    layer.setup(_FakeServer(), gui, _contacts_ctx())

    frame = _contacts_frame(solved=True)
    layer.update(frame, _UI)

    # Vérification préalable : visible=True après update() données présentes
    assert layer._h_target.visible is True, "prérequis : nuage cible visible avant le clic"

    # Simulation clic utilisateur en pause : décocher 'contact cible'
    gui.checkboxes[0].value = False   # premier checkbox = cb_target
    gui.checkboxes[0].trigger()       # invoque le callback on_update capturé

    # Le toggle doit être actif immédiatement — c'est l'invariant cassé avant le correctif
    assert layer._h_target.visible is False, \
        "bascule en pause : _h_target doit être masqué immédiatement"
    # L'autre nuage (atteint) ne doit PAS être masqué (cb_achieved toujours True)
    assert layer._h_achieved.visible is True, \
        "bascule en pause : _h_achieved ne doit pas être affecté par le toggle cible"


def test_contacts_recheck_restores_visibility():
    """Recocher 'contact cible' après l'avoir décoché -> visible=True immédiatement."""
    gui = _CaptureGui(cb_val=True)
    layer = ContactsLayer()
    layer.setup(_FakeServer(), gui, _contacts_ctx())

    frame = _contacts_frame(solved=True)
    layer.update(frame, _UI)

    # Décocher
    gui.checkboxes[0].value = False
    gui.checkboxes[0].trigger()
    assert layer._h_target.visible is False

    # Recocher
    gui.checkboxes[0].value = True
    gui.checkboxes[0].trigger()
    assert layer._h_target.visible is True, \
        "recocher la checkbox doit restaurer la visibilité immédiatement"


def test_contacts_achieved_toggle_paused():
    """Décocher 'contact atteint' en pause -> masquage immédiat (second checkbox)."""
    gui = _CaptureGui(cb_val=True)
    layer = ContactsLayer()
    layer.setup(_FakeServer(), gui, _contacts_ctx())

    frame = _contacts_frame(solved=True)
    layer.update(frame, _UI)

    assert layer._h_achieved.visible is True

    # Décocher 'contact atteint' (deuxième checkbox)
    gui.checkboxes[1].value = False
    gui.checkboxes[1].trigger()

    assert layer._h_achieved.visible is False, \
        "bascule en pause : _h_achieved doit être masqué immédiatement"
    # Nuage cible non affecté
    assert layer._h_target.visible is True


def test_contacts_toggle_noop_before_first_update():
    """Déclencher le callback avant tout update() (last_frame=None) -> ne lève pas."""
    gui = _CaptureGui(cb_val=True)
    layer = ContactsLayer()
    layer.setup(_FakeServer(), gui, _contacts_ctx())

    # Aucun update() encore appelé — last_frame=None — le callback doit être silencieux
    gui.checkboxes[0].value = False
    gui.checkboxes[0].trigger()   # ne doit pas lever


# =============================================================================
# Tests : CorrespondenceLayer — bascule en pause
# =============================================================================

def test_correspondence_toggle_paused():
    """Décocher 'lignes SMPL↔G1' en pause -> masquage immédiat sans appel player.

    Même invariant que contacts : le callback on_update doit déclencher un re-rendu.
    """
    gui = _CaptureGui(cb_val=True)
    layer = CorrespondenceLayer()
    layer.setup(_FakeServer(), gui, _corr_ctx(np.array([0, 1])))

    frame = _corr_frame(solved=True, M=2)
    layer.update(frame, _UI)

    assert layer._h.visible is True, "prérequis : segments visibles avant le clic"

    # Clic en pause : décocher
    gui.checkboxes[0].value = False
    gui.checkboxes[0].trigger()

    assert layer._h.visible is False, \
        "bascule en pause : handle segments doit être masqué immédiatement"


def test_correspondence_recheck_restores():
    """Recocher 'lignes SMPL↔G1' après décocher -> visible=True immédiatement."""
    gui = _CaptureGui(cb_val=True)
    layer = CorrespondenceLayer()
    layer.setup(_FakeServer(), gui, _corr_ctx(np.array([0, 1])))

    frame = _corr_frame(solved=True, M=2)
    layer.update(frame, _UI)

    gui.checkboxes[0].value = False
    gui.checkboxes[0].trigger()
    assert layer._h.visible is False

    gui.checkboxes[0].value = True
    gui.checkboxes[0].trigger()
    assert layer._h.visible is True, \
        "recocher doit restaurer la visibilité immédiatement"


def test_correspondence_toggle_noop_before_first_update():
    """Déclencher le callback avant tout update() -> ne lève pas."""
    gui = _CaptureGui(cb_val=True)
    layer = CorrespondenceLayer()
    layer.setup(_FakeServer(), gui, _corr_ctx(np.array([0, 1])))

    gui.checkboxes[0].value = False
    gui.checkboxes[0].trigger()   # last_frame=None -> silencieux


# =============================================================================
# Fakes robot (setup() avec stubs yourdfpy/ViserUrdf)
# =============================================================================

class _FakeUrdf:
    """Substitut ViserUrdf pour setup() sans dépendance viser/yourdfpy."""

    def __init__(self, dof: int = 4) -> None:
        self.show_visual = True
        self._dof = dof
        self._cfg_calls: list = []

    def update_cfg(self, cfg) -> None:
        """Enregistre la config joints reçue."""
        self._cfg_calls.append(np.asarray(cfg, np.float64))

    def get_actuated_joint_limits(self):
        """Retourne une liste fictive de longueur dof pour initialiser _dof dans setup."""
        return [None] * self._dof


class _FakeBaseFrame:
    """Substitut du frame de base viser (position + wxyz écrits par update)."""

    def __init__(self) -> None:
        self.visible = True
        self.position = np.zeros(3)
        self.wxyz = np.array([1.0, 0.0, 0.0, 0.0])


class _FakeSceneRobot:
    """Scène factice retournant un frame de base pré-construit (pour add_frame)."""

    def __init__(self, base: _FakeBaseFrame) -> None:
        self._base = base

    def add_frame(self, name, show_axes=False) -> _FakeBaseFrame:
        return self._base


class _FakeServerRobot:
    """Serveur factice portant une scène robot."""

    def __init__(self, base: _FakeBaseFrame) -> None:
        self.scene = _FakeSceneRobot(base)


# =============================================================================
# Tests : RobotLayer — bascule en pause
# =============================================================================

def test_robot_toggle_paused():
    """Décocher 'Show solved G1' en pause → masquage immédiat sans appel player.

    Reproduit le chemin interactif sur RobotLayer (couche la plus distincte
    structurellement : _last_frame/_last_ui initialisés dans __init__,
    callback _on_change câblé dans setup()).
    setup() est appelé avec des stubs yourdfpy/ViserUrdf injectés dans sys.modules
    (imports lazy dans setup — pas besoin d'un vrai serveur viser).
    Vérifie que le callback on_update capturé masque immédiatement _base.visible
    et _urdf.show_visual sans nudge du slider.
    """
    dof = 4
    fake_urdf = _FakeUrdf(dof=dof)
    fake_base = _FakeBaseFrame()

    # Stubber les modules lourds importés lazily dans setup() sans vrai serveur
    _yourdfpy = types.ModuleType("yourdfpy")
    _yourdfpy.URDF = types.SimpleNamespace(load=lambda *a, **kw: None)

    _viser_extras = types.ModuleType("viser.extras")
    _viser_extras.ViserUrdf = lambda *a, **kw: fake_urdf

    _viser = types.ModuleType("viser")
    _viser.extras = _viser_extras

    _saved = {k: sys.modules.pop(k, None) for k in ("yourdfpy", "viser", "viser.extras")}
    sys.modules.update({"yourdfpy": _yourdfpy, "viser": _viser, "viser.extras": _viser_extras})
    try:
        gui = _CaptureGui(cb_val=True)
        ctx = types.SimpleNamespace(robot_urdf_path="dummy", has_solve=True)
        layer = RobotLayer()
        layer.setup(_FakeServerRobot(fake_base), gui, ctx)
    finally:
        # Restaure sys.modules dans tous les cas (exception incluse)
        for k, v in _saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    # _on_change doit avoir été capturé par on_update
    cb = gui.checkboxes[0]
    assert len(cb._callbacks) == 1, "_on_change doit être enregistré via on_update"

    # Premier update : frame avec solved présent → robot visible
    q = np.zeros(7 + dof)
    q[3:7] = [0.0, 0.0, 0.0, 1.0]  # quaternion identité xyzw (qx=qy=qz=0, qw=1)
    solved_ns = types.SimpleNamespace(q=q)
    frame = types.SimpleNamespace(solved=solved_ns)

    layer.update(frame, _UI)

    assert fake_base.visible is True, "prérequis : robot visible après update() données présentes"
    assert fake_urdf.show_visual is True

    # Simulation clic en pause : décocher le toggle
    cb.value = False
    cb.trigger()  # invoque _on_change → re-appelle update() → show=False

    # Invariant : masquage immédiat sans nudge du slider
    assert fake_base.visible is False, \
        "bascule en pause : _base doit être masquée immédiatement"
    assert fake_urdf.show_visual is False, \
        "bascule en pause : _urdf.show_visual doit être False immédiatement"
