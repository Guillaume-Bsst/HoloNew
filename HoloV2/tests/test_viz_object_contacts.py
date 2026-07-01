"""Tests de la couche ObjectContactsLayer (roadmap #3b — contacts objets).

Familles :
  1. ``_obj_contact_colors`` — fonction pure : forme/dtype, distance/active/uniform.
  2. ``ObjectContactsLayer.update()`` — chemin nominal (nuages source+résolu + witness dessinés).
  3. Gardes données manquantes (solved=None, targets=None, pose=None, canal inconnu → masquage).
  4. Cloud résolu distinct du cloud source quand la pose résolue diffère (object_cloud_solved).
  5. Mapping witness via pose CANAL (oi), pas via pose objet k.
  6. Incohérence nombre d'objets (moins de clouds que de handles → masquage).
  7. Toggle en pause (clic checkbox → update sans appel player)."""
import types

import numpy as np
import pytest

from src.targets.contracts import MultiChannelField
from src.viz.core.layer import UiState
from src.viz.layers.object_contacts import ObjectContactsLayer, _obj_contact_colors


# =============================================================================
# Données de test communes
# =============================================================================

def _field(C: int = 2, P: int = 4) -> MultiChannelField:
    """Champ multi-canal minimal pour les P probes d'un objet, avec activations variées."""
    dist = np.zeros((C, P))
    dist[0] = np.array([0.0, 0.02, 0.05, 0.09])   # canal 0 varie
    direction = np.zeros((C, P, 3))
    direction[..., 2] = 1.0
    witness = np.zeros((C, P, 3))
    active = np.zeros((C, P), bool)
    active[0] = np.array([True, True, False, False])
    return MultiChannelField(
        distance=dist, direction=direction, witness=witness,
        active=active, channels=tuple(f"c{i}" for i in range(C)),
    )


# =============================================================================
# 1. Tests de la fonction pure _obj_contact_colors
# =============================================================================

def test_obj_contact_colors_distance_mode():
    """Mode 'distance' : forme (P, 3) uint8 + au moins deux couleurs distinctes."""
    f = _field()
    cols = _obj_contact_colors(f, channel_idx=0, mode="distance", margin=0.1,
                               uniform_rgb=np.array([255, 170, 0], np.uint8))
    assert cols.shape == (4, 3) and cols.dtype == np.uint8
    assert len({tuple(c) for c in cols}) >= 2


def test_obj_contact_colors_active_mode():
    """Mode 'active' : actifs et inactifs reçoivent des couleurs distinctes."""
    f = _field()
    cols = _obj_contact_colors(f, channel_idx=0, mode="active", margin=0.1,
                               uniform_rgb=np.array([255, 170, 0], np.uint8))
    assert cols.shape == (4, 3) and cols.dtype == np.uint8
    assert tuple(cols[0]) == tuple(cols[1])   # les 2 actifs partagent la même couleur
    assert tuple(cols[0]) != tuple(cols[2])   # actif ≠ inactif


def test_obj_contact_colors_uniform_mode():
    """Mode 'uniform' : tous les P points reçoivent la même couleur."""
    f = _field()
    rgb = np.array([0, 200, 120], np.uint8)
    cols = _obj_contact_colors(f, channel_idx=1, mode="uniform", margin=0.1, uniform_rgb=rgb)
    assert cols.shape == (4, 3) and cols.dtype == np.uint8
    assert len({tuple(c) for c in cols}) == 1


# =============================================================================
# Fakes duck-typés
# =============================================================================

class _Handle:
    """Poignée nuage de points factice."""

    def __init__(self):
        self.visible = True
        self.points = None
        self.colors = None
        self.point_size = None


class _SegHandle:
    """Poignée segments de ligne factice."""

    def __init__(self):
        self.visible = True
        self.points = None
        self.colors = None


class _Cb:
    """Checkbox factice simple."""

    def __init__(self, val: bool = True):
        self.value = val

    def on_update(self, callback) -> None:
        pass


class _CaptureCb:
    """Checkbox factice capturant le callback on_update (simulation d'un clic en pause)."""

    def __init__(self, val: bool = True) -> None:
        self.value = val
        self._callbacks: list = []

    def on_update(self, cb) -> None:
        self._callbacks.append(cb)

    def trigger(self) -> None:
        """Simule un clic utilisateur : invoque tous les callbacks enregistrés."""
        for cb in self._callbacks:
            cb(None)


class _FolderCtx:
    def __init__(self, gui):
        self._gui = gui

    def __enter__(self):
        return self._gui

    def __exit__(self, *a):
        pass


class _FakeGui:
    """GUI factice retournant des _Cb pour add_checkbox."""

    def __init__(self, cb_val: bool = True):
        self._cb_val = cb_val

    def add_folder(self, name: str):
        return _FolderCtx(self)

    def add_checkbox(self, label: str, default: bool = True):
        return _Cb(self._cb_val)


class _CaptureGui:
    """GUI factice capturant toutes les checkboxes créées dans une liste ordonnée."""

    def __init__(self, cb_val: bool = True) -> None:
        self._cb_val = cb_val
        self.checkboxes: list[_CaptureCb] = []

    def add_folder(self, name: str):
        return _FolderCtx(self)

    def add_checkbox(self, label: str, default: bool = True) -> _CaptureCb:
        cb = _CaptureCb(self._cb_val)
        self.checkboxes.append(cb)
        return cb


class _FakeScene:
    """Scene factice : add_point_cloud → _Handle ; add_line_segments → _SegHandle."""

    def add_point_cloud(self, name, pts, cols, point_size=None):
        return _Handle()

    def add_line_segments(self, name, segs, cols, line_width=None):
        return _SegHandle()


class _FakeServer:
    def __init__(self):
        self.scene = _FakeScene()


def _make_ctx(n_objects: int = 1, channel_names: tuple | None = None):
    """Contexte viz minimal — canal 0 = ground (object_idx=None), canal c≥1 = objet c-1."""
    if channel_names is None:
        channel_names = ("ground",) + tuple(f"obj{i}" for i in range(n_objects))
    channels = tuple(
        types.SimpleNamespace(object_idx=(None if i == 0 else i - 1))
        for i in range(len(channel_names))
    )
    return types.SimpleNamespace(
        channel_names=channel_names,
        margin=0.1,
        channels=channels,
        n_objects=n_objects,
    )


def _make_frame(
    *,
    solved: bool = True,
    targets: bool = True,
    env_interaction: bool = True,
    pose: bool = True,
    contact_achieved: bool = True,
    n_objects: int = 1,
    C: int = 2,
    P: int = 4,
) -> types.SimpleNamespace:
    """VizFrame duck-typé couvrant les combinaisons de gardes pour ObjectContactsLayer.

    Paramètres :
        solved            : si False, ``frame.solved`` est None.
        targets           : si False, ``frame.targets`` est None.
        env_interaction   : si False, ``frame.targets.env_interaction`` est None.
        pose              : si False, ``frame.pose`` est None.
        contact_achieved  : si False, ``frame.solved.contact_achieved`` est None.
        n_objects         : nombre d'objets simulés.
    """
    # Champ commun pour cible et atteint de chaque objet
    tgt_field = _field(C, P)

    # env_interaction : per_object = tuple de n_objects champs
    env_int_ns = (
        types.SimpleNamespace(per_object=tuple(tgt_field for _ in range(n_objects)))
        if env_interaction else None
    )
    tgts_ns = (
        types.SimpleNamespace(env_interaction=env_int_ns)
        if targets else None
    )

    # pose : rotations et positions objets
    pose_ns = (
        types.SimpleNamespace(
            object_rot=np.tile(np.eye(3), (n_objects, 1, 1)),  # (N, 3, 3) identité
            object_pos=np.zeros((n_objects, 3)),
        )
        if pose else None
    )

    # object_poses résolu : (N, 7) format [x, y, z, qw, qx, qy, qz]
    object_poses = np.zeros((n_objects, 7))
    object_poses[:, 3] = 1.0   # qw = 1 (identité)

    # ContactEnvEval factice : porte juste un .field
    env_evals = tuple(types.SimpleNamespace(field=tgt_field) for _ in range(n_objects))
    ach_ns = (
        types.SimpleNamespace(env=env_evals)
        if contact_achieved else None
    )
    solved_ns = (
        types.SimpleNamespace(
            object_poses=object_poses,
            contact_achieved=ach_ns,
        )
        if solved else None
    )

    # Clouds objets : un nuage (P, 3) par objet (coordonnées nulles)
    clouds = tuple(np.zeros((P, 3), np.float32) for _ in range(n_objects))

    return types.SimpleNamespace(
        solved=solved_ns,
        targets=tgts_ns,
        pose=pose_ns,
        object_clouds_world=clouds,
    )


def _build_layer(
    cb_val: bool = True,
    n_objects: int = 1,
    channel_names: tuple | None = None,
) -> ObjectContactsLayer:
    """Construit et initialise une ObjectContactsLayer sur des fakes."""
    layer = ObjectContactsLayer()
    layer.setup(_FakeServer(), _FakeGui(cb_val=cb_val), _make_ctx(n_objects, channel_names))
    return layer


def _build_layer_capture(
    cb_val: bool = True,
    n_objects: int = 1,
) -> tuple:
    """Construit une ObjectContactsLayer avec GUI capturante."""
    gui = _CaptureGui(cb_val=cb_val)
    layer = ObjectContactsLayer()
    layer.setup(_FakeServer(), gui, _make_ctx(n_objects))
    return layer, gui


# =============================================================================
# 2. Chemin nominal — nuages + witness dessinés
# =============================================================================

def test_happy_path_clouds_visible_and_points_set():
    """Chemin nominal : solved + targets + pose + canal connu → points définis, visible=True."""
    layer = _build_layer(cb_val=True, n_objects=1)
    frame = _make_frame(solved=True, targets=True, pose=True)
    ui = UiState(channel="ground", color_mode="distance", point_size=0.01)

    layer.update(frame, ui)

    assert layer._h_targets[0].points is not None
    assert layer._h_achieved[0].points is not None
    assert layer._h_targets[0].visible is True
    assert layer._h_achieved[0].visible is True


def test_happy_path_cb_false_makes_clouds_invisible():
    """Checkboxes désactivées → nuages non visibles sur le chemin nominal."""
    layer = _build_layer(cb_val=False, n_objects=1)
    frame = _make_frame(solved=True, targets=True, pose=True)
    ui = UiState(channel="ground", color_mode="active", point_size=0.01)

    layer.update(frame, ui)

    assert layer._h_targets[0].visible is False
    assert layer._h_achieved[0].visible is False


def test_happy_path_witness_segments_visible():
    """Chemin nominal avec sondes actives → handles witness peuplés et visibles."""
    layer = _build_layer(cb_val=True, n_objects=1)
    frame = _make_frame(solved=True, targets=True, pose=True)
    # _field() a active[0]=[T,T,F,F] → 2 sondes actives sur le canal ground
    ui = UiState(channel="ground", color_mode="distance", point_size=0.01)

    layer.update(frame, ui)

    assert layer._h_wit_targets[0].points is not None
    assert layer._h_wit_targets[0].points.shape == (2, 2, 3)
    assert layer._h_wit_targets[0].visible is True
    assert layer._h_wit_achieved[0].points is not None
    assert layer._h_wit_achieved[0].points.shape == (2, 2, 3)
    assert layer._h_wit_achieved[0].visible is True


# =============================================================================
# 3. Gardes données manquantes
# =============================================================================

def test_solved_none_hides_all():
    """solved=None (solve-gated) → tous les handles masqués, aucune levée."""
    layer = _build_layer(n_objects=1)
    frame = _make_frame(solved=False)
    ui = UiState(channel="ground", color_mode="distance", point_size=0.01)

    layer.update(frame, ui)

    assert layer._h_targets[0].visible is False
    assert layer._h_achieved[0].visible is False
    assert layer._h_wit_targets[0].visible is False
    assert layer._h_wit_achieved[0].visible is False


def test_targets_none_hides_all():
    """targets=None → tous les handles masqués, aucune levée."""
    layer = _build_layer(n_objects=1)
    frame = _make_frame(targets=False)
    ui = UiState(channel="ground", color_mode="distance", point_size=0.01)

    layer.update(frame, ui)

    assert layer._h_targets[0].visible is False
    assert layer._h_achieved[0].visible is False


def test_env_interaction_none_hides_all():
    """env_interaction=None → tous les handles masqués, aucune levée."""
    layer = _build_layer(n_objects=1)
    frame = _make_frame(env_interaction=False)
    ui = UiState(channel="ground", color_mode="distance", point_size=0.01)

    layer.update(frame, ui)

    assert layer._h_targets[0].visible is False
    assert layer._h_achieved[0].visible is False


def test_pose_none_hides_all():
    """pose=None → tous les handles masqués, aucune levée."""
    layer = _build_layer(n_objects=1)
    frame = _make_frame(pose=False)
    ui = UiState(channel="ground", color_mode="distance", point_size=0.01)

    layer.update(frame, ui)

    assert layer._h_targets[0].visible is False
    assert layer._h_achieved[0].visible is False


def test_unknown_channel_hides_all():
    """Canal inconnu (UI en transition) → masqués, aucune levée."""
    layer = _build_layer(n_objects=1)
    frame = _make_frame(solved=True, targets=True, pose=True)
    ui = UiState(channel="canal_inconnu", color_mode="distance", point_size=0.01)

    layer.update(frame, ui)

    assert layer._h_targets[0].visible is False
    assert layer._h_achieved[0].visible is False
    assert layer._h_wit_targets[0].visible is False
    assert layer._h_wit_achieved[0].visible is False


def test_contact_achieved_none_hides_achieved_only():
    """contact_achieved=None → cible visible, atteint masqué (résolution partielle)."""
    layer = _build_layer(cb_val=True, n_objects=1)
    frame = _make_frame(solved=True, targets=True, pose=True, contact_achieved=False)
    ui = UiState(channel="ground", color_mode="uniform", point_size=0.01)

    layer.update(frame, ui)

    assert layer._h_targets[0].visible is True       # cible : données présentes → visible
    assert layer._h_achieved[0].visible is False     # atteint : contact_achieved absent → masqué
    assert layer._h_wit_achieved[0].visible is False


# =============================================================================
# 4. Cloud résolu ≠ cloud source quand la pose résolue diffère
# =============================================================================

def test_both_clouds_use_solved_cloud_when_pose_differs():
    """Cible et atteint affichent tous les deux le cloud résolu (scène résolue).

    Un objet avec cloud source non-nul + translation résolue [10,0,0] →
    h_targets et h_achieved contiennent tous les deux le cloud décalé de +10 en x.
    """
    n_objects, C, P = 1, 2, 4
    layer = _build_layer(cb_val=True, n_objects=n_objects)

    # Cloud source non-nul
    cloud_src = np.ones((P, 3), np.float32)
    clouds = (cloud_src,)

    # Pose source = identité
    pose_ns = types.SimpleNamespace(
        object_rot=np.tile(np.eye(3), (n_objects, 1, 1)),
        object_pos=np.zeros((n_objects, 3)),
    )

    # Pose résolue : translation [10, 0, 0] sur l'objet 0
    object_poses = np.zeros((n_objects, 7))
    object_poses[0, 0] = 10.0   # x = 10
    object_poses[0, 3] = 1.0    # qw = 1 (rotation identité)

    tgt_field = _field(C, P)
    env_int_ns = types.SimpleNamespace(per_object=(tgt_field,))
    tgts_ns = types.SimpleNamespace(env_interaction=env_int_ns)

    env_evals = (types.SimpleNamespace(field=tgt_field),)
    ach_ns = types.SimpleNamespace(env=env_evals)
    solved_ns = types.SimpleNamespace(object_poses=object_poses, contact_achieved=ach_ns)

    frame = types.SimpleNamespace(
        solved=solved_ns, targets=tgts_ns, pose=pose_ns, object_clouds_world=clouds,
    )
    ui = UiState(channel="ground", color_mode="uniform", point_size=0.01)

    layer.update(frame, ui)

    cible_pts = layer._h_targets[0].points
    atteint_pts = layer._h_achieved[0].points

    assert cible_pts is not None and atteint_pts is not None
    # Les deux nuages doivent afficher le cloud résolu (identique entre cible et atteint)
    np.testing.assert_allclose(cible_pts, atteint_pts, atol=1e-5,
                                err_msg="cible et atteint doivent afficher le même cloud résolu")
    # Le cloud résolu est décalé de +10 en x par rapport au cloud source original
    np.testing.assert_allclose(cible_pts[:, 0], cloud_src[:, 0] + 10.0, atol=1e-5,
                                err_msg="cloud résolu décalé de +10 en x par rapport au source")


# =============================================================================
# 5. Mapping witness via pose CANAL (oi), pas pose objet k
# =============================================================================

def test_witness_mapped_via_channel_object_pose_not_object_k():
    """Le witness est mappé via la pose RÉSOLUE de l'objet DU CANAL c (oi), pas celle de k.

    Configuration :
      - canal 1 = 'obj0' → oi=0 → l'objet 0 est le canal-objet.
      - 2 objets : k=0 (canal-objet) et k=1 (autre objet).
      - Pose résolue oi=0 : t_sol=[20,0,0] → witness cible endpoint = [1,0,0]+[20,0,0] = [21,0,0].
      - Pose résolue oi=0 : t_sol=[20,0,0] → witness atteint endpoint = [1,0,0]+[20,0,0] = [21,0,0].
    On teste sur l'objet k=1 (l'objet non-canal) et le canal obj0 : le witness de k=1 sur le
    canal obj0 doit utiliser la pose RÉSOLUE de oi=0 (le canal-objet) pour cible et atteint.
    """
    n_objects, C, P = 2, 3, 4
    # Channels : ground (oi=None), obj0 (oi=0), obj1 (oi=1)
    channel_names = ("ground", "obj0", "obj1")
    ctx = _make_ctx(n_objects=n_objects, channel_names=channel_names)
    gui = _FakeGui(cb_val=True)
    layer = ObjectContactsLayer()
    layer.setup(_FakeServer(), gui, ctx)

    # Witness local [1,0,0] sur canal 1 (obj0), sonde 0 uniquement active
    dist = np.zeros((C, P))
    direction = np.zeros((C, P, 3))
    direction[..., 2] = 1.0
    witness = np.zeros((C, P, 3))
    witness[1, 0] = [1.0, 0.0, 0.0]       # witness local canal obj0 (idx=1)
    active = np.zeros((C, P), bool)
    active[1, 0] = True                    # sonde 0 active sur canal obj0

    tgt_field = MultiChannelField(
        distance=dist, direction=direction, witness=witness,
        active=active, channels=channel_names,
    )

    # Pose source objet 0 (oi=0) : R=I, t=[5,0,0]
    t_src_oi0 = np.array([5.0, 0.0, 0.0])
    # Pose résolue objet 0 (oi=0) : R=I, t=[20,0,0]
    t_sol_oi0 = np.array([20.0, 0.0, 0.0])

    pose_ns = types.SimpleNamespace(
        object_rot=np.tile(np.eye(3), (n_objects, 1, 1)),
        object_pos=np.stack([t_src_oi0, np.zeros(3)]),  # oi=0 → t_src, oi=1 → 0
    )

    object_poses = np.zeros((n_objects, 7))
    object_poses[0, :3] = t_sol_oi0    # t_sol pour oi=0
    object_poses[:, 3] = 1.0           # qw=1 pour tous

    env_int_ns = types.SimpleNamespace(
        per_object=(tgt_field, tgt_field),  # 2 objets
    )
    tgts_ns = types.SimpleNamespace(env_interaction=env_int_ns)

    env_evals = (
        types.SimpleNamespace(field=tgt_field),
        types.SimpleNamespace(field=tgt_field),
    )
    ach_ns = types.SimpleNamespace(env=env_evals)
    solved_ns = types.SimpleNamespace(object_poses=object_poses, contact_achieved=ach_ns)

    clouds = tuple(np.zeros((P, 3), np.float32) for _ in range(n_objects))

    frame = types.SimpleNamespace(
        solved=solved_ns, targets=tgts_ns, pose=pose_ns, object_clouds_world=clouds,
    )
    # Canal = obj0 (index 1 dans channel_names) → oi=0
    ui = UiState(channel="obj0", color_mode="distance", point_size=0.01)
    layer.update(frame, ui)

    # Vérification sur l'objet k=1 (le non-canal) : le witness doit utiliser oi=0 (pose RÉSOLUE)
    # Witness cible endpoint : [1,0,0] @ I.T + [20,0,0] = [21,0,0] (pose RÉSOLUE du canal-objet)
    seg_tgt = layer._h_wit_targets[1].points
    assert seg_tgt is not None and seg_tgt.shape == (1, 2, 3)
    np.testing.assert_allclose(seg_tgt[0, 1], [21.0, 0.0, 0.0], atol=1e-5,
                                err_msg="endpoint witness CIBLE via pose RÉSOLUE du canal-objet")

    # Witness atteint endpoint : [1,0,0] @ I.T + [20,0,0] = [21,0,0] (pose RÉSOLUE du canal-objet)
    seg_ach = layer._h_wit_achieved[1].points
    assert seg_ach is not None and seg_ach.shape == (1, 2, 3)
    np.testing.assert_allclose(seg_ach[0, 1], [21.0, 0.0, 0.0], atol=1e-5,
                                err_msg="endpoint witness ATTEINT via pose RÉSOLUE du canal-objet")


# =============================================================================
# 6. Moins d'objets dans le frame que de handles
# =============================================================================

def test_fewer_objects_in_frame_than_handles_masks_extras():
    """Deux handles créés (n_objects=2), frame avec 1 seul cloud → le 2e handle est masqué."""
    layer = _build_layer(n_objects=2, cb_val=True)

    # Frame avec seulement 1 cloud objet (k=0 OK, k=1 absent)
    single_cloud = (np.zeros((4, 3), np.float32),)
    tgt_field = _field(C=2, P=4)
    env_int_ns = types.SimpleNamespace(per_object=(tgt_field,))   # 1 seul objet
    tgts_ns = types.SimpleNamespace(env_interaction=env_int_ns)

    pose_ns = types.SimpleNamespace(
        object_rot=np.tile(np.eye(3), (1, 1, 1)),   # 1 seul objet
        object_pos=np.zeros((1, 3)),
    )
    object_poses = np.zeros((1, 7))
    object_poses[0, 3] = 1.0
    env_evals = (types.SimpleNamespace(field=tgt_field),)
    ach_ns = types.SimpleNamespace(env=env_evals)
    solved_ns = types.SimpleNamespace(object_poses=object_poses, contact_achieved=ach_ns)

    frame = types.SimpleNamespace(
        solved=solved_ns, targets=tgts_ns, pose=pose_ns, object_clouds_world=single_cloud,
    )
    ui = UiState(channel="ground", color_mode="uniform", point_size=0.01)
    layer.update(frame, ui)

    # Objet k=0 : doit être rendu
    assert layer._h_targets[0].visible is True
    # Objet k=1 : absent → masqué sans levée
    assert layer._h_targets[1].visible is False
    assert layer._h_achieved[1].visible is False
    assert layer._h_wit_targets[1].visible is False
    assert layer._h_wit_achieved[1].visible is False


# =============================================================================
# 7. Toggle en pause
# =============================================================================

def test_toggle_paused_rerenders_immediately():
    """Déclencher un checkbox callback en pause → update() immédiat sans appel slider.

    Ordre des checkboxes créées par setup() :
      0 = 'cloud cible'    1 = 'cloud atteint'
      2 = 'witness cible'  3 = 'witness atteint'
    """
    layer, gui = _build_layer_capture(cb_val=True, n_objects=1)
    frame = _make_frame(solved=True, targets=True, pose=True)
    ui = UiState(channel="ground", color_mode="distance", point_size=0.01)

    layer.update(frame, ui)

    # Prérequis : cloud cible visible
    assert layer._h_targets[0].visible is True

    # Simulation clic : décocher 'cloud cible' en pause
    gui.checkboxes[0].value = False
    gui.checkboxes[0].trigger()

    assert layer._h_targets[0].visible is False, \
        "toggle en pause : cloud cible doit être masqué immédiatement"
    # Les autres handles ne doivent pas être affectés
    assert layer._h_achieved[0].visible is True


def test_toggle_witness_cible_paused():
    """Décocher 'witness cible' (index 2) en pause → masquage immédiat du witness cible."""
    layer, gui = _build_layer_capture(cb_val=True, n_objects=1)
    frame = _make_frame(solved=True, targets=True, pose=True)
    ui = UiState(channel="ground", color_mode="distance", point_size=0.01)

    layer.update(frame, ui)

    assert layer._h_wit_targets[0].visible is True, "prérequis : witness cible visible"

    gui.checkboxes[2].value = False
    gui.checkboxes[2].trigger()

    assert layer._h_wit_targets[0].visible is False, \
        "bascule pause : witness cible doit être masqué"
    assert layer._h_wit_achieved[0].visible is True   # witness atteint non affecté


def test_toggle_noop_before_first_update():
    """Déclencher le callback avant tout update() (last_frame=None) → ne lève pas."""
    layer, gui = _build_layer_capture(cb_val=True, n_objects=1)

    # Aucun update() encore appelé
    gui.checkboxes[0].value = False
    gui.checkboxes[0].trigger()   # ne doit pas lever


# =============================================================================
# 8. Normales de contact — chemin nominal + gardes + toggle
# =============================================================================

def test_happy_path_normal_segments_visible():
    """Chemin nominal avec sondes actives → handles normales peuplés et visibles.

    _field() a active[0]=[T,T,F,F] → 2 sondes actives et direction[...,2]=1 sur le canal ground.
    Avec cloud=0, R_sol=I, length=0.05 : chaque segment = [[0,0,0],[0,0,0.05]].
    """
    layer = _build_layer(cb_val=True, n_objects=1)
    frame = _make_frame(solved=True, targets=True, pose=True)
    ui = UiState(channel="ground", color_mode="distance", point_size=0.01)

    layer.update(frame, ui)

    assert layer._h_nrm_targets[0].points is not None
    assert layer._h_nrm_targets[0].points.shape == (2, 2, 3)
    assert layer._h_nrm_targets[0].visible is True
    assert layer._h_nrm_achieved[0].points is not None
    assert layer._h_nrm_achieved[0].points.shape == (2, 2, 3)
    assert layer._h_nrm_achieved[0].visible is True
    # Extrémité = sonde + [0,0,1]*0.05 = [0,0,0.05]
    np.testing.assert_allclose(layer._h_nrm_targets[0].points[0, 1], [0.0, 0.0, 0.05], atol=1e-5)


def test_normal_cb_false_makes_handles_not_visible():
    """Checkboxes normales désactivées (cb_val=False) → handles normales non visibles."""
    layer = _build_layer(cb_val=False, n_objects=1)
    frame = _make_frame(solved=True, targets=True, pose=True)
    ui = UiState(channel="ground", color_mode="distance", point_size=0.01)

    layer.update(frame, ui)

    assert layer._h_nrm_targets[0].visible is False
    assert layer._h_nrm_achieved[0].visible is False


def test_normal_solved_none_hides_all():
    """solved=None → handles normales masqués, aucune levée."""
    layer = _build_layer(n_objects=1)
    frame = _make_frame(solved=False)
    ui = UiState(channel="ground", color_mode="distance", point_size=0.01)

    layer.update(frame, ui)

    assert layer._h_nrm_targets[0].visible is False
    assert layer._h_nrm_achieved[0].visible is False


def test_normal_contact_achieved_none_hides_achieved_normal_only():
    """contact_achieved=None → normale cible visible, normale atteint masquée."""
    layer = _build_layer(cb_val=True, n_objects=1)
    frame = _make_frame(solved=True, targets=True, pose=True, contact_achieved=False)
    ui = UiState(channel="ground", color_mode="distance", point_size=0.01)

    layer.update(frame, ui)

    assert layer._h_nrm_targets[0].visible is True     # cible : données présentes → visible
    assert layer._h_nrm_achieved[0].visible is False   # atteint : contact_achieved absent → masquée


def test_fewer_objects_masks_extra_normals():
    """Deux handles créés, frame avec 1 cloud → le 2e handle normale est masqué."""
    layer = _build_layer(n_objects=2, cb_val=True)

    single_cloud = (np.zeros((4, 3), np.float32),)
    tgt_field = _field(C=2, P=4)
    env_int_ns = types.SimpleNamespace(per_object=(tgt_field,))
    tgts_ns = types.SimpleNamespace(env_interaction=env_int_ns)
    pose_ns = types.SimpleNamespace(
        object_rot=np.tile(np.eye(3), (1, 1, 1)),
        object_pos=np.zeros((1, 3)),
    )
    object_poses = np.zeros((1, 7))
    object_poses[0, 3] = 1.0
    env_evals = (types.SimpleNamespace(field=tgt_field),)
    ach_ns = types.SimpleNamespace(env=env_evals)
    solved_ns = types.SimpleNamespace(object_poses=object_poses, contact_achieved=ach_ns)

    frame = types.SimpleNamespace(
        solved=solved_ns, targets=tgts_ns, pose=pose_ns, object_clouds_world=single_cloud,
    )
    ui = UiState(channel="ground", color_mode="uniform", point_size=0.01)
    layer.update(frame, ui)

    # Objet k=1 : absent → normales masquées sans levée
    assert layer._h_nrm_targets[1].visible is False
    assert layer._h_nrm_achieved[1].visible is False


def test_toggle_normale_cible_paused():
    """Décocher 'normales cible' (index 4) en pause → masquage immédiat.

    Ordre des checkboxes créées par setup() :
      0 = 'cloud cible'    1 = 'cloud atteint'
      2 = 'witness cible'  3 = 'witness atteint'
      4 = 'normales cible' 5 = 'normales atteint'
    """
    layer, gui = _build_layer_capture(cb_val=True, n_objects=1)
    frame = _make_frame(solved=True, targets=True, pose=True)
    ui = UiState(channel="ground", color_mode="distance", point_size=0.01)

    layer.update(frame, ui)

    assert layer._h_nrm_targets[0].visible is True, "prérequis : normale cible visible"

    gui.checkboxes[4].value = False
    gui.checkboxes[4].trigger()

    assert layer._h_nrm_targets[0].visible is False, \
        "bascule pause : normale cible doit être masquée"
    assert layer._h_nrm_achieved[0].visible is True   # normale atteint non affectée
