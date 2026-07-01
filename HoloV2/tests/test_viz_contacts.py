"""Tests de la couche contacts (roadmap #3).

Trois familles :
  1. ``contact_colors`` — fonction pure, sans écran, sans torch. On vérifie forme/dtype,
     que 'distance' produit des couleurs distinctes pour des distances différentes, et que
     'active' sépare actifs/inactifs.
  2. ``ContactsLayer.update()`` — appelé sur des fakes duck-typés (SimpleNamespace + classes
     légères). Couvre le chemin nominal ET les gardes de données manquantes (solved=None,
     targets/robot_interaction absents, canal inconnu).
  3. Lignes witness (cible + atteint) : segments peuplés sur chemin nominal, mapping de pose
     correct (source pour cible, résolue pour atteint), gardes de données, toggle en pause."""
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


class _SegHandle:
    """Poignée de segment de ligne factice — accepte les setters witness de ContactsLayer."""

    def __init__(self):
        self.visible = True
        self.points = None
        self.colors = None


class _Cb:
    """Checkbox factice."""

    def __init__(self, val: bool = True):
        self.value = val

    def on_update(self, callback) -> None:
        """No-op : les tests existants n'exercent pas le chemin on_update."""
        pass


class _CaptureCb:
    """Checkbox factice capturant les callbacks on_update (simulation d'un clic en pause)."""

    def __init__(self, val: bool = True) -> None:
        self.value = val
        self._callbacks: list = []

    def on_update(self, cb) -> None:
        """Enregistre le callback pour invocation manuelle par le test."""
        self._callbacks.append(cb)

    def trigger(self) -> None:
        """Simule un clic utilisateur : invoque tous les callbacks enregistrés."""
        for cb in self._callbacks:
            cb(None)


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


class _FakeScene:
    """Scene factice : add_point_cloud retourne un _Handle, add_line_segments un _SegHandle."""

    def add_point_cloud(self, name, pts, cols, point_size=None):
        return _Handle()

    def add_line_segments(self, name, segs, cols, line_width=None):
        return _SegHandle()


class _FakeServer:
    """Serveur viser factice."""

    def __init__(self):
        self.scene = _FakeScene()


def _make_ctx(channel_names: tuple = ("ground", "obj0")):
    """Contexte viz minimal duck-typé avec channels portant object_idx.

    Convention : canal 0 = ground (object_idx=None), canal c >= 1 = objet c-1.
    """
    channels = tuple(
        types.SimpleNamespace(object_idx=(None if i == 0 else i - 1))
        for i in range(len(channel_names))
    )
    return types.SimpleNamespace(channel_names=channel_names, margin=0.1, channels=channels)


def _make_frame(*, solved: bool = True, targets: bool = True,
                robot_interaction: bool = True,
                contact_achieved: bool = True, C: int = 2, M: int = 4):
    """VizFrame factice duck-typé couvrant les combinaisons de gardes.

    Paramètres :
        solved              : si False, ``frame.solved`` est None (solve-gated).
        targets             : si False, ``frame.targets`` est None.
        robot_interaction   : si False, ``frame.targets.robot_interaction`` est None.
        contact_achieved    : si False, ``frame.solved.contact_achieved`` est None.

    Note : ``frame.pose`` n'est pas inclus — ces tests portent sur le canal ground (oi=None)
    et la production ne l'accède donc jamais (accès conditionnel à oi is not None).
    """
    # Champ partagé pour cible et atteint
    tgt_field = _field(C, M)
    ach_ns = types.SimpleNamespace(field=tgt_field) if contact_achieved else None
    solved_ns = (
        types.SimpleNamespace(
            robot_points_world=np.zeros((M, 3), np.float32),
            contact_achieved=ach_ns,
            object_poses=np.zeros((1, 7)),  # jamais accédé (oi=None pour ground)
        )
        if solved else None
    )
    ri_ns = types.SimpleNamespace(field=tgt_field) if robot_interaction else None
    tgts_ns = types.SimpleNamespace(robot_interaction=ri_ns) if targets else None
    # pose.object_rot/pos : jamais accédés sur le canal ground (oi=None)
    pose_ns = types.SimpleNamespace(
        object_rot=np.tile(np.eye(3), (1, 1, 1)),  # (1, 3, 3) identité
        object_pos=np.zeros((1, 3)),
    )
    return types.SimpleNamespace(solved=solved_ns, targets=tgts_ns, pose=pose_ns)


def _make_frame_obj(
    R_src: np.ndarray,
    t_src: np.ndarray,
    R_sol: np.ndarray,
    t_sol: np.ndarray,
    wit_local: np.ndarray | None = None,
    active_mask: np.ndarray | None = None,
    C: int = 2,
    M: int = 4,
) -> types.SimpleNamespace:
    """VizFrame duck-typé pour test du canal objet avec poses source et résolue spécifiques.

    Le canal 1 (obj0) porte le witness et les activations indiqués.
    ``object_poses`` est construit avec une rotation identité qw=1 (simplifie le test).
    Pour une pose résolue quelconque : seule la translation est variable ici.

    Paramètres
    ----------
    R_src, t_src : pose objet SOURCE (cible witness mappé via cette pose).
    R_sol, t_sol : pose objet RÉSOLUE (atteint witness mappé via cette pose).
                   NOTE : on suppose R_sol = I dans les tests (quaternion identité).
    wit_local     : (M, 3) witness local pour canal 1 ; zeros par défaut.
    active_mask   : (M,) booléen pour canal 1 ; premier seul actif par défaut.
    """
    dist = np.zeros((C, M))
    direction = np.zeros((C, M, 3))
    direction[..., 2] = 1.0
    witness = np.zeros((C, M, 3))
    if wit_local is not None:
        witness[1] = wit_local
    active = np.zeros((C, M), bool)
    if active_mask is not None:
        active[1] = active_mask
    else:
        if M > 0:
            active[1, 0] = True   # au moins une sonde active

    tgt_field = MultiChannelField(
        distance=dist, direction=direction, witness=witness,
        active=active, channels=("ground", "obj0"),
    )

    # object_poses : format (N, 7) = [x, y, z, qw, qx, qy, qz]
    # On utilise R_sol = I → quaternion identité wxyz = [1, 0, 0, 0]
    object_poses = np.zeros((1, 7))
    object_poses[0, :3] = t_sol
    object_poses[0, 3] = 1.0   # qw = 1 (identité)

    ach_ns = types.SimpleNamespace(field=tgt_field)
    solved_ns = types.SimpleNamespace(
        robot_points_world=np.zeros((M, 3), np.float32),
        contact_achieved=ach_ns,
        object_poses=object_poses,
    )
    ri_ns = types.SimpleNamespace(field=tgt_field)
    tgts_ns = types.SimpleNamespace(robot_interaction=ri_ns)
    pose_ns = types.SimpleNamespace(
        object_rot=R_src[np.newaxis],   # (1, 3, 3)
        object_pos=t_src[np.newaxis],   # (1, 3)
    )
    return types.SimpleNamespace(solved=solved_ns, targets=tgts_ns, pose=pose_ns)


def _build_layer(cb_val: bool = True, channel_names: tuple = ("ground", "obj0")) -> ContactsLayer:
    """Construit et initialise une ContactsLayer sur des fakes."""
    layer = ContactsLayer()
    layer.setup(_FakeServer(), _FakeGui(cb_val=cb_val), _make_ctx(channel_names))
    return layer


def _build_layer_capture(cb_val: bool = True,
                          channel_names: tuple = ("ground", "obj0")) -> tuple:
    """Construit une ContactsLayer avec GUI capturante (pour tests toggle en pause).

    Retourne (layer, gui) pour que les tests accèdent aux checkboxes capturées.
    """
    gui = _CaptureGui(cb_val=cb_val)
    layer = ContactsLayer()
    layer.setup(_FakeServer(), gui, _make_ctx(channel_names))
    return layer, gui


# =============================================================================
# 3. Tests d'update() — chemin nominal + gardes (nuages de points)
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


# =============================================================================
# 4. Tests witness — chemin nominal (canal ground)
# =============================================================================

def test_witness_target_handle_populated_and_visible_on_happy_path():
    """Chemin nominal canal ground : _h_wit_target peuplé (≥1 sonde active) et visible=True."""
    # _field() a active[0]=[T,T,F,F] -> 2 sondes actives sur le canal ground
    layer = _build_layer(cb_val=True)
    frame = _make_frame(solved=True, targets=True)
    ui = UiState(channel="ground", color_mode="distance", point_size=0.01)

    layer.update(frame, ui)

    assert layer._h_wit_target.points is not None, "_h_wit_target.points doit être défini"
    assert layer._h_wit_target.points.shape == (2, 2, 3), "2 segments (S=2 actives)"
    assert layer._h_wit_target.visible is True


def test_witness_achieved_handle_populated_and_visible_on_happy_path():
    """Chemin nominal canal ground : _h_wit_achieved peuplé (≥1 sonde active) et visible=True."""
    layer = _build_layer(cb_val=True)
    frame = _make_frame(solved=True, targets=True)
    ui = UiState(channel="ground", color_mode="distance", point_size=0.01)

    layer.update(frame, ui)

    assert layer._h_wit_achieved.points is not None, "_h_wit_achieved.points doit être défini"
    assert layer._h_wit_achieved.points.shape == (2, 2, 3), "2 segments (S=2 actives)"
    assert layer._h_wit_achieved.visible is True


def test_witness_cb_false_makes_handles_not_visible():
    """Checkboxes witness désactivées (cb_val=False) -> handles witness non visibles."""
    layer = _build_layer(cb_val=False)
    frame = _make_frame(solved=True, targets=True)
    ui = UiState(channel="ground", color_mode="distance", point_size=0.01)

    layer.update(frame, ui)

    assert layer._h_wit_target.visible is False
    assert layer._h_wit_achieved.visible is False


# =============================================================================
# 5. Tests witness — mapping de pose (canal objet)
# =============================================================================

def test_witness_cible_uses_source_pose_atteint_uses_solved_pose():
    """Canal objet : cible mappé via pose SOURCE, atteint via pose RÉSOLUE.

    Vérifie que les endpoints witness diffèrent selon la pose utilisée :
    - Pose source : R=I, t=[5, 0, 0]  → endpoint cible = [1,0,0]+[5,0,0] = [6,0,0]
    - Pose résolue : R=I, t=[20, 0, 0] → endpoint atteint = [1,0,0]+[20,0,0] = [21,0,0]
    """
    R_src = np.eye(3)
    t_src = np.array([5.0, 0.0, 0.0])
    R_sol = np.eye(3)   # rotation identité → quaternion wxyz = [1,0,0,0]
    t_sol = np.array([20.0, 0.0, 0.0])

    # Witness local du canal obj0 (canal 1) : [1, 0, 0] pour la sonde 0 (seule active)
    wit_local = np.zeros((4, 3))
    wit_local[0] = [1.0, 0.0, 0.0]
    active_mask = np.array([True, False, False, False])

    frame = _make_frame_obj(R_src, t_src, R_sol, t_sol, wit_local, active_mask)
    layer = _build_layer(cb_val=True)
    ui = UiState(channel="obj0", color_mode="distance", point_size=0.01)

    layer.update(frame, ui)

    # Endpoint cible : [1,0,0] @ I.T + [5,0,0] = [6,0,0]
    assert layer._h_wit_target.points is not None, "segments cible doivent être définis"
    assert layer._h_wit_target.points.shape == (1, 2, 3)
    np.testing.assert_allclose(layer._h_wit_target.points[0, 1], [6.0, 0.0, 0.0], atol=1e-5,
                                err_msg="endpoint witness CIBLE doit utiliser la pose SOURCE")

    # Endpoint atteint : [1,0,0] @ I.T + [20,0,0] = [21,0,0]
    assert layer._h_wit_achieved.points is not None, "segments atteint doivent être définis"
    assert layer._h_wit_achieved.points.shape == (1, 2, 3)
    np.testing.assert_allclose(layer._h_wit_achieved.points[0, 1], [21.0, 0.0, 0.0], atol=1e-5,
                                err_msg="endpoint witness ATTEINT doit utiliser la pose RÉSOLUE")


def test_witness_ground_channel_endpoints_unchanged():
    """Canal sol (R=I, t=0) : witness déjà en monde — endpoint identique au witness local."""
    # Champ ground avec un witness local non-trivial
    wit_ground = np.zeros((4, 3))
    wit_ground[0] = [3.0, 4.0, 5.0]

    # Construire un champ custom pour canal 0 (ground)
    C, M = 2, 4
    dist = np.zeros((C, M))
    direction = np.zeros((C, M, 3))
    witness = np.zeros((C, M, 3))
    witness[0] = wit_ground
    active = np.zeros((C, M), bool)
    active[0, 0] = True

    tgt_field = MultiChannelField(
        distance=dist, direction=direction, witness=witness,
        active=active, channels=("ground", "obj0"),
    )

    ach_ns = types.SimpleNamespace(field=tgt_field)
    solved_ns = types.SimpleNamespace(
        robot_points_world=np.zeros((M, 3), np.float32),
        contact_achieved=ach_ns,
        object_poses=np.zeros((1, 7)),
    )
    ri_ns = types.SimpleNamespace(field=tgt_field)
    tgts_ns = types.SimpleNamespace(robot_interaction=ri_ns)
    pose_ns = types.SimpleNamespace(
        object_rot=np.tile(np.eye(3), (1, 1, 1)),
        object_pos=np.zeros((1, 3)),
    )
    frame = types.SimpleNamespace(solved=solved_ns, targets=tgts_ns, pose=pose_ns)

    layer = _build_layer(cb_val=True)
    ui = UiState(channel="ground", color_mode="distance", point_size=0.01)
    layer.update(frame, ui)

    # Endpoint CIBLE = witness local (sol déjà en monde)
    assert layer._h_wit_target.points is not None
    np.testing.assert_allclose(layer._h_wit_target.points[0, 1], [3.0, 4.0, 5.0], atol=1e-5,
                                err_msg="canal sol : endpoint witness = witness local (monde)")


# =============================================================================
# 6. Tests witness — gardes de données manquantes
# =============================================================================

def test_witness_solved_none_hides_all_handles():
    """solved=None -> nuages ET handles witness tous masqués, aucune levée."""
    layer = _build_layer()
    frame = _make_frame(solved=False)
    ui = UiState(channel="ground", color_mode="distance", point_size=0.01)

    layer.update(frame, ui)

    assert layer._h_target.visible is False
    assert layer._h_achieved.visible is False
    assert layer._h_wit_target.visible is False
    assert layer._h_wit_achieved.visible is False


def test_witness_targets_none_hides_all_handles():
    """targets=None -> handles witness masqués comme les nuages, aucune levée."""
    layer = _build_layer()
    frame = _make_frame(targets=False)
    ui = UiState(channel="ground", color_mode="distance", point_size=0.01)

    layer.update(frame, ui)

    assert layer._h_wit_target.visible is False
    assert layer._h_wit_achieved.visible is False


def test_witness_unknown_channel_hides_all_handles():
    """Canal inconnu -> handles witness masqués, aucune levée."""
    layer = _build_layer()
    frame = _make_frame(solved=True, targets=True)
    ui = UiState(channel="canal_inconnu", color_mode="distance", point_size=0.01)

    layer.update(frame, ui)

    assert layer._h_wit_target.visible is False
    assert layer._h_wit_achieved.visible is False


def test_witness_contact_achieved_none_hides_achieved_witness_only():
    """contact_achieved=None (résolution partielle) : witness cible visible, atteint masqué.

    _h_wit_target doit rester visible (sondes actives, cb=True) ;
    _h_wit_achieved doit être masqué (contact_achieved absent).
    """
    layer = _build_layer(cb_val=True)
    frame = _make_frame(solved=True, targets=True, contact_achieved=False)
    ui = UiState(channel="ground", color_mode="distance", point_size=0.01)

    layer.update(frame, ui)

    # Witness cible : actif (2 sondes actives, cb=True) → visible
    assert layer._h_wit_target.visible is True
    # Witness atteint : contact_achieved absent → masqué
    assert layer._h_wit_achieved.visible is False


# =============================================================================
# 7. Tests witness — toggle en pause
# =============================================================================

def test_witness_target_toggle_paused():
    """Décocher 'witness cible' en pause -> masquage immédiat sans appel player.

    Ordre des checkboxes créées par setup() :
      0 = 'contact cible'  1 = 'contact atteint'
      2 = 'witness cible'  3 = 'witness atteint'
    """
    layer, gui = _build_layer_capture(cb_val=True)
    frame = _make_frame(solved=True, targets=True)
    ui = UiState(channel="ground", color_mode="distance", point_size=0.01)

    layer.update(frame, ui)

    # Prérequis : witness cible visible après update() avec données présentes + sondes actives
    assert layer._h_wit_target.visible is True, "prérequis : witness cible visible avant le clic"

    # Simulation clic utilisateur en pause : décocher 'witness cible'
    gui.checkboxes[2].value = False    # index 2 = cb_wit_target
    gui.checkboxes[2].trigger()        # invoque le callback on_update capturé

    assert layer._h_wit_target.visible is False, \
        "bascule en pause : _h_wit_target doit être masqué immédiatement"
    # Les autres handles ne doivent pas être affectés par ce toggle
    assert layer._h_target.visible is True
    assert layer._h_achieved.visible is True
    assert layer._h_wit_achieved.visible is True


def test_witness_achieved_toggle_paused():
    """Décocher 'witness atteint' en pause -> masquage immédiat (checkbox index 3)."""
    layer, gui = _build_layer_capture(cb_val=True)
    frame = _make_frame(solved=True, targets=True)
    ui = UiState(channel="ground", color_mode="distance", point_size=0.01)

    layer.update(frame, ui)

    assert layer._h_wit_achieved.visible is True, "prérequis : witness atteint visible avant le clic"

    gui.checkboxes[3].value = False    # index 3 = cb_wit_achieved
    gui.checkboxes[3].trigger()

    assert layer._h_wit_achieved.visible is False, \
        "bascule en pause : _h_wit_achieved doit être masqué immédiatement"
    # Les autres handles ne doivent pas être affectés
    assert layer._h_wit_target.visible is True
    assert layer._h_target.visible is True
    assert layer._h_achieved.visible is True


def test_witness_toggle_noop_before_first_update():
    """Déclencher le callback witness avant tout update() (last_frame=None) -> ne lève pas."""
    layer, gui = _build_layer_capture(cb_val=True)

    # Aucun update() encore appelé — le callback doit être silencieux
    gui.checkboxes[2].value = False
    gui.checkboxes[2].trigger()   # ne doit pas lever
