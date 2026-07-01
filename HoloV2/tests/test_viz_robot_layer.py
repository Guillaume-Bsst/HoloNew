"""Tests unitaires pour ``RobotLayer`` — parties pures testables sans serveur ni URDF réel.

Couvre :
  1. Conformité structurelle (isinstance Layer + dossier GUI).
  2. Réordonnance quaternion pinocchio xyzw → viser wxyz (valeur connue — risque #1).
  3. Comportement no-op quand ``frame.solved is None`` (masquage, pas de crash).
  4. Chemin nominal (solved présent, toggle actif) : position, wxyz, cfg joints mis à jour.

Ce qui N'est PAS testé ici (nécessite un vrai serveur viser) :
  - ``setup()`` complet (ViserUrdf + yourdfpy + serveur graphique).
  - Appel effectif de ``ViserUrdf.update_cfg`` via viser (exercé dans le smoke Phase B/Task 4).
"""
import numpy as np
import pytest

from src.viz.core.layer import Layer
from src.viz.layers.robot import RobotLayer


# ---------------------------------------------------------------------------
# Faux handles et contrôles GUI (sans viser)
# ---------------------------------------------------------------------------

class FakeUrdf:
    """Substitut ViserUrdf enregistrant les appels à update_cfg et show_visual."""

    def __init__(self, dof: int = 29) -> None:
        self.show_visual = True
        self._cfg_calls: list[np.ndarray] = []  # historique des configs reçues
        self._dof = dof

    def update_cfg(self, cfg: np.ndarray) -> None:
        """Enregistre la config joints reçue."""
        self._cfg_calls.append(np.asarray(cfg, np.float64))

    def get_actuated_joint_limits(self):
        """Retourne une liste fictive pour initialiser _dof."""
        return [None] * self._dof


class FakeBase:
    """Substitut de frame de base viser (position + wxyz écrits par update)."""

    def __init__(self) -> None:
        self.visible = True
        self.position = np.zeros(3)
        self.wxyz = np.array([1.0, 0.0, 0.0, 0.0])  # quaternion identité wxyz


class FakeCheckbox:
    """Fausse checkbox GUI dont on peut forcer la valeur dans les tests."""

    def __init__(self, value: bool = True) -> None:
        self.value = value

    def on_update(self, callback) -> None:
        """Enregistre le callback (pas d'implémentation ici)."""
        pass


class FakeSolved:
    """Substitut SolvedFrame avec un vecteur q connu."""

    def __init__(self, q: np.ndarray) -> None:
        self.q = q


class FakeFrame:
    """Substitut VizFrame : seul .solved est utilisé par RobotLayer."""

    def __init__(self, solved=None) -> None:
        self.solved = solved


class FakeUiState:
    """Substitut UiState (non consommé par RobotLayer — passé par protocole)."""


def _make_layer(dof: int = 29, toggle_value: bool = True) -> RobotLayer:
    """Construit un RobotLayer avec des faux handles injectés directement (sans setup)."""
    layer = RobotLayer()
    layer._urdf = FakeUrdf(dof)
    layer._base = FakeBase()
    layer._dof = dof
    layer._toggle = FakeCheckbox(toggle_value)
    return layer


# ---------------------------------------------------------------------------
# 1. Conformité structurelle
# ---------------------------------------------------------------------------

def test_robot_layer_is_layer():
    """RobotLayer doit satisfaire le protocole Layer (@runtime_checkable)."""
    layer = RobotLayer()
    assert isinstance(layer, Layer)


def test_robot_layer_folder():
    """Le dossier GUI doit être 'Robot (solved)'."""
    assert RobotLayer.folder == "Robot (solved)"


# ---------------------------------------------------------------------------
# 2. Réordonnance quaternion pinocchio xyzw → viser wxyz (valeur connue)
# ---------------------------------------------------------------------------

def test_quat_reorder_known_value():
    """Réordonnance xyzw → wxyz : q[3:7] = [qx, qy, qz, qw] → wxyz = [qw, qx, qy, qz].

    Cas concret : pinocchio quat xyzw = (0.1, 0.2, 0.3, 0.9) (approximativement normalisé).
    Viser attend wxyz = (0.9, 0.1, 0.2, 0.3).
    Index dans q : q[3]=qx=0.1, q[4]=qy=0.2, q[5]=qz=0.3, q[6]=qw=0.9
    → q[[6, 3, 4, 5]] = [0.9, 0.1, 0.2, 0.3].
    """
    dof = 4
    # Construction d'un q de taille 7 + dof
    q = np.zeros(7 + dof)
    q[0:3] = [1.0, 2.0, 3.0]           # position
    q[3:7] = [0.1, 0.2, 0.3, 0.9]      # xyzw pinocchio : qx, qy, qz, qw
    q[7:] = np.arange(dof, dtype=np.float64)

    layer = _make_layer(dof=dof, toggle_value=True)
    solved = FakeSolved(q)
    layer.update(FakeFrame(solved=solved), FakeUiState())

    # Vérification réordonnance
    expected_wxyz = np.array([0.9, 0.1, 0.2, 0.3])
    np.testing.assert_array_almost_equal(layer._base.wxyz, expected_wxyz,
                                         err_msg="Réordonnance xyzw→wxyz incorrecte")
    # Vérification position
    np.testing.assert_array_almost_equal(layer._base.position, [1.0, 2.0, 3.0],
                                         err_msg="Position base incorrecte")


# ---------------------------------------------------------------------------
# 3. No-op quand frame.solved is None
# ---------------------------------------------------------------------------

def test_update_solved_none_hides_robot():
    """solved=None → urdf masqué + base masquée, aucun crash."""
    layer = _make_layer(toggle_value=True)
    layer.update(FakeFrame(solved=None), FakeUiState())

    assert layer._urdf.show_visual is False, "urdf doit être masqué quand solved is None"
    assert layer._base.visible is False, "base doit être masquée quand solved is None"
    # Aucun appel update_cfg (pas de joints à mettre à jour)
    assert layer._urdf._cfg_calls == []


def test_update_solved_none_toggle_off_hides_robot():
    """solved=None + toggle désactivé → masquage identique, pas de crash."""
    layer = _make_layer(toggle_value=False)
    layer.update(FakeFrame(solved=None), FakeUiState())

    assert layer._urdf.show_visual is False
    assert layer._base.visible is False


# ---------------------------------------------------------------------------
# 4. Chemin nominal — solved présent + toggle actif
# ---------------------------------------------------------------------------

def test_update_happy_path_show():
    """Solved présent + toggle actif → robot visible, cfg joints et pose mis à jour."""
    dof = 29
    q = np.zeros(7 + dof)
    q[:3] = [0.5, 0.0, 0.8]           # position pelvis
    q[3:7] = [0.0, 0.0, 0.0, 1.0]     # quat identité xyzw : qx=qy=qz=0, qw=1
    q[7:] = np.linspace(0.1, 0.9, dof)

    layer = _make_layer(dof=dof, toggle_value=True)
    layer.update(FakeFrame(solved=FakeSolved(q)), FakeUiState())

    assert layer._urdf.show_visual is True
    assert layer._base.visible is True

    # Joints : update_cfg appelé une fois avec q[7:]
    assert len(layer._urdf._cfg_calls) == 1
    np.testing.assert_array_almost_equal(layer._urdf._cfg_calls[0], q[7:7 + dof])

    # Quaternion identité xyzw (qx=0, qy=0, qz=0, qw=1) → wxyz = (1, 0, 0, 0)
    np.testing.assert_array_almost_equal(layer._base.wxyz, [1.0, 0.0, 0.0, 0.0])


def test_update_toggle_off_solved_present_hides():
    """Toggle désactivé + solved présent → robot masqué (toggle prime)."""
    dof = 5
    q = np.zeros(7 + dof)
    q[3:7] = [0.0, 0.0, 0.0, 1.0]  # quat valide

    layer = _make_layer(dof=dof, toggle_value=False)
    layer.update(FakeFrame(solved=FakeSolved(q)), FakeUiState())

    assert layer._urdf.show_visual is False
    assert layer._base.visible is False
    # Aucun update_cfg (return anticipé car show=False)
    assert layer._urdf._cfg_calls == []
