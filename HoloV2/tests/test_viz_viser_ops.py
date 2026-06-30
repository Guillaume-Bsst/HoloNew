"""viser_ops — les helpers purs (quat->R, hide) testés sans serveur viser. hide() est vérifié
avec un handle duck-typé, donc pas besoin d'écran."""
import numpy as np

from src.viz.core.viser_ops import hide, quat_wxyz_to_R


def test_quat_identity():
    """Test d'un quaternion identité (wxyz)."""
    R = quat_wxyz_to_R(np.array([[1.0, 0.0, 0.0, 0.0]]))     # identité wxyz
    assert R.shape == (1, 3, 3)
    assert np.allclose(R[0], np.eye(3), atol=1e-12)


def test_quat_90deg_about_z():
    """Test d'une rotation de 90° autour de l'axe z (wxyz)."""
    s = np.sqrt(0.5)
    R = quat_wxyz_to_R(np.array([[s, 0.0, 0.0, s]]))         # 90 deg autour de +z (wxyz)
    expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(R[0], expected, atol=1e-9)


def test_hide_sets_visible_false():
    """Test que hide() met visible à False sur un handle."""
    class _H:
        visible = True
    h = _H()
    hide(h)
    assert h.visible is False
