"""SceneScaleConfig + resolve_scale/apply_scene_scale : la similarité de scène partagée."""
import numpy as np
import pytest

from src.targets.config import SceneScaleConfig, TargetsConfig, StyleConfig
from src.targets.scale import resolve_scale, apply_scene_scale, scale_object_trajectory


def test_config_defaults_and_validation():
    c = SceneScaleConfig()
    assert c.scale_xy is None and c.scale_z is None          # défaut = None -> ratio
    assert TargetsConfig().scene_scale == SceneScaleConfig()
    assert TargetsConfig().style == StyleConfig()
    with pytest.raises(ValueError):
        SceneScaleConfig(scale_xy=0.0)                        # facteur explicite doit être > 0
    with pytest.raises(ValueError):
        SceneScaleConfig(scale_z=-1.0)


def test_resolve_scale_none_is_ratio():
    assert resolve_scale(SceneScaleConfig(), ratio=0.5) == (0.5, 0.5)          # None,None -> ratio
    assert resolve_scale(SceneScaleConfig(scale_xy=1.0), ratio=0.5) == (1.0, 0.5)  # xy fixe, z=ratio
    assert resolve_scale(SceneScaleConfig(scale_xy=1.0, scale_z=2.0), 0.5) == (1.0, 2.0)


def test_apply_scene_scale_anchor_and_axes():
    pts = np.array([[2.0, 4.0, 6.0], [1.0, 0.0, 0.0]], np.float64)
    out = apply_scene_scale(pts, s_xy=0.5, s_z=0.25, ground_height=0.0)
    np.testing.assert_allclose(out, [[1.0, 2.0, 1.5], [0.5, 0.0, 0.0]])
    assert out.dtype == np.float64
    np.testing.assert_allclose(pts[0], [2.0, 4.0, 6.0])      # input non muté (copie)


def test_apply_scene_scale_z_anchored_on_ground():
    # un point SUR le sol reste sur le sol (z invariant) ; xy scalé autour de l'origine.
    pts = np.array([[3.0, 3.0, 0.2]], np.float64)
    out = apply_scene_scale(pts, s_xy=1.0, s_z=0.5, ground_height=0.2)
    np.testing.assert_allclose(out, [[3.0, 3.0, 0.2]])       # z == ground_height -> inchangé


def test_scale_object_trajectory_frame0_invariance():
    # 2 objets posés à leur hauteur de référence (z == object_z0) : un objet rigide de taille fixe
    # framé hors de son point de contact ne doit PAS s'enfoncer quand on scale s_z. Ancre = frame 0.
    object_z0 = np.array([0.3, 0.8], np.float64)                          # (2,)
    object_pos = np.array([[2.0, 4.0, 0.3], [1.0, -2.0, 0.8]], np.float64)  # (2, 3) z == z0
    out = scale_object_trajectory(object_pos, object_z0, s_xy=0.5, s_z=0.5)
    # z reste EXACTEMENT z0 (anti-enfoncement ; l'ancre-sol donnerait 0.15 / 0.40)
    np.testing.assert_allclose(out[:, 2], [0.3, 0.8], atol=1e-12)
    # xy scalés autour de l'origine (* s_xy)
    np.testing.assert_allclose(out[:, 0], [1.0, 0.5], atol=1e-12)
    np.testing.assert_allclose(out[:, 1], [2.0, -1.0], atol=1e-12)


def test_scale_object_trajectory_scaled_lift():
    # déviation temporelle par rapport à la frame 0 scalée : z0 + (z - z0) * s_z
    object_z0 = np.array([0.3], np.float64)
    object_pos = np.array([[0.0, 0.0, 1.0]], np.float64)
    out = scale_object_trajectory(object_pos, object_z0, s_xy=1.0, s_z=0.5)
    np.testing.assert_allclose(out[:, 2], [0.65], atol=1e-12)             # 0.3 + (1.0-0.3)*0.5


def test_scale_object_trajectory_no_mutation():
    object_z0 = np.array([0.3], np.float64)
    object_pos = np.array([[2.0, 4.0, 1.0]], np.float64)
    ref = object_pos.copy()
    out = scale_object_trajectory(object_pos, object_z0, s_xy=0.5, s_z=0.5)
    np.testing.assert_array_equal(object_pos, ref)                        # entrée non mutée
    assert out.dtype == np.float64


def test_scale_object_trajectory_empty():
    object_z0 = np.empty((0,), np.float64)
    object_pos = np.empty((0, 3), np.float64)
    out = scale_object_trajectory(object_pos, object_z0, s_xy=0.5, s_z=0.5)
    assert out.shape == (0, 3)                                            # N=0 : pas d'erreur


def test_scale_ground_channels():
    from src.targets.contracts import MultiChannelField
    from src.targets.scale import scale_ground_channels
    C, P = 2, 3
    distance = np.array([[0.1, 0.2, 5.0], [1.0, 1.0, 1.0]], np.float64)     # canal 0 = sol, 1 = objet
    witness = np.zeros((C, P, 3), np.float64)
    witness[0] = [[2.0, 4.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]        # sol : frame monde
    witness[1] = [[9.0, 9.0, 9.0]] * P                                       # objet : local, NE DOIT PAS bouger
    direction = np.tile(np.array([0.0, 0.0, 1.0]), (C, P, 1))
    active = np.array([[True, True, False], [True, True, True]])
    f = MultiChannelField(distance=distance, direction=direction, witness=witness,
                          active=active, channels=("ground", "obj0"))
    out = scale_ground_channels(f, ground_idx=(0,), s_xy=0.5, s_z=0.25, ground_height=0.0)
    # sol : witness xy * 0.5, z sur le plan ; distance * 0.25 là où active (la 3e inactive inchangée)
    np.testing.assert_allclose(out.witness[0], [[1.0, 2.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(out.distance[0], [0.025, 0.05, 5.0])         # 3e (inactive) inchangée
    # canal objet intact ; direction/active intacts
    np.testing.assert_array_equal(out.witness[1], witness[1])
    np.testing.assert_allclose(out.distance[1], [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(out.active, active)
    np.testing.assert_allclose(out.direction, direction)
