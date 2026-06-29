"""SceneScaleConfig + resolve_scale/apply_scene_scale : la similarité de scène partagée."""
import numpy as np
import pytest

from src.targets.config import SceneScaleConfig, TargetsConfig, StyleConfig
from src.targets.scale import resolve_scale, apply_scene_scale


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
