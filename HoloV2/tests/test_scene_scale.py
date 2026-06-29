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
