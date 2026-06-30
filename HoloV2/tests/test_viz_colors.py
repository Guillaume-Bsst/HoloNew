"""colors — known input -> known uint8 RGB (ports viewer.py/_heat_distance/_active_colors,
cloud.py/_heat, sdf.py/_diverging into one module)."""
import numpy as np

from src.viz.core.colors import AXIS_COLORS, active_mask, diverging, heat_distance, parity


def test_heat_distance_anchors():
    # d <= 0 -> NEAR (blue) ; d == margin -> FAR (red) ; clamped both ends.
    out = heat_distance(np.array([-1.0, 0.0, 0.05]), 0.05)
    assert out.dtype == np.uint8 and out.shape == (3, 3)
    assert np.array_equal(out[0], [40, 90, 255])     # clamped to NEAR
    assert np.array_equal(out[1], [40, 90, 255])     # d=0 -> NEAR
    assert np.array_equal(out[2], [255, 60, 50])     # d=margin -> FAR


def test_active_mask():
    out = active_mask(np.array([True, False]))
    assert out.dtype == np.uint8
    assert np.array_equal(out[0], [90, 255, 130])    # active -> bright green
    assert np.array_equal(out[1], [70, 70, 80])      # inactive -> dim grey


def test_diverging_white_blue_red():
    out = diverging(np.array([0.0, -1.0, 1.0]), 1.0)
    assert out.dtype == np.uint8
    assert np.array_equal(out[0], [255, 255, 255])   # 0 -> white
    assert np.array_equal(out[1], [51, 89, 255])     # -vmax -> blue
    assert np.array_equal(out[2], [255, 63, 51])     # +vmax -> red


def test_parity_blue_to_red():
    out = parity(np.array([0.0, 0.02]), 0.02)
    assert out.dtype == np.uint8
    assert np.array_equal(out[0], [0, 0, 255])       # err 0 -> blue
    assert np.array_equal(out[1], [255, 0, 0])       # err >= vmax -> red


def test_axis_colors():
    assert AXIS_COLORS.dtype == np.uint8 and AXIS_COLORS.shape == (3, 3)
    assert np.array_equal(AXIS_COLORS[0], [255, 80, 80])
