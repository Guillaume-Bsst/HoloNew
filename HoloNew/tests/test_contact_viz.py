import numpy as np
from HoloNew.src.gmr_socp_v2.contact.viz import signed_distance_colors


def test_signed_distance_colors_diverging_endpoints():
    margin = 0.1
    cols = signed_distance_colors(np.array([-margin, 0.0, margin]), margin)
    assert cols.shape == (3, 3) and cols.dtype == np.uint8
    assert cols[0, 0] > cols[0, 2]      # penetration: red > blue
    assert cols[2, 2] > cols[2, 0]      # far: blue > red
    assert cols[1].min() > 200          # contact (d=0): ~white


def test_signed_distance_colors_clamps_outside_band():
    margin = 0.1
    cols = signed_distance_colors(np.array([-10.0, 10.0]), margin)
    # values beyond the band clamp to the red / blue endpoints (no overflow).
    assert cols[0, 0] > cols[0, 2]
    assert cols[1, 2] > cols[1, 0]
