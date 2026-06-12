"""Colour helpers for visualising contact / SDF signed-distance fields."""
from __future__ import annotations

import numpy as np

_RED = np.array([220, 40, 40], dtype=np.float64)     # penetration (d <= -margin)
_WHITE = np.array([245, 245, 245], dtype=np.float64)  # contact (d ~ 0)
_BLUE = np.array([60, 110, 230], dtype=np.float64)    # far / approaching (d >= +margin)


def signed_distance_colors(dist: np.ndarray, margin: float) -> np.ndarray:
    """Diverging colormap on [-margin, +margin]: red (penetration) -> white (contact)
    -> blue (far). Returns (N, 3) uint8. Values outside the band are clamped."""
    t = np.clip((np.asarray(dist, dtype=np.float64) + margin) / (2.0 * margin), 0.0, 1.0)
    lo = (t < 0.5)[:, None]
    a = (t / 0.5)[:, None]                 # 0 -> red, 1 -> white  (t in [0, 0.5])
    b = ((t - 0.5) / 0.5)[:, None]         # 0 -> white, 1 -> blue (t in [0.5, 1])
    col = np.where(lo, _RED * (1 - a) + _WHITE * a, _WHITE * (1 - b) + _BLUE * b)
    return col.astype(np.uint8)
