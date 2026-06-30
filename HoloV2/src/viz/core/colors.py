"""Colormaps pour les couches viz — la SEULE source des mappages heat / diverging / parity / active
(portent les maths dupliquées à travers viewer.py / cloud.py / sdf.py). Pur numpy, sans viser : une entrée
connue s'applique à un uint8 RGB connu, donc chaque colormap est testée directement. Tout retourne (P, 3) uint8."""
from __future__ import annotations

import numpy as np

# Ancres heatmap distance (uint8 RGB), portées depuis viewer.py:_heat_distance.
_NEAR = np.array([40, 90, 255], np.float64)    # Près/pénétrant (d <= 0) -> bleu
_FAR = np.array([255, 60, 50], np.float64)     # Loin dans la marge (d ~ marge) -> rouge

# Couleurs de repère par axe (x, y, z), portées depuis viewer.py:_AXIS_COLORS.
AXIS_COLORS = np.array([[255, 80, 80], [80, 230, 80], [80, 130, 255]], np.uint8)


def heat_distance(dist: np.ndarray, margin: float) -> np.ndarray:
    """Distance signée (P,) -> (P, 3) uint8. Bleu près/pénétrant (d<=0) -> rouge loin (d ~ marge)."""
    t = np.clip(np.asarray(dist, np.float64) / max(float(margin), 1e-9), 0.0, 1.0)[:, None]
    return (t * _FAR + (1.0 - t) * _NEAR).astype(np.uint8)


def active_mask(active: np.ndarray) -> np.ndarray:
    """Booléen (P,) -> (P, 3) uint8. Vert brillant où actif (dans la bande de contact), gris éteint ailleurs."""
    a = np.asarray(active, bool)
    col = np.tile(np.array([70, 70, 80], np.uint8), (len(a), 1))
    col[a] = (90, 255, 130)
    return col


def diverging(signed: np.ndarray, vmax: float) -> np.ndarray:
    """Valeur signée (P,) -> (P, 3) uint8. -vmax = bleu, 0 = blanc, +vmax = rouge (limité)."""
    t = np.clip(np.asarray(signed, np.float64) / max(float(vmax), 1e-9), -1.0, 1.0)
    col = np.ones((len(t), 3), np.float64)                          # Blanc pour t = 0
    neg = t < 0
    a = (-t[neg])[:, None]
    col[neg] = (1 - a) * np.array([1, 1, 1]) + a * np.array([0.20, 0.35, 1.0])    # -> Bleu
    b = (t[~neg])[:, None]
    col[~neg] = (1 - b) * np.array([1, 1, 1]) + b * np.array([1.0, 0.25, 0.20])   # -> Rouge
    return (col * 255).astype(np.uint8)


def parity(err: np.ndarray, vmax: float) -> np.ndarray:
    """Erreur non-négative (P,) -> (P, 3) uint8. Bleu (0) -> rouge (>= vmax). Porté depuis cloud.py:_heat."""
    t = np.clip(np.asarray(err, np.float64) / max(float(vmax), 1e-9), 0.0, 1.0)[:, None]
    return (np.concatenate([t, np.zeros_like(t), 1.0 - t], axis=1) * 255).astype(np.uint8)
