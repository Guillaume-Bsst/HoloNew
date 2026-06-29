"""scale — la similarité de scène PARTAGÉE par ``style`` et ``interaction`` (placement des refs).

``resolve_scale`` résout ``None -> ratio`` ; ``apply_scene_scale`` applique la similarité diagonale
(xy autour de l'origine monde, z autour du sol). Pur, float64, torch-free ; aucune mutation de
l'entrée. ``style/build.py`` n'appelle PAS ``apply_scene_scale`` directement (le morphologique du
pelvis est entrelacé à son placement) : il partage ``resolve_scale`` + la convention d'ancre, et
applique les facteurs dans sa propre formule de root. ``interaction`` (trajectoire objet + witness
sol) utilise ``apply_scene_scale``.
"""
from __future__ import annotations

import numpy as np

from .config import SceneScaleConfig


def resolve_scale(cfg: SceneScaleConfig, ratio: float) -> tuple[float, float]:
    """``(s_xy, s_z)`` : chaque axe ``None`` -> ``ratio`` (= stature / human_height_assumption)."""
    s_xy = ratio if cfg.scale_xy is None else cfg.scale_xy
    s_z = ratio if cfg.scale_z is None else cfg.scale_z
    return s_xy, s_z


def apply_scene_scale(points: np.ndarray, s_xy: float, s_z: float,
                      ground_height: float = 0.0) -> np.ndarray:
    """``(..., 3)`` -> copie scalée : ``x,y *= s_xy`` (autour de 0), ``z`` autour de ``ground_height``
    (un point sur le sol reste sur le sol). Pur, float64, entrée non mutée."""
    out = np.asarray(points, np.float64).copy()
    out[..., 0] *= s_xy
    out[..., 1] *= s_xy
    out[..., 2] = ground_height + (out[..., 2] - ground_height) * s_z
    return out
