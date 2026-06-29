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
from .contracts import MultiChannelField


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


def scale_ground_channels(field: MultiChannelField, ground_idx: tuple[int, ...],
                          s_xy: float, s_z: float, ground_height: float = 0.0) -> MultiChannelField:
    """Scale, pour les canaux SOL (frame monde, ``ground_idx``), le ``witness`` (similarité de scène)
    et la ``distance`` (= hauteur, ``*= s_z``) là où ``active``. Les canaux OBJET (witness local,
    qui suivent la pose objet scalée), ``direction`` et ``active`` sont inchangés. Retourne un nouveau
    ``MultiChannelField`` (frozen).

    Hypothèse : SOL PLAT horizontal (witness sur le plan, normale +z). Exact pour le plan SDF par
    défaut, et pour un scale isotrope autour de l'origine/sol même sur terrain. Pour un TERRAIN avec
    un scale ANISOTROPE (``s_xy != s_z``), le witness scalé ne reste pas sur la surface et
    ``distance *= s_z`` / ``direction`` inchangée deviennent approximatifs — à revoir quand le
    terrain sera câblé."""
    distance = np.asarray(field.distance, np.float64).copy()           # (C, P)
    witness = np.asarray(field.witness, np.float64).copy()             # (C, P, 3)
    active = np.asarray(field.active, dtype=bool)                      # (C, P)
    for c in ground_idx:
        witness[c] = apply_scene_scale(witness[c], s_xy, s_z, ground_height)
        distance[c] = np.where(active[c], distance[c] * s_z, distance[c])
    return MultiChannelField(distance=distance, direction=field.direction, witness=witness,
                             active=field.active, channels=field.channels)
