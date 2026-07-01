"""scale — la similarité de scène PARTAGÉE par ``style`` et ``interaction`` (placement des refs).

``resolve_scale`` résout ``None -> ratio``. DEUX ancres Z distinctes selon ce qu'on scale :
 * ``apply_scene_scale`` — ancre le z sur le SOL (xy autour de 0, z autour de ``ground_height`` ; un
   point sur le sol reste sur le sol). C'est le WITNESS SOL, appliqué via ``scale_ground_channels``.
 * ``scale_object_trajectory`` — ancre le z sur la HAUTEUR DE FRAME 0 par objet (seule la déviation
   temporelle est scalée). C'est la TRAJECTOIRE OBJET : un objet rigide de taille fixe framé hors de son
   point de contact resterait posé (l'ancre sol l'enfoncerait). Miroir V1 ``scale_object_poses_to_center``.
Pur, float64, torch-free ; aucune mutation de l'entrée. ``style/build.py`` n'appelle PAS ces fonctions
directement (le morphologique du pelvis est entrelacé à son placement) : il partage ``resolve_scale`` +
la convention d'ancre, et applique les facteurs dans sa propre formule de root.
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


def scale_object_trajectory(object_pos: np.ndarray, object_z0: np.ndarray,
                            s_xy: float, s_z: float) -> np.ndarray:
    """``(N, 3)`` centres objets -> copie scalée (placement de scène). XY autour de l'origine
    (``*= s_xy``) ; Z autour de la hauteur de RÉFÉRENCE PAR OBJET ``object_z0`` (la frame 0), on ne
    scale que la déviation temporelle : ``z0 + (z - z0) * s_z``. Distinct d'``apply_scene_scale``
    (ancre SOL) : un objet rigide de taille FIXE framé hors de son point de contact resterait posé —
    l'ancre sol l'enfoncerait de ``(1-s_z)*demi_hauteur``. Miroir V1
    ``holosoma/preprocess.scale_object_poses_to_center`` (ancre = frame 0 ; suppose l'objet au repos à
    la frame 0, sinon ``z0 != hauteur de contact`` — limitation héritée de V1). Cette MÊME hypothèse
    (``z0 == demi-hauteur``) garde l'ancre cohérente avec le canal objet↔sol de ``scale_ground_channels``
    (``distance *= s_z``) : les deux ne concordent sur la hauteur du point bas que si l'objet repose à
    f0. Pur, float64, entrée non mutée."""
    out = np.asarray(object_pos, np.float64).copy()          # (N, 3)
    z0 = np.asarray(object_z0, np.float64)                    # (N,)
    out[..., 0] *= s_xy
    out[..., 1] *= s_xy
    out[..., 2] = z0 + (out[..., 2] - z0) * s_z
    return out


def scale_ground_channels(field: MultiChannelField, ground_idx: tuple[int, ...],
                          s_xy: float, s_z: float, ground_height: float = 0.0) -> MultiChannelField:
    """Scale, pour les canaux SOL (frame monde, ``ground_idx``), le ``witness`` (similarité de scène)
    et la ``distance`` (= hauteur, ``*= s_z``) là où ``active``. Les canaux OBJET (witness local,
    qui suivent la pose objet scalée), ``direction`` et ``active`` sont inchangés. Retourne un nouveau
    ``MultiChannelField`` (frozen).

    NB : pour un canal OBJET↔sol, ce ``distance *= s_z`` (écart au sol) ne concorde avec la pose objet
    scalée par ``scale_object_trajectory`` (ancre frame 0) que si l'objet repose à f0 (``z0 == demi-hauteur``).

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
