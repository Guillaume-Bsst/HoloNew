"""Helpers de géométrie purs pour les viewers ``viz/debug`` — numpy-only, sans viser, sans scipy.

Ce module centralise les calculs NON-contractuels que les viewers (scene, cloud) inlinaient
autrefois : le point MONDE le plus bas d'un objet rigide / de l'humain (overlay de mise à terre),
la surface barycentrique de référence utilisée pour colorier le nuage par son erreur de parité
contre l'avance SMPL complète.

Règle de conception : les appelants convertissent les quaternions en matrices de rotation via
``core.viser_ops.quat_wxyz_to_R`` et passent les matrices ici — ce module reste donc viser-free
ET ne re-roule pas la conversion quat."""
from __future__ import annotations

import numpy as np


def object_world_lowz(
    verts_local: np.ndarray,
    rot: np.ndarray,
    pos: np.ndarray,
    cap: int = 8000,
) -> tuple[np.ndarray, np.ndarray]:
    """Point MONDE le plus bas par-frame d'un objet rigide.

    Paramètres
    ----------
    verts_local :
        Sommets locaux (V, 3).
    rot :
        Matrice de rotation par-frame (F, 3, 3).
    pos :
        Translation par-frame (F, 3).
    cap :
        Nombre maximal de sommets à considérer — ``verts_local`` est sous-échantillonné à ``cap``
        pour borner le coût sur les scans denses (point le plus bas quasi-exact, suffisant pour un
        marqueur débogage).

    Retourne
    --------
    min_z : np.ndarray
        Coordonnée z minimale par-frame, (F,).
    low_point : np.ndarray
        Point 3-D le plus bas dans le repère monde par-frame, (F, 3).
    """
    v = verts_local
    if v.shape[0] > cap:
        # sous-échantillonnage déterministe avec graine fixe
        v = v[np.random.default_rng(0).choice(v.shape[0], cap, replace=False)]
    # (F, V, 3) : rotation puis translation
    world = np.einsum("fij,vj->fvi", rot, v) + pos[:, None, :]
    z = world[:, :, 2]
    lo = z.argmin(axis=1)                                   # (F,) indice du sommet le plus bas
    return z.min(axis=1), world[np.arange(world.shape[0]), lo]


def lowest_point(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Point le plus bas par-frame d'un ensemble de points en mouvement.

    Paramètres
    ----------
    points :
        Nuage de points par-frame (F, P, 3).

    Retourne
    --------
    min_z : np.ndarray
        Coordonnée z minimale par-frame, (F,).
    low_point : np.ndarray
        Point 3-D le plus bas dans le repère monde par-frame, (F, 3).
    """
    z = points[:, :, 2]
    lo = z.argmin(axis=1)                                   # (F,)
    return z.min(axis=1), points[np.arange(points.shape[0]), lo]


def surface_points(
    verts: np.ndarray,
    tri_idx: np.ndarray,
    bary: np.ndarray,
) -> np.ndarray:
    """Échantillons baryentriques sur un maillage.

    Pour chacun des N échantillons, combinaison barycentrique pondérée par ``bary`` des trois
    sommets du triangle associé. C'est la référence de surface POSÉE VRAIE contre laquelle le
    viewer cloud compare son nuage mesh-free (erreur de parité).

    Paramètres
    ----------
    verts :
        Positions de tous les sommets du maillage, (V, 3).
    tri_idx :
        Indices des trois sommets de chaque triangle d'échantillon, (N, 3).
    bary :
        Coordonnées barycentriques, (N, 3), lignes sommant à 1.

    Retourne
    --------
    np.ndarray
        Positions 3-D reconstruites, (N, 3).
    """
    # verts[tri_idx] : (N, 3, 3) — les trois positions de sommet par échantillon
    return np.einsum("nij,ni->nj", verts[tri_idx], bary)


def parity_error(posed: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Erreur de parité L2 par point entre le nuage posé et la surface de référence.

    Paramètres
    ----------
    posed :
        Points posés via le chemin mesh-free (LBS creux), (N, 3).
    ref :
        Points de référence issus de ``surface_points`` (avance SMPL complète), (N, 3).

    Retourne
    --------
    np.ndarray
        Distance ``‖posed − ref‖₂`` par point, (N,).
    """
    return np.linalg.norm(posed - ref, axis=1)
