"""Helpers purs (numpy-only, sans viser) pour les couches de debug contact/objet.

Trois fonctions :
- ``object_cloud_solved`` : repose un nuage de points objet rigide de la pose source
  vers la pose résolue via la composition ``T_résolu ∘ T_source⁻¹``.
- ``witness_segments`` : construit les segments (sonde → point witness mappé monde)
  pour les sondes actives, avec sous-échantillonnage déterministe.
- ``normal_segments`` : construit les segments (sonde → sonde + normale mappée monde)
  pour les sondes actives, avec le même sous-échantillonnage déterministe. La normale
  est un VECTEUR direction : mappage local→monde par rotation seule (pas de translation).
"""
from __future__ import annotations

import numpy as np

# Plafond du nombre de segments witness rendus (cohérent avec fields.py)
_MAX_SEG = 400


def object_cloud_solved(
    cloud_src: np.ndarray,
    R_src: np.ndarray,
    t_src: np.ndarray,
    R_solved: np.ndarray,
    t_solved: np.ndarray,
) -> np.ndarray:
    """Repose le nuage source d'un objet rigide vers sa pose résolue.

    Applique la composition ``T_résolu ∘ T_source⁻¹`` à chaque point du nuage
    source, entièrement vectorisée (pas de boucle Python par point).

    Pour un point ``p`` du cloud source :
        ``p_résolu = R_résolu @ (R_source.T @ (p - t_source)) + t_résolu``

    Paramètres
    ----------
    cloud_src :
        (P, 3) nuage de points dans le repère monde à la pose source.
    R_src :
        (3, 3) rotation de la pose source (matrice orthogonale).
    t_src :
        (3,) translation de la pose source.
    R_solved :
        (3, 3) rotation de la pose résolue.
    t_solved :
        (3,) translation de la pose résolue.

    Retourne
    --------
    np.ndarray
        (P, 3) float64 — positions du nuage dans le repère monde à la pose résolue.
    """
    # Dé-pose source : ramène les points dans le référentiel local de l'objet
    p_local = (np.asarray(cloud_src, np.float64) - t_src) @ R_src   # (P, 3)  — R_src.T appliqué par ligne
    # Re-pose résolue : projette depuis le local vers le monde résolu
    return p_local @ R_solved.T + t_solved                           # (P, 3)


def witness_segments(
    probe_pts: np.ndarray,
    witness_local: np.ndarray,
    active: np.ndarray,
    R_obj: np.ndarray,
    t_obj: np.ndarray,
    *,
    cap: int = _MAX_SEG,
) -> np.ndarray:
    """Construit les segments (sonde monde → witness monde) pour les sondes actives.

    Mappe le witness du référentiel objet-local (ou sol = déjà monde) vers le repère
    monde via ``witness_monde = witness_local @ R_obj.T + t_obj``.  Pour un canal sol,
    l'appelant passe ``R_obj=np.eye(3), t_obj=np.zeros(3)`` → identité.

    Si le nombre de sondes actives dépasse ``cap``, un sous-échantillonnage
    déterministe est effectué (RNG de graine 0, cohérent avec ``fields.py``).

    Paramètres
    ----------
    probe_pts :
        (P, 3) positions monde des sondes (toutes, actives ou non).
    witness_local :
        (P, 3) positions des points witness dans le référentiel objet-local.
    active :
        (P,) booléen — masque des sondes actives.
    R_obj :
        (3, 3) rotation de la pose objet (monde← local).
    t_obj :
        (3,) translation de la pose objet.
    cap :
        Plafond du nombre de segments retournés (défaut : 400).

    Retourne
    --------
    np.ndarray
        (S, 2, 3) float32 — ``seg[s] = [probe_monde, witness_monde]``.
        Forme (0, 2, 3) si aucune sonde n'est active.
    """
    # Indices des sondes actives
    idx = np.where(np.asarray(active, bool))[0]

    # Cas vide : aucune sonde active
    if len(idx) == 0:
        return np.zeros((0, 2, 3), np.float32)

    # Sous-échantillonnage déterministe si le plafond est atteint
    if len(idx) > cap:
        idx = np.random.default_rng(0).choice(idx, cap, replace=False)

    pts = np.asarray(probe_pts, np.float64)[idx]            # (S, 3) monde
    wit_loc = np.asarray(witness_local, np.float64)[idx]    # (S, 3) local objet

    # Projection local→monde : wit_loc @ R_obj.T + t_obj
    # (pour le canal sol, R_obj=I et t_obj=0 → passage transparent)
    wit_world = wit_loc @ np.asarray(R_obj, np.float64).T + np.asarray(t_obj, np.float64)  # (S, 3)

    # Assemblage des segments [sonde, witness] et cast float32 pour le rendu
    return np.stack([pts, wit_world], axis=1).astype(np.float32)   # (S, 2, 3)


def normal_segments(
    probe_pts: np.ndarray,
    direction_local: np.ndarray,
    active: np.ndarray,
    R_obj: np.ndarray,
    *,
    length: float = 0.05,
    cap: int = _MAX_SEG,
) -> np.ndarray:
    """Construit les segments (sonde monde → sonde + normale monde) pour les sondes actives.

    La normale de contact (« surface → point », dans le repère du canal) est un VECTEUR
    direction : son passage local→monde se fait par ROTATION SEULE (pas de translation),
    ``dir_monde = direction_local @ R_obj.T``.  Pour un canal sol, l'appelant passe
    ``R_obj=np.eye(3)`` → direction déjà exprimée en monde (passage transparent).

    Le segment part de la position monde de la sonde et pointe dans la direction de la
    normale sur une longueur fixe ``length`` (purement diagnostique, pas de mise à
    l'échelle par la distance de contact).

    Si le nombre de sondes actives dépasse ``cap``, un sous-échantillonnage
    déterministe est effectué (RNG de graine 0, même logique que ``witness_segments``).

    Paramètres
    ----------
    probe_pts :
        (P, 3) positions monde des sondes (toutes, actives ou non).
    direction_local :
        (P, 3) normales de contact dans le référentiel du canal (« surface → point »).
    active :
        (P,) booléen — masque des sondes actives.
    R_obj :
        (3, 3) rotation de la pose objet (monde← local). Identité pour le canal sol.
    length :
        Longueur monde du segment de normale dessiné (défaut : 0.05 m).
    cap :
        Plafond du nombre de segments retournés (défaut : 400).

    Retourne
    --------
    np.ndarray
        (S, 2, 3) float32 — ``seg[s] = [sonde_monde, sonde_monde + dir_monde * length]``.
        Forme (0, 2, 3) si aucune sonde n'est active.
    """
    # Indices des sondes actives
    idx = np.where(np.asarray(active, bool))[0]

    # Cas vide : aucune sonde active
    if len(idx) == 0:
        return np.zeros((0, 2, 3), np.float32)

    # Sous-échantillonnage déterministe si le plafond est atteint
    if len(idx) > cap:
        idx = np.random.default_rng(0).choice(idx, cap, replace=False)

    pts = np.asarray(probe_pts, np.float64)[idx]              # (S, 3) monde
    dir_loc = np.asarray(direction_local, np.float64)[idx]    # (S, 3) local canal

    # Passage local→monde par rotation seule : dir_loc @ R_obj.T
    # (pour le canal sol, R_obj=I → direction déjà en monde)
    dir_world = dir_loc @ np.asarray(R_obj, np.float64).T     # (S, 3)

    # Extrémité du segment = sonde + normale mise à l'échelle par la longueur fixe
    tip = pts + dir_world * float(length)                     # (S, 3)

    # Assemblage des segments [sonde, extrémité] et cast float32 pour le rendu
    return np.stack([pts, tip], axis=1).astype(np.float32)   # (S, 2, 3)
