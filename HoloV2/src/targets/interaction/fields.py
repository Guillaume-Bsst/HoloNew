"""eval_fields — évaluer un nuage posé contre TOUS les canaux → ``MultiChannelField`` (premier canal
``(C, P)``). La matrice nuages × canaux exécute la MÊME op pour le nuage humain ET chaque nuage objet.

Pour chaque ``Channel`` : un canal avec ``object_idx is None`` est le SOL statique (frame mondial) ;
un canal avec ``object_idx=i`` mappe les points dans le frame local de l'objet i via sa ``(R, t)``
par frame. Ensuite, le ``sdf`` du canal est échantillonné par interpolation trilinéaire — UN chemin,
pas de cas spécial sol-plat (le sol plat est un SDF de plan exact) — et la direction du contact est
reconstruite à partir du témoin trilinéairement interpolé (stable aux arêtes/coins des boîtes). Pur,
orienté tableaux (axe = points), sans torch. Porté de HoloNew ``contact/contact_field`` +
``contact/combined`` + ``backends/floor``.

La diagonale de la matrice (un nuage OBJET vs son PROPRE canal) est l'une exception : là le nuage
repose sur sa propre surface, un auto-contact dégénéré, donc ``self_idx`` le court-circuite sur la
forme fermée (distance 0, témoin = le point lui-même) SANS échantillonner le SDF — gardant la
disposition ``(C, P)`` homogène tout en laissant la résolution aval ignorer cette diagonale à bon marché.
"""
from __future__ import annotations

import numpy as np

from ..contracts import MultiChannelField
from ...prepare.contracts import Channel, SDF


def eval_fields(points: np.ndarray, channels: tuple[Channel, ...], object_rot: np.ndarray,
                object_pos: np.ndarray, margin: float, self_idx: int | None = None) -> MultiChannelField:
    """Champ ``(C, P)`` de ``points`` (``(P, 3)`` mondial) vs chaque ``Channel``.

    ``object_rot (N, 3, 3)`` / ``object_pos (N, 3)`` sont les transformations mondiales d'objet par frame
    (de ``FramePose``, réutilisé — pas de recalcul) : un canal avec ``object_idx=i`` lit
    ``(object_rot[i], object_pos[i])`` comme son frame, un canal avec ``object_idx is None`` utilise
    le frame mondial (sol). Un sondage est ``actif`` dans la ``margin`` de la surface ; les sondages
    inactifs portent ``distance = +margin`` et direction/témoin mis à zéro, donc la sortie est homogène ``(C, P)``.

    ``self_idx`` est l'index d'objet du nuage en cours d'évaluation (son PROPRE objet), ou ``None`` pour un
    nuage qui n'est pas un objet (l'humain). Quand ``object_idx`` d'un canal == ``self_idx`` le nuage repose sur
    sa PROPRE surface, donc le champ est forme fermée — distance 0, témoin = le point lui-même (local-objet),
    normal nul, actif partout — et est rempli DIRECTEMENT sans échantillonner ce SDF (sautant l'aller-retour
    auto dégénéré). La diagonale de la matrice objet×canal ; le côté humain n'en a pas.
    """
    pts = np.asarray(points, np.float64)                            # (P, 3) positions monde des probes
    margin = float(margin)
    p = len(pts)

    dist_ch, dir_ch, wit_ch, act_ch = [], [], [], []
    for ch in channels:
        if ch.object_idx is None:
            probe = pts                                            # sol : frame local == monde
        else:
            rot = np.asarray(object_rot[ch.object_idx], np.float64)  # (3, 3) rotation monde de l'objet
            pos = np.asarray(object_pos[ch.object_idx], np.float64)  # (3,)   position monde de l'objet
            probe = (pts - pos) @ rot                              # (P, 3) = R.T @ (p - t), objet-local

        if self_idx is not None and ch.object_idx == self_idx:
            # Canal auto : ce nuage EST l'objet ``self_idx``, donc chaque sondage repose sur sa propre surface.
            # Remplissage en forme fermée (pas d'échantillonnage SDF) : distance 0, témoin = le sondage lui-même
            # (local-objet), pas de normale de contact, actif partout (sur surface => dans margin).
            dist_ch.append(np.zeros(p))                            # (P,)
            dir_ch.append(np.zeros((p, 3)))                       # (P, 3) pas de normale sur soi
            wit_ch.append(probe)                                  # (P, 3) point propre, objet-local
            act_ch.append(np.ones(p, bool))                       # (P,)
            continue

        dist, witness, in_grid = _sample(ch.sdf, probe)            # (P,), (P, 3), (P,) — frame local
        active = in_grid & (dist < margin)                         # (P,) dans la bande ET dans la grille

        delta = probe - witness                                    # (P, 3) surface -> point, frame canal
        norm = np.linalg.norm(delta, axis=1, keepdims=True)        # (P, 1)
        # Direction du témoin stocké (interpolé), PAS un gradient de distance : un vrai vecteur unitaire
        # même aux arêtes/coins des boîtes. Le dénominateur gardé (1.0 sur surface) évite un avertissement
        # 0/0 ; ``where`` met toujours à zéro la direction où sondage == témoin (sur surface) et absorbe
        # le résidu d'aller-retour témoin float32 là.
        den = np.where(norm > 1e-6, norm, 1.0)                     # (P, 1) diviseur sûr
        direction = np.where(norm > 1e-6, delta / den, 0.0)        # (P, 3) normale de contact unitaire

        # Homogénéiser la disposition (C, P) : les sondages inactifs portent distance=+margin, dir/témoin mis à zéro.
        dist_ch.append(np.where(active, dist, margin))             # (P,)
        dir_ch.append(np.where(active[:, None], direction, 0.0))   # (P, 3)
        wit_ch.append(np.where(active[:, None], witness, 0.0))     # (P, 3)
        act_ch.append(active)                                      # (P,)

    return MultiChannelField(
        distance=np.stack(dist_ch),                                # (C, P)
        direction=np.stack(dir_ch),                                # (C, P, 3)
        witness=np.stack(wit_ch),                                  # (C, P, 3)
        active=np.stack(act_ch),                                   # (C, P)
        channels=tuple(ch.name for ch in channels),               # (C,)
    )


def _sample(sdf: SDF, probe: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Échantillon trilinéaire d'un ``SDF`` au sondage local ``probe`` (``(P, 3)``) → ``(distance (P,),
    témoin (P, 3), in_grid (P,))``, tous dans le frame local du SDF. Le nœud ``(ix, iy, iz)`` s'assoit à
    ``origine + espacement*(ix, iy, iz)`` ; la distance ET les grilles de témoin stockées sont
    interpolées avec les MÊMES poids des 8 coins. ``in_grid`` signale les sondages dont la cellule
    englobante est entièrement à l'intérieur de la grille (coins sur chaque axe) ; les sondages hors
    grille ne sont serrés que pour que la cueillette reste dans les limites — l'appelant les traite
    comme inactifs. Calcul float64 ; vectorisé sur P (la boucle des 8 coins est fixe)."""
    shape = np.array(sdf.grid.shape)                               # (3,) nœuds par axe
    g = (probe - sdf.origin) / sdf.spacing                         # (P, 3) indice de grille continu
    i0 = np.floor(g).astype(np.int64)                             # (P, 3) coin inférieur
    in_grid = np.all((i0 >= 0) & (i0 < shape - 1), axis=1)         # (P,) les deux coins valides par axe
    t = g - i0                                                    # (P, 3) offset fractionnaire dans [0, 1)
    i0 = np.clip(i0, 0, shape - 2)                                # clamp pour que le gather 8-coins reste dans les bornes
    ix, iy, iz = i0[:, 0], i0[:, 1], i0[:, 2]

    dist = np.zeros(len(probe), np.float64)                        # (P,)
    witness = np.zeros((len(probe), 3), np.float64)               # (P, 3)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (np.where(dx, t[:, 0], 1 - t[:, 0]) *
                     np.where(dy, t[:, 1], 1 - t[:, 1]) *
                     np.where(dz, t[:, 2], 1 - t[:, 2]))           # (P,) poids trilinéaire du coin
                dist += w * sdf.grid[ix + dx, iy + dy, iz + dz]    # (P,)
                witness += w[:, None] * sdf.witness[ix + dx, iy + dy, iz + dz]  # (P, 3)
    return dist, witness, in_grid
