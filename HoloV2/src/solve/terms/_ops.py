"""Ops complexes AU SERVICE EXCLUSIF DES RÉSIDUELS — les contractions spécifiques à ``solve`` / cartes de
frame / logs de variété que la spec garde EN DEHORS de l'évaluateur (sans-ref) ``targets``. Numpy pur
(float64), pas d'I/O, pas de mutation. Partagé par C et CO (règle #8 homogénéité) : une contraction ``dist_jac``,
une carte de frame ``world_normal``, un ``so3_log``.

Conventions (verrouillées par ``targets``) :
  * Les Jacobiennes de point robot (``point_jac``, ``jac_pos``, ``jac_rot``) sont WORLD / LOCAL_WORLD_ALIGNED.
  * Les gradients de ``direction``/``witness``/géodésique de canal OBJECT sont OBJECT-LOCAL -> mappé au monde avec
    ``world_normal(R_i, …)`` avant contraction avec une Jacobienne monde ; contracter avec le vecteur LOCAL BRUT
    contre la Jacobienne tangente d'objet ``probe_jac_obj`` (local à l'objet).
  * Le résiduel d'orientation est le log de frame MONDE ``log(R_cur·R_refᵀ)`` pour s'apparier avec la Jacobienne
    angulaire mondiale ``jac_rot`` (``omega_world = jac_rot·v``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...prepare.contracts import GeodesicTable


def world_normal(R: np.ndarray, n_local: np.ndarray) -> np.ndarray:
    """Mappe une direction/normale/gradient LOCAL-d'objet au MONDE : ``n_world = R · n_local``.
    ``R`` (3,3) [une trame] ou (M,3,3) [par ligne] ; ``n_local`` (...,3). Canal sol : ``R = I``."""
    R = np.asarray(R, np.float64)
    n = np.asarray(n_local, np.float64)
    return np.einsum("...ij,...j->...i", R, n)


def dist_jac(direction: np.ndarray, jac: np.ndarray) -> np.ndarray:
    """``∂(directionᵀ·point)/∂step`` = ``directionᵀ·jac`` ligne-par-ligne. ``direction`` (M,3), ``jac``
    (M,3,K) -> (M,K). Le gradient de distance signée par rapport au point est la normale de contact unitaire,
    donc ceci donne ``∂d/∂step`` pour la tangente robot (K=nv, ``point_jac``) et la tangente d'objet
    (K=6, ``probe_jac_obj`` / ``cloud_jac_self``)."""
    direction = np.asarray(direction, np.float64)
    jac = np.asarray(jac, np.float64)
    return np.einsum("mi,mij->mj", direction, jac)


def geo_chain(grad: np.ndarray, jac: np.ndarray) -> np.ndarray:
    """``∂geo/∂step`` = ``gradᵀ·jac`` — la MÊME contraction que ``dist_jac`` (le gradient géodésique est
    tangent à la surface ; sa composante normale, le cas échéant, est annihilée par la Jacobienne tangente).
    Conservé comme op nommée pour la lisibilité du builder (règle #8)."""
    return dist_jac(grad, jac)


def quat_to_rot(wxyz: np.ndarray) -> np.ndarray:
    """Quaternion(s) unitaire(s) ``wxyz`` (...,4) -> matrice de rotation (...,3,3). Normalise défensivement."""
    q = np.asarray(wxyz, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), np.float64)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z); R[..., 0, 1] = 2 * (x * y - z * w); R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w); R[..., 1, 1] = 1 - 2 * (x * x + z * z); R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w); R[..., 2, 1] = 2 * (y * z + x * w); R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _log_one(E: np.ndarray) -> np.ndarray:
    """Log SO(3) d'une seule matrice de rotation -> vecteur de rotation (3,). Robuste près de 0 et π."""
    cos = np.clip((np.trace(E) - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos)
    if theta < 1e-7:                                    # près de l'identité : premier ordre
        return 0.5 * np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
    if np.pi - theta < 1e-4:                            # près de π : axe de la partie symétrique
        Aerr = (E + np.eye(3)) * 0.5
        axis = np.sqrt(np.clip(np.diag(Aerr), 0.0, None))
        # corrige les signes via la partie hors-diagonale de (E - Eᵀ)
        s = np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
        axis = np.where(s < 0, -axis, axis)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return theta * axis
    w = theta / (2.0 * np.sin(theta))
    return w * np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])


def so3_log(R_ref: np.ndarray, R_cur: np.ndarray) -> np.ndarray:
    """Erreur d'orientation de frame monde par ligne : ``log(R_cur·R_refᵀ)`` (L,3,3),(L,3,3) -> (L,3).
    Résiduel Gauss-Newton ``c = so3_log(R_ref, R_cur)`` ; la Jacobienne du premier ordre EXACTE d'un
    bump (gauche) du monde est ``A = J_l⁻¹(c)·jac_rot``, NON le ``jac_rot`` brut — le facteur Jacobienne-inverse-gauche
    importe à erreur finie (voir ``style._so3_left_jac_inv`` ; l'omettre échoue le test FD S-rot)."""
    R_ref = np.asarray(R_ref, np.float64); R_cur = np.asarray(R_cur, np.float64)
    E = np.einsum("lij,lkj->lik", R_cur, R_ref)         # R_cur · R_refᵀ
    return np.stack([_log_one(E[l]) for l in range(E.shape[0])])


def se3_log_world(R_ref: np.ndarray, p_ref: np.ndarray,
                  R_cur: np.ndarray, p_cur: np.ndarray) -> np.ndarray:
    """Erreur SE(3) alignée au monde par objet : ``[p_cur − p_ref, log(R_cur·R_refᵀ)]`` (N,6). Correspond à la
    tangente d'objet alignée au monde ``δξ = (δt, δθ)`` (le terme O ancre l'objet à sa pose observée)."""
    p_ref = np.asarray(p_ref, np.float64); p_cur = np.asarray(p_cur, np.float64)
    out = np.empty((p_ref.shape[0], 6), np.float64)
    out[:, :3] = p_cur - p_ref
    out[:, 3:] = so3_log(R_ref, R_cur)
    return out


def scatter_obj(block: np.ndarray, object_idx: int, n_obj: int) -> np.ndarray:
    """Place un bloc Jacobienne ``(m,6)`` par-objet dans la matrice complète ``(m, n_obj*6)`` de
    couplage d'objet (sparse : zéros pour les autres objets). ``object_idx`` dans ``[0, n_obj)``."""
    block = np.asarray(block, np.float64)
    m = block.shape[0]
    A_obj = np.zeros((m, n_obj * 6), np.float64)
    A_obj[:, object_idx * 6:(object_idx + 1) * 6] = block
    return A_obj


@dataclass(frozen=True)
class GeoField:
    """Tables géodésiques par-canal + frames MONDE des canaux — le paquet que ``build_contact`` lit comme
    son argument ``geo``. Assemblé par Plan C à partir de ``InteractionContext`` (les tables géodésiques) +
    ``FrameTargets.object_rot/pos``. Laisse ``build_contact`` (a) lire le champ géodésique par canal et
    (b) mapper les directions/gradients de champ LOCAL-d'objet au monde via ``world_normal(rot[c], …)`` —
    uniformément sur les canaux (frame sol = identité). Voir Assumption 3 du plan."""

    tables: tuple[GeodesicTable | None, ...]  # (C,) table géodésique par canal ; None -> pas de ligne C-X
    rot: np.ndarray                           # (C, 3, 3) rotation monde par canal (sol = I)
    pos: np.ndarray                           # (C, 3)    translation monde par canal (sol = 0)
    object_idx: tuple[int, ...]               # (C,) canal -> index d'objet (-1 pour le sol)
