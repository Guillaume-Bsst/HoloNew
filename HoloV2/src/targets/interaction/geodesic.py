"""Query online du champ géodésique précalculé (``prepare.GeodesicTable``), q-DÉPENDANTE car lue à un
``witness(q)`` continu. NUMPY-ONLY / torch-free (``targets`` reste léger) : pas de scipy — le k-NN est
un brute-force vectorisé ``(Q,P)`` ; les k voisins sont sélectionnés par ``np.argpartition``.

``nearest_index`` snappe un point fixe (``witness_ref``) sur sa source la plus proche (OFFLINE, pas de
gradient). ``geo_value_grad`` lit le champ mono-source ``geo[source]`` à un ``query_xyz`` continu par
MLS degré-1 (moindres carrés locaux pondérés, fit ``f≈c+b·(p-q)``) → valeur ``c`` ET gradient ``∇c(q)``,
naturellement tangent à la surface. Le gradient retourné est le gradient analytique de ``c(q)`` à bande
passante FIXE (théorème des fonctions implicites sur les équations normales), et non le seul ``b``
(gradient du polynôme local). Pour un champ linéaire les deux coïncident (résidus nuls → correction
nulle ; gradient EXACT à O(ridge) près). Pour un champ NON-linéaire, la correction porte les résidus
mais OMET le terme de bande passante adaptative ``∂h/∂q`` (lui aussi nul pour un champ linéaire) → le
gradient est APPROCHÉ (écart résiduel ~1e-2, d'où la tolérance ~2e-2 du test de différences finies),
borné et bonne direction de descente pour le solveur — pas la précision machine. Vectorisé sur Q."""
from __future__ import annotations

import numpy as np

from ...prepare.contracts import GeodesicTable


def nearest_index(points: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Indice du point de surface le plus proche de ``xyz`` (snap ``witness_ref`` → source, offline).
    ``xyz`` (3,) ou (Q,3) → (Q,) int. Brute-force numpy."""
    pts = np.asarray(points, np.float64)
    q = np.atleast_2d(np.asarray(xyz, np.float64))
    d2 = ((q[:, None, :] - pts[None, :, :]) ** 2).sum(-1)            # (Q, P)
    return np.argmin(d2, axis=1)


def geo_value_grad(table: GeodesicTable, source_idx: np.ndarray, query_xyz: np.ndarray,
                   k: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Champ géodésique depuis ``source_idx`` lu à ``query_xyz`` par MLS degré-1.
    Renvoie ``(g:(Q,), grad:(Q,3))``. Vectorisé sur Q ; reproduit un champ localement linéaire.

    Le gradient retourné est ``∇c(q)`` à bande passante FIXE (gradient analytique de la valeur MLS,
    via les fonctions implicites). Champ linéaire : EXACT à O(ridge)~1e-9 (résidus nuls → correction
    nulle). Champ non-linéaire : le terme de bande passante adaptative ``∂h/∂q`` est OMIS (non nul) →
    gradient APPROCHÉ ; l'écart aux différences finies est ~1e-2, absorbé par la tolérance du test."""
    pts = np.asarray(table.points, np.float64)                       # (P,3)
    q = np.atleast_2d(np.asarray(query_xyz, np.float64))             # (Q,3)
    src = np.atleast_1d(np.asarray(source_idx, np.int64))            # (Q,)
    P, Q = pts.shape[0], q.shape[0]
    k = min(k, P)

    d2 = ((q[:, None, :] - pts[None, :, :]) ** 2).sum(-1)            # (Q,P)
    nn = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]               # (Q,k) k plus proches
    d2_nn = np.take_along_axis(d2, nn, axis=1)                       # (Q,k)

    dist = np.sqrt(d2_nn)                                            # (Q,k)
    h = dist.mean(axis=1, keepdims=True) + 1e-12                     # (Q,1) bande passante adaptative
    h2 = h ** 2                                                      # (Q,1)
    w = np.exp(-d2_nn / h2)                                          # (Q,k) poids gaussiens

    fields = table.geo[src]                                          # (Q,P) champ mono-source
    f = np.take_along_axis(fields, nn, axis=1).astype(np.float64)    # (Q,k) gather puis cast
    p = pts[nn]                                                      # (Q,k,3)
    dq = p - q[:, None, :]                                           # (Q,k,3) = p - q (centré en q)
    X = np.concatenate([np.ones((Q, k, 1)), dq], axis=2)            # (Q,k,4) design [1, p-q]
    Xw = X * w[:, :, None]                                           # (Q,k,4)
    A = np.einsum("qki,qkj->qij", Xw, X) + 1e-9 * np.eye(4)         # (Q,4,4) normal eq. + ridge

    # Résolution simultanée : [c,b] et α = A⁻¹ e₀ (1ère col. de A⁻¹, utilisée pour ∇c)
    e0 = np.zeros((Q, 4)); e0[:, 0] = 1.0                          # (Q,4) vecteur e₀ (robuste si k<4)
    rhs = np.einsum("qki,qk->qi", Xw, f)                             # (Q,4)
    rhs_aug = np.stack([rhs, e0], axis=-1)                            # (Q,4,2)
    sol_aug = np.linalg.solve(A, rhs_aug)                             # (Q,4,2)
    sol = sol_aug[:, :, 0]                                            # (Q,4) = [c, bx, by, bz]
    alpha = sol_aug[:, :, 1]                                          # (Q,4) = A⁻¹ e₀

    c = sol[:, 0]                                                     # (Q,)
    b = sol[:, 1:]                                                    # (Q,3)

    # Résidus aux voisins (nuls pour un champ linéaire → correction nulle)
    r = f - (X * sol[:, None, :]).sum(-1)                             # (Q,k)

    # Gradient analytique ∇c(q) = b + correction (théorème des fonctions implicites)
    # correction_j = (2/h²) Σ_k w_k (p_kj−q_j) (αᵀX_k) r_k  −  α_{j+1} Σ_k w_k r_k
    alpha_X = (alpha[:, None, :] * X).sum(-1)                        # (Q,k) = αᵀX_i
    wr = w * r                                                        # (Q,k)
    term1 = (2.0 / h2) * np.einsum("qk,qkj->qj", wr * alpha_X, dq) # (Q,3)
    term2 = alpha[:, 1:] * wr.sum(1, keepdims=True)                  # (Q,3)
    grad = b + term1 - term2                                          # (Q,3) = ∇c(q)

    return c, grad
