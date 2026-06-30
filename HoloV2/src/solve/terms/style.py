"""S-pos / S-rot — le builder résiduel STYLE (robot uniquement, ``δv``). Linéarise l'erreur de suivi
de position et d'orientation par-lien de ``StyleEval`` (FK courant + Jacobiennes) vs ``StyleTargets`` (la
posture de référence). Les poids ``cfg.w_pos`` / ``cfg.w_rot`` sont repliés dans ``A`` et ``c`` (règle :
``ResidualBlock`` ne porte pas de poids séparé). ``A_obj = None`` (le style ne touche pas l'objet)."""
from __future__ import annotations

import numpy as np

from ..contracts import ResidualBlock
from ..config import SolveConfig
from ._ops import quat_to_rot, so3_log
from ...targets.contracts import StyleEval, StyleTargets


def _so3_left_jac_inv(phi: np.ndarray) -> np.ndarray:                    # (3,) -> (3,3)
    """Jacobienne inverse GAUCHE de SO(3) en ``phi`` (un vecteur so(3)). C'est la carte du premier ordre EXACTE
    d'une perturbation (gauche) du frame MONDE sur le résiduel ``so3_log`` : pour ``E = exp([phi]_×)`` et un
    bump gauche ``exp([δ]_×)·E``, ``log(exp([δ]_×)·E) ≈ phi + J_l⁻¹(phi)·δ``. L'omettre (utiliser la Jacobienne
    angulaire brute) n'est exact qu'en ``phi = 0`` et dériv O(‖phi‖) à erreur de suivi finie — c'est pourquoi le
    ``A = jac_rot`` naïf échoue le test FD S-rot. ``J_l⁻¹(phi) = I − ½[phi]_× + c·[phi]_×²`` avec
    ``c = 1/θ² − cot(θ/2)/(2θ)`` (numériquement sûr pour θ ∈ (0, 2π); → I − ½[phi]_× quand θ → 0)."""
    theta = float(np.linalg.norm(phi))
    K = np.array([[0.0, -phi[2], phi[1]], [phi[2], 0.0, -phi[0]], [-phi[1], phi[0], 0.0]])
    if theta < 1e-6:
        return np.eye(3) - 0.5 * K                                      # c·K² est O(θ²) ici -> négligeable
    c = 1.0 / theta**2 - (np.cos(theta / 2.0) / np.sin(theta / 2.0)) / (2.0 * theta)
    return np.eye(3) - 0.5 * K + c * (K @ K)


def build_style(style_eval: StyleEval, style_targets: StyleTargets,
                cfg: SolveConfig) -> list[ResidualBlock]:
    """``[S-pos]`` (+ ``[S-rot]`` si les cibles portent l'orientation). Empile les L liens en ``3L``
    lignes. ``S-pos`` : ``A = w_pos·jac_pos`` (3L,nv), ``c = w_pos·(pos_cur − pos_ref)``. ``S-rot`` :
    ``c = w_rot·so3_log(R_ref, R_cur)`` (R_ref via quat→R, log de frame monde), ``A = w_rot·J_l⁻¹(c/w_rot)·
    jac_rot`` — le facteur Jacobienne-inverse-gauche rend la linéarisation le changement du premier ordre EXACT
    du résiduel so3_log (un ``jac_rot`` brut n'est exact qu'à erreur zéro)."""
    L, nv = style_eval.position.shape[0], style_eval.jac_pos.shape[2]
    blocks: list[ResidualBlock] = []

    A_pos = (cfg.w_pos * style_eval.jac_pos).reshape(L * 3, nv)
    c_pos = (cfg.w_pos * (style_eval.position - style_targets.position)).reshape(L * 3)
    blocks.append(ResidualBlock(A=A_pos, c=c_pos, A_obj=None, name="S-pos"))

    if style_targets.orientation is not None:
        R_ref = quat_to_rot(style_targets.orientation)                  # (L,3,3) depuis wxyz
        r0 = so3_log(R_ref, style_eval.rotation)                        # (L,3) erreur frame monde par lien
        A_rot = np.empty((L, 3, nv))
        for l in range(L):                                              # GN exact : J_l⁻¹(r0) · jac_rot
            A_rot[l] = _so3_left_jac_inv(r0[l]) @ style_eval.jac_rot[l]
        A_rot = (cfg.w_rot * A_rot).reshape(L * 3, nv)
        c_rot = (cfg.w_rot * r0).reshape(L * 3)
        blocks.append(ResidualBlock(A=A_rot, c=c_rot, A_obj=None, name="S-rot"))

    return blocks
