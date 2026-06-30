"""S-pos / S-rot — the STYLE residual builder (robot only, ``δv``). Linearises the per-link position
and orientation tracking error from ``StyleEval`` (current FK + Jacobians) vs ``StyleTargets`` (the
reference posture). Weights ``cfg.w_pos`` / ``cfg.w_rot`` are folded into ``A`` and ``c`` (rule:
``ResidualBlock`` carries no separate weight). ``A_obj = None`` (style does not touch the object)."""
from __future__ import annotations

import numpy as np

from ..contracts import ResidualBlock
from ..config import SolveConfig
from ._ops import quat_to_rot, so3_log
from ...targets.contracts import StyleEval, StyleTargets


def _so3_left_jac_inv(phi: np.ndarray) -> np.ndarray:                    # (3,) -> (3,3)
    """Inverse LEFT Jacobian of SO(3) at ``phi`` (an so(3) vector). It is the exact first-order map of a
    WORLD-frame (left) perturbation onto the ``so3_log`` residual: for ``E = exp([phi]_×)`` and a left
    bump ``exp([δ]_×)·E``, ``log(exp([δ]_×)·E) ≈ phi + J_l⁻¹(phi)·δ``. Omitting it (using the raw angular
    Jacobian) is only exact at ``phi = 0`` and drifts O(‖phi‖) at finite tracking error — which is why the
    naive ``A = jac_rot`` fails the S-rot FD check. ``J_l⁻¹(phi) = I − ½[phi]_× + c·[phi]_×²`` with
    ``c = 1/θ² − cot(θ/2)/(2θ)`` (numerically safe for θ ∈ (0, 2π); → I − ½[phi]_× as θ → 0)."""
    theta = float(np.linalg.norm(phi))
    K = np.array([[0.0, -phi[2], phi[1]], [phi[2], 0.0, -phi[0]], [-phi[1], phi[0], 0.0]])
    if theta < 1e-6:
        return np.eye(3) - 0.5 * K                                      # c·K² is O(θ²) here -> negligible
    c = 1.0 / theta**2 - (np.cos(theta / 2.0) / np.sin(theta / 2.0)) / (2.0 * theta)
    return np.eye(3) - 0.5 * K + c * (K @ K)


def build_style(style_eval: StyleEval, style_targets: StyleTargets,
                cfg: SolveConfig) -> list[ResidualBlock]:
    """``[S-pos]`` (+ ``[S-rot]`` if the targets carry orientation). Stacks the L links into ``3L``
    rows. ``S-pos``: ``A = w_pos·jac_pos`` (3L,nv), ``c = w_pos·(pos_cur − pos_ref)``. ``S-rot``:
    ``c = w_rot·so3_log(R_ref, R_cur)`` (R_ref via quat→R, world-frame log), ``A = w_rot·J_l⁻¹(c/w_rot)·
    jac_rot`` — the inverse-left-Jacobian factor makes the linearisation the EXACT first-order change of
    the so3_log residual (a raw ``jac_rot`` is only exact at zero error)."""
    L, nv = style_eval.position.shape[0], style_eval.jac_pos.shape[2]
    blocks: list[ResidualBlock] = []

    A_pos = (cfg.w_pos * style_eval.jac_pos).reshape(L * 3, nv)
    c_pos = (cfg.w_pos * (style_eval.position - style_targets.position)).reshape(L * 3)
    blocks.append(ResidualBlock(A=A_pos, c=c_pos, A_obj=None, name="S-pos"))

    if style_targets.orientation is not None:
        R_ref = quat_to_rot(style_targets.orientation)                  # (L,3,3) from wxyz
        r0 = so3_log(R_ref, style_eval.rotation)                        # (L,3) world-frame error per link
        A_rot = np.empty((L, 3, nv))
        for l in range(L):                                              # exact GN: J_l⁻¹(r0) · jac_rot
            A_rot[l] = _so3_left_jac_inv(r0[l]) @ style_eval.jac_rot[l]
        A_rot = (cfg.w_rot * A_rot).reshape(L * 3, nv)
        c_rot = (cfg.w_rot * r0).reshape(L * 3)
        blocks.append(ResidualBlock(A=A_rot, c=c_rot, A_obj=None, name="S-rot"))

    return blocks
