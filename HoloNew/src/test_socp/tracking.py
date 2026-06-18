"""GMR world-frame pos/rot tracking objective for TEST-SOCP.

The baseline (always-on) tracking term, extracted from the SQP loop so it is
unit-testable and carries the same σ convention as every other cost: each
residual is divided by its characteristic scale σ and weighted by a global
priority λ, with the per-point IK weights (w_p / w_r) kept as the intra-term
distribution. At the defaults (λ=1, σ=1) the effective weight is exactly the
legacy w_p / w_r, so behavior is unchanged.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def build_tracking_blocks(rt, frame_targets, q_mj, lambda_pos, sigma_p,
                          lambda_rot, sigma_rot, activate_pos=True, activate_rot=True):
    """ResidualBlock form of build_tracking_terms (same math, weights folded into A/c).

    Each block satisfies ‖b.c‖² == the corresponding cvxpy term value at dqa=0.
    """
    from HoloNew.src.test_socp.solve.spec import ResidualBlock

    scale_p = lambda_pos / (sigma_p * sigma_p)
    scale_r = lambda_rot / (sigma_rot * sigma_rot)
    blocks = []
    for frame, (p_t, R_t, w_p, w_r) in frame_targets.items():
        body = rt.robot_link_names[frame]
        Jp, Jr = rt._body_jac(q_mj, body)

        if activate_pos and w_p > 0:
            s = np.sqrt(scale_p * w_p)
            p_c = rt.body_position(q_mj, body)
            blocks.append(ResidualBlock(A=s * Jp, c=-s * (p_t - p_c), name=f"track_pos/{body}"))

        if activate_rot and w_r > 0:
            s = np.sqrt(scale_r * w_r)
            R_c = rt.body_rotation(q_mj, body)
            e = Rotation.from_matrix(R_c.T @ R_t).as_rotvec()
            Jr_body = R_c.T @ Jr
            blocks.append(ResidualBlock(A=s * Jr_body, c=-s * e, name=f"track_rot/{body}"))
    return blocks
