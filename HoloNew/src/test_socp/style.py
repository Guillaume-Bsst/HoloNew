"""Pelvis-relative Style objective pieces for TEST-SOCP.
See docs/specs/2026-06-13-brick3-pelvis-relative-style-design.md."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

_ZHAT = np.array([0.0, 0.0, 1.0])
_ZSKEW = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # [zhat]_x


def pelvis_tilt_residual(rt, q_mj, R_B_ref):
    """Roll/pitch tilt residual r0 - A@dqa for ||(R_ref)ᵀẑ − R_Bᵀẑ||²."""
    R_B = rt.body_rotation(q_mj, "pelvis")
    _, Jr_B = rt._body_jac(q_mj, "pelvis")
    u = R_B.T @ _ZHAT
    r0 = R_B_ref.T @ _ZHAT - u
    A = R_B.T @ _ZSKEW @ Jr_B
    return r0, A


def build_style_blocks(rt, q_mj, frame_targets, lambda_ws, sigma_R=1.0,
                       style_weights=None):
    """ResidualBlock form of build_style_terms (same math, weights folded into A/c).

    Same early-return logic as build_style_terms; no dqa argument.
    Each block satisfies ‖b.c‖² == the corresponding cvxpy term value at dqa=0.
    """
    from HoloNew.src.test_socp.solve.spec import ResidualBlock

    if lambda_ws <= 0 or not getattr(rt, "activate_rot_tracking", True):
        return []
    pelvis_body = "pelvis"
    R_B0 = rt.body_rotation(q_mj, pelvis_body)
    R_Bref = next((R_t for frame, (p_t, R_t, w_p, w_r) in frame_targets.items()
                   if rt.robot_link_names[frame] == pelvis_body), None)
    if R_Bref is None:
        return []

    def _raw(body, w_r):
        if style_weights is None:
            return float(w_r)
        key = "__pelvis_tilt__" if body == pelvis_body else body
        return float(style_weights.get(key, 0.0))

    raw_by_frame = {f: _raw(rt.robot_link_names[f], w_r)
                    for f, (p_t, R_t, w_p, w_r) in frame_targets.items()}
    w_tot = sum(v for v in raw_by_frame.values() if v > 0)
    if w_tot <= 0:
        return []

    inv_sig2 = 1.0 / (sigma_R * sigma_R)
    blocks = []
    for frame, (p_t, R_t, w_p, w_r) in frame_targets.items():
        body = rt.robot_link_names[frame]
        raw = raw_by_frame[frame]
        if raw <= 0:
            continue
        omega = lambda_ws * (raw / w_tot) * inv_sig2
        s = np.sqrt(omega)
        if body == pelvis_body:
            r0, A = pelvis_tilt_residual(rt, q_mj, R_Bref)            # S_B
            blocks.append(ResidualBlock(A=s * A, c=-s * r0, name="style_tilt"))
        else:
            _, Jr = rt._body_jac(q_mj, body)                         # S_k
            R_c = rt.body_rotation(q_mj, body)
            R_target = R_B0 @ R_Bref.T @ R_t
            e = Rotation.from_matrix(R_c.T @ R_target).as_rotvec()
            Jr_body = R_c.T @ Jr
            blocks.append(ResidualBlock(A=s * Jr_body, c=-s * e, name=f"style/{body}"))
    return blocks
