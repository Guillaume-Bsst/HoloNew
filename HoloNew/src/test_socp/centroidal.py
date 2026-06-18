"""Centroidal W^c (CoM acceleration) + W^c_pos (CoM position) + W^L (angular momentum).

See docs/specs/2026-06-13-brick4-centroidal-design.md.
"""
from __future__ import annotations

import numpy as np

from HoloNew.src.test_socp.solve.spec import ResidualBlock


def mapped_frame_masses_and_names(rt):
    """Robot link names + masses for the 14 mapped bodies (MAPPED_BODY_NAMES order).

    Each mapped human body tracks a robot frame (via the IK table); the toe frames
    are remapped to the ankle-roll links that actually exist in the pinocchio model.
    Returns (frame_names: list[str], masses: np.ndarray (K,)). Used by the lumped
    angular-momentum tracking (W^L) — the same masses are used for the reference and
    the current momentum, so the two are consistent despite the lumping.
    """
    from .tables import IK_MATCH_TABLE1, MAPPED_BODY_NAMES
    _remap = {"left_toe_link": "left_ankle_roll_link",
              "right_toe_link": "right_ankle_roll_link"}
    inv = {entry[0]: frame for frame, entry in IK_MATCH_TABLE1.items()}
    m = rt.pin.model
    frames, masses = [], []
    for body in MAPPED_BODY_NAMES:
        fn = _remap.get(inv[body], inv[body])
        frames.append(fn)
        fid = m.getFrameId(fn)
        masses.append(float(m.inertias[m.frames[fid].parentJoint].mass))
    return frames, np.asarray(masses, dtype=np.float64)


def reference_orbital_angular_momentum(ref_pos, masses, dt):
    """Lumped reference orbital angular momentum L_ref(t) (T, 3).

    Built from the reference mapped-body world trajectory (the GMR targets), the
    same finite-difference way cddot_ref is built from the reference CoM:
        c = sum_k m_k p_k / M,  v_k, c_dot via causal differences,
        L = sum_k m_k (p_k - c) x (v_k - c_dot).
    Captures the body's net angular momentum (e.g. an aerial cartwheel/somersault),
    which W^L should track instead of driving to zero. The lumped mass set (the 14
    mapped links) under-represents the total mass, but the SAME masses build the
    current momentum in build_lumped_L_term, so the tracked residual is consistent;
    the absolute scale is absorbed by lambda_l_track.

    Args:
        ref_pos: (T, K, 3) reference world positions of the mapped bodies.
        masses: (K,) robot link masses.
        dt: timestep.

    Returns:
        (T, 3) reference angular momentum, zeros for frames 0/1 (no velocity yet).
    """
    ref_pos = np.asarray(ref_pos, dtype=np.float64)
    T = ref_pos.shape[0]
    M = masses.sum()
    c = (masses[None, :, None] * ref_pos).sum(axis=1) / M          # (T,3)
    v = np.zeros_like(ref_pos); v[1:] = (ref_pos[1:] - ref_pos[:-1]) / dt
    cdot = np.zeros_like(c); cdot[1:] = (c[1:] - c[:-1]) / dt
    L = np.zeros((T, 3))
    for k in range(masses.shape[0]):
        L += masses[k] * np.cross(ref_pos[:, k] - c, v[:, k] - cdot)
    return L


def build_centroidal_blocks(rt, q_t0, q_tm1, c_tm1, c_tm2, cddot_ref, c_ref,
                             lambda_c, lambda_c_pos, lambda_l, dt, *,
                             sigma_a=1.0, sigma_L=1.0,
                             lambda_cv=0.0, sigma_cv=1.0, cdot_ref=None):
    """ResidualBlock version of build_centroidal_terms (no dqa argument).

    Returns list[ResidualBlock] with the same gating/early-returns as the
    original.  Each block satisfies cost = ‖A·dqa + c‖², with sqrt(lambda)
    and 1/sigma already folded in, so evaluating at dqa=0 gives ‖c‖² ==
    the corresponding cvxpy term evaluated at dqa.value=0.

    Block names: "W_c", "W_c_pos", "W_L", "W_c_vel".
    """
    blocks = []

    # Shared quantities: c0 and Jc reused by W^c and W^c_pos.
    c0 = rt.pin.com(q_t0)                                      # (3,)
    Jc = rt.pin.com_jacobian(q_t0)[:, rt.v_a_indices]          # (3, nv_a)

    # --- W^c: CoM acceleration tracking ---
    if lambda_c > 0:
        s_c = np.sqrt(lambda_c) / (sigma_a * dt**2)
        A_c = s_c * Jc                                          # (3, nv_a)
        b_c = (s_c * (c0 - 2.0*np.asarray(c_tm1) + np.asarray(c_tm2))
               - (np.sqrt(lambda_c) / sigma_a) * np.asarray(cddot_ref))  # (3,)
        blocks.append(ResidualBlock(A=A_c, c=b_c, name="W_c"))

    # --- W^c_pos: CoM absolute position anchor ---
    if lambda_c_pos > 0:
        s_p = np.sqrt(lambda_c_pos)
        A_p = s_p * Jc                                          # (3, nv_a)
        b_p = s_p * (c0 - np.asarray(c_ref))                   # (3,)
        blocks.append(ResidualBlock(A=A_p, c=b_p, name="W_c_pos"))

    # --- W^L and W^c_vel both need the centroidal map ---
    if lambda_l > 0 or (lambda_cv > 0 and cdot_ref is not None):
        Ag = rt.pin.centroidal_map(q_t0)                       # (6, nv)
        v0, Jd = rt.pin.difference_and_jac(q_tm1, q_t0)       # v0: (nv,), Jd: (nv, nv)
        Jd_a = Jd[:, rt.v_a_indices]                           # (nv, nv_a)

        if lambda_l > 0:
            AgL = Ag[3:6, :]                                   # (3, nv) angular-momentum rows
            A_L = (np.sqrt(lambda_l) / sigma_L) * (AgL @ Jd_a)  # (3, nv_a)
            b_L = (np.sqrt(lambda_l) / sigma_L) * (AgL @ v0)     # (3,)
            blocks.append(ResidualBlock(A=A_L, c=b_L, name="W_L"))

        # --- W^c_vel: CoM velocity tracking ---
        if lambda_cv > 0 and cdot_ref is not None:
            M = sum(float(I.mass) for I in rt.pin.model.inertias)   # total robot mass
            AgP = Ag[0:3, :]                                   # (3, nv) linear-momentum rows
            s = np.sqrt(lambda_cv) / (sigma_cv * M)
            A_cv = s * (AgP @ Jd_a)                            # (3, nv_a)
            b_cv = s * (AgP @ v0) - (np.sqrt(lambda_cv) / sigma_cv) * np.asarray(cdot_ref)
            blocks.append(ResidualBlock(A=A_cv, c=b_cv, name="W_c_vel"))

    return blocks


def build_lumped_L_block(rt, q_pin, q_pin_prev, frames, masses, L_ref_t,
                         lambda_l, dt, *, sigma_L=1.0):
    """ResidualBlock version of build_lumped_L_term (no dqa argument).

    The original builds r = (sqrt(lambda_l)/sigma_L) * (A_L @ dqa + (b_L0 - L_ref_t))
    where A_L and b_L0 are the dqa-coefficient matrix and constant vector of the
    linearised lumped angular momentum (see build_lumped_L_term for derivation).

    Here we absorb the scale factor directly into A and c:
        A_block = (sqrt(lambda_l)/sigma_L) * A_L   (3, nv_a)
        c_block = (sqrt(lambda_l)/sigma_L) * (b_L0 - L_ref_t)

    Returns list[ResidualBlock] with one block named "W_L_lumped", matching the
    same gating / early-return logic as the original (lambda_l == 0 is callers'
    responsibility; function always returns a list with one block).
    """
    from HoloNew.src.test_socp.interaction import _skew

    K = len(frames)
    M = masses.sum()
    p0 = np.array([rt.pin.body_position(q_pin, f) for f in frames])          # (K,3)
    pprev = np.array([rt.pin.body_position(q_pin_prev, f) for f in frames])  # (K,3)
    Js = [rt.pin.frame_translational_jacobian(q_pin, f)[:, rt.v_a_indices] for f in frames]
    c0 = (masses[:, None] * p0).sum(0) / M
    cprev = (masses[:, None] * pprev).sum(0) / M
    Jc = sum(masses[k] * Js[k] for k in range(K)) / M                        # (3, nv_a)

    A_L = np.zeros((3, rt.nv_a))
    b_L0 = np.zeros(3)
    for k in range(K):
        arm = p0[k] - c0
        v_rel0 = ((p0[k] - pprev[k]) - (c0 - cprev)) / dt
        Jrel = (Js[k] - Jc) / dt
        A_L += masses[k] * (_skew(arm) @ Jrel)
        b_L0 += masses[k] * np.cross(arm, v_rel0)

    scale = np.sqrt(lambda_l) / sigma_L
    A_block = scale * A_L                                                      # (3, nv_a)
    c_block = scale * (b_L0 - np.asarray(L_ref_t, dtype=np.float64))         # (3,)
    return [ResidualBlock(A=A_block, c=c_block, name="W_L_lumped")]
