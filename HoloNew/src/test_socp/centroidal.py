"""Centroidal W^c (CoM acceleration) + W^c_pos (CoM position) + W^L (angular momentum).

See docs/specs/2026-06-13-brick4-centroidal-design.md.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np


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


def build_lumped_L_term(rt, q_pin, q_pin_prev, dqa, frames, masses, L_ref_t,
                        lambda_l, dt):
    """W^L tracking: lambda_l * ||L_lumped(dqa) - L_ref_t||^2.

    Current lumped orbital angular momentum, linearized in dqa with the moment arms
    held at the current config (mirroring how A_G(q0) @ v fixes q0):
        L(dqa) = sum_k m_k (p_k0 - c0) x (v_k(dqa) - cdot(dqa)),
        v_k(dqa) = (p_k0 - p_k_prev)/dt + (J_k/dt) dqa,
    which is affine in dqa. p_k0/J_k at q_pin; p_k_prev at q_pin_prev (previous solved
    frame). Returns a scalar cvxpy expression.
    """
    import cvxpy as cp
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

    r = np.sqrt(lambda_l) * (A_L @ dqa + (b_L0 - np.asarray(L_ref_t, dtype=np.float64)))
    return cp.sum_squares(r)


def build_centroidal_terms(rt, q_t0, q_tm1, c_tm1, c_tm2, cddot_ref, c_ref, dqa,
                            lambda_c, lambda_c_pos, lambda_l, dt):
    """Assemble W^c (CoM accel) + W^c_pos (CoM position) + W^L (angular momentum -> 0).

    W^c = lambda_c * ||c_ddot - cddot_ref||^2
        c_ddot = (c0 + Jc @ dqa - 2*c_tm1 + c_tm2) / dt^2   (linearised in dqa)
        Jc = com_jacobian(q_t0)[:, v_a_indices]

    W^c_pos = lambda_c_pos * ||c0 + Jc @ dqa - c_ref||^2
        c_ref: reference CoM position for this frame (absolute anchor).
        Prevents the constant-velocity drift that W^c alone cannot constrain.

    W^L = lambda_l * ||L||^2
        L = (A_G @ v)[3:6],  v = difference(q_tm1, q_t0) + Jd[:, v_a_indices] @ dqa
        A_G = centroidal_map(q_t0)

    Each term is represented as cp.sum_squares(A @ dqa + b) so that the sqrt(lambda)
    folding makes the squared norm equal to the weighted residual.

    c0 and Jc are computed once and reused by W^c and W^c_pos.

    Terms with lambda == 0 are skipped (not appended).

    Args:
        rt: TestSocpRetargeter instance (provides rt.pin, rt.v_a_indices).
        q_t0: pinocchio config at current time step (q_t).
        q_tm1: pinocchio config at previous time step (q_{t-1}).
        c_tm1: CoM (3,) at t-1, previously solved.
        c_tm2: CoM (3,) at t-2, previously solved.
        cddot_ref: reference CoM acceleration (3,).
        c_ref: reference CoM position for this frame (3,).
        dqa: cvxpy Variable (nv_a,), the active-joint tangent increment.
        lambda_c: weight for W^c (CoM acceleration tracking).
        lambda_c_pos: weight for W^c_pos (CoM absolute position anchor).
        lambda_l: weight for W^L (angular momentum -> 0).
        dt: time step in seconds.

    Returns:
        List of up to three cvxpy expressions [W_c_expr, W_c_pos_expr, W_L_expr],
        skipping any term whose lambda is 0.
    """
    terms = []

    # Shared quantities: c0 and Jc reused by W^c and W^c_pos.
    c0 = rt.pin.com(q_t0)                                      # (3,)
    Jc = rt.pin.com_jacobian(q_t0)[:, rt.v_a_indices]          # (3, nv_a)

    # --- W^c: CoM acceleration tracking ---
    # In inertia mode W^c is kept WEAK (below the interaction weights): when
    # contacts are active they place the body and W^c only shapes the residual;
    # when contacts deactivate (free flight) W^c becomes the sole CoM term and the
    # ballistic parabola emerges (cddot_ref carries -g z from the data). The
    # free-flight branch is implemented but UNVALIDATED — no demo_data clip
    # contains a flight phase (see the inertia-mode design doc).
    if lambda_c > 0:
        # Fold sqrt(lambda_c) and 1/dt^2 into A and b so that:
        #   ||A_c @ dqa + b_c||^2 = lambda_c * ||cddot - cddot_ref||^2
        s_c = np.sqrt(lambda_c) / dt**2
        A_c = s_c * Jc                                          # (3, nv_a)
        b_c = (s_c * (c0 - 2.0*np.asarray(c_tm1) + np.asarray(c_tm2))
               - np.sqrt(lambda_c) * np.asarray(cddot_ref))    # (3,)
        terms.append(cp.sum_squares(A_c @ dqa + b_c))

    # --- W^c_pos: CoM absolute position anchor ---
    if lambda_c_pos > 0:
        # ||A_p @ dqa + b_p||^2 = lambda_c_pos * ||c0 + Jc @ dqa - c_ref||^2
        s_p = np.sqrt(lambda_c_pos)
        A_p = s_p * Jc                                          # (3, nv_a)
        b_p = s_p * (c0 - np.asarray(c_ref))                   # (3,)
        terms.append(cp.sum_squares(A_p @ dqa + b_p))

    # --- W^L: weak angular-momentum spin regularizer (toward 0) ---
    # NOTE: the paper tracks the reference centroidal angular momentum
    # (||L - L_ref||^2). Computing L_ref needs a reference robot VELOCITY (not just
    # target orientations) and a free-flight clip to matter (in stance L is
    # dominated by the contacts). We have neither, so W^L is kept as a weak
    # spin-toward-zero regularizer — sane for the grounded manipulation/climb clips
    # available. True L_ref tracking is deferred (see the inertia-mode design doc).
    if lambda_l > 0:
        Ag = rt.pin.centroidal_map(q_t0)                       # (6, nv)
        v0, Jd = rt.pin.difference_and_jac(q_tm1, q_t0)       # v0: (nv,), Jd: (nv, nv)
        AgL = Ag[3:6, :]                                       # (3, nv)
        # v_active contribution via Jd columns for active joints
        A_L = np.sqrt(lambda_l) * (AgL @ Jd[:, rt.v_a_indices])  # (3, nv_a)
        b_L = np.sqrt(lambda_l) * (AgL @ v0)                   # (3,)
        terms.append(cp.sum_squares(A_L @ dqa + b_L))

    return terms
