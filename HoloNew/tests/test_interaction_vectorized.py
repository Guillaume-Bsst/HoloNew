"""Equivalence tests for the vectorized D/X/P cvxpy assembly.

These tests prove that the vectorized build_dx_terms / build_p_terms produce
objectives numerically identical (rtol=1e-8) to an independent per-point numpy
ground truth.  The ground truth is computed entirely in numpy (plain loops) and
does NOT reuse any matrices from the vectorized path — so any drift will be caught.

Step 1: run against the current (loop) implementation to validate the ground truth.
Step 2: run again after vectorization — must still pass.
"""
from __future__ import annotations

import numpy as np
import pytest
import cvxpy as cp


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _rt():
    from HoloNew.examples.robot_retarget import RetargetingConfig
    from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
    return TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="object_interaction", task_name="sub3_largebox_003",
        data_format="smplh"))


def _skip_if_missing(rt):
    if rt.correspondence is None or rt.object_sdf is None:
        pytest.skip("assets not present")


# ---------------------------------------------------------------------------
# Ground-truth helpers (independent per-point numpy)
# ---------------------------------------------------------------------------

def _pointwise_dx_objective(rt, q_pin: np.ndarray, val: np.ndarray,
                             t: int, obj_pose: np.ndarray,
                             lambda_D: float, lambda_X: float) -> float:
    """Per-point numpy ground truth for build_dx_terms evaluated at dqa=val.

    Replicates every coefficient from the spec:
      - active-set selection (alpha > 0 AND fobj/fflr.active)
      - alpha(d_ref) = clamp((1 - d_ref/L)^2, 0)
      - weight w = alpha / (L^2 * N_k)
      - object channel: residuals in object-local frame (Robj.T @ Ji)
      - floor channel: residuals in world frame (Ji directly)
      - D: (n0^T J dqa - (d_ref - d0))^2  * lambda_D * w
      - X: || Pi0 J dqa - Pi0(x_ref - x0) ||^2  * lambda_X * w
    """
    from HoloNew.src.test_socp.interaction import (
        robot_control_points, query_entities, frame_references,
        _activation, _robj_from_pose,
    )

    corr = rt.correspondence
    M = corr.link_idx.shape[0]
    L = rt.smplx_ground_probe.margin

    # Per-link 1/N_k counts.
    n_links = len(corr.link_names)
    link_counts = np.zeros(n_links, dtype=float)
    for li in range(n_links):
        link_counts[li] = float(np.sum(corr.link_idx == li))
    Nk = link_counts[corr.link_idx]  # (M,)

    P = robot_control_points(rt, q_pin)
    fobj, fflr = query_entities(rt, P, obj_pose, margin=L)
    d_obj_ref, x_obj_ref, d_flr_ref, x_flr_ref, _p_ref = frame_references(rt, t)
    Robj = _robj_from_pose(obj_pose)

    alpha_obj = np.array([_activation(d_obj_ref[i], L) for i in range(M)])
    active_obj = (alpha_obj > 0) & np.asarray(fobj.active, dtype=bool)
    alpha_flr = np.array([_activation(d_flr_ref[i], L) for i in range(M)])
    active_flr = (alpha_flr > 0) & np.asarray(fflr.active, dtype=bool)

    active_union = np.where(active_obj | active_flr)[0]
    if active_union.size == 0:
        return 0.0

    link_names_active = [corr.link_names[corr.link_idx[i]] for i in active_union]
    offsets_active = corr.offset_local[active_union]
    jacs_full = rt.pin.point_jacobians(q_pin, link_names_active, offsets_active)
    jacs = [J[:, rt.v_a_indices] for J in jacs_full]
    idx_to_pos = {int(active_union[k]): k for k in range(len(active_union))}

    I3 = np.eye(3)
    total = 0.0

    # Object channel.
    for i in np.where(active_obj)[0]:
        alpha = alpha_obj[i]
        w = alpha / (L ** 2 * Nk[i])
        Ji = jacs[idx_to_pos[i]]
        Jloc = Robj.T @ Ji
        n0 = np.asarray(fobj.direction[i], dtype=float)
        d0 = float(fobj.distance[i])
        x0 = np.asarray(fobj.witness[i], dtype=float)

        if lambda_D > 0:
            res_d = n0 @ (Jloc @ val) - float(d_obj_ref[i] - d0)
            total += (lambda_D * w) * res_d ** 2

        if lambda_X > 0:
            Pi0 = I3 - np.outer(n0, n0)
            rhs_x = Pi0 @ (np.asarray(x_obj_ref[i], dtype=float) - x0)
            res_x = Pi0 @ (Jloc @ val) - rhs_x
            total += (lambda_X * w) * float(res_x @ res_x)

    # Floor channel.
    for i in np.where(active_flr)[0]:
        alpha = alpha_flr[i]
        w = alpha / (L ** 2 * Nk[i])
        Ji = jacs[idx_to_pos[i]]
        n0 = np.asarray(fflr.direction[i], dtype=float)
        d0 = float(fflr.distance[i])
        x0 = np.asarray(fflr.witness[i], dtype=float)

        if lambda_D > 0:
            res_d = n0 @ (Ji @ val) - float(d_flr_ref[i] - d0)
            total += (lambda_D * w) * res_d ** 2

        if lambda_X > 0:
            Pi0 = I3 - np.outer(n0, n0)
            rhs_x = Pi0 @ (np.asarray(x_flr_ref[i], dtype=float) - x0)
            res_x = Pi0 @ (Ji @ val) - rhs_x
            total += (lambda_X * w) * float(res_x @ res_x)

    return total


def _pointwise_p_objective(rt, q_pin: np.ndarray, val: np.ndarray,
                            t: int, obj_pose: np.ndarray,
                            lambda_P: float, sigma_v: float, dt: float) -> float:
    """Per-point numpy ground truth for build_p_terms evaluated at dqa=val."""
    from HoloNew.src.test_socp.interaction import (
        robot_control_points, query_entities, frame_references,
        _activation, _robj_from_pose,
    )

    state = rt._p_state
    corr = rt.correspondence
    M = corr.link_idx.shape[0]
    L = rt.smplx_ground_probe.margin
    scale_sq = lambda_P / (sigma_v * dt) ** 2

    n_links = len(corr.link_names)
    link_counts = np.zeros(n_links, dtype=float)
    for li in range(n_links):
        link_counts[li] = float(np.sum(corr.link_idx == li))
    Nk = link_counts[corr.link_idx]

    Robj_t = _robj_from_pose(obj_pose)
    obj_t = np.asarray(obj_pose[4:7], dtype=float)
    obj_prev_pose = state["obj_prev"]
    Robj_tm1 = _robj_from_pose(obj_prev_pose)
    obj_tm1 = np.asarray(obj_prev_pose[4:7], dtype=float)

    P = robot_control_points(rt, q_pin)
    fobj, fflr = query_entities(rt, P, obj_pose, margin=L)

    d_obj_ref_t,  _,  d_flr_ref_t,  _, p_ref_t   = frame_references(rt, t)
    d_obj_ref_tm1, _, d_flr_ref_tm1, _, p_ref_tm1 = frame_references(rt, t - 1)

    d_prev_obj  = state["d_prev_obj"]
    d_prev_flr  = state["d_prev_flr"]
    a_prev_obj  = state["a_prev_obj"]
    a_prev_flr  = state["a_prev_flr"]
    p_prev_world = state["p_prev_world"]
    dp_ref_world = p_ref_t - p_ref_tm1

    alpha_obj_t = np.array([_activation(d_obj_ref_t[i], L) for i in range(M)])
    alpha_flr_t = np.array([_activation(d_flr_ref_t[i], L) for i in range(M)])

    def _hat(d_prev_i):
        return _activation(float(d_prev_i), L)

    gamma_obj = np.minimum(np.minimum(alpha_obj_t, a_prev_obj),
                           np.array([_hat(d_prev_obj[i]) for i in range(M)]))
    active_obj = (gamma_obj > 0) & np.asarray(fobj.active, dtype=bool)

    gamma_flr = np.minimum(np.minimum(alpha_flr_t, a_prev_flr),
                           np.array([_hat(d_prev_flr[i]) for i in range(M)]))
    active_flr = (gamma_flr > 0) & np.asarray(fflr.active, dtype=bool)

    active_union = np.where(active_obj | active_flr)[0]
    if active_union.size == 0:
        return 0.0

    link_names_active = [corr.link_names[corr.link_idx[i]] for i in active_union]
    offsets_active = corr.offset_local[active_union]
    jacs_full = rt.pin.point_jacobians(q_pin, link_names_active, offsets_active)
    jacs = [J[:, rt.v_a_indices] for J in jacs_full]
    idx_to_pos = {int(active_union[k]): k for k in range(len(active_union))}

    I3 = np.eye(3)
    total = 0.0

    # Object channel.
    for i in np.where(active_obj)[0]:
        gamma = gamma_obj[i]
        w = gamma / Nk[i]
        Ji = jacs[idx_to_pos[i]]
        Jloc = Robj_t.T @ Ji
        n0 = np.asarray(fobj.direction[i], dtype=float)
        Pi0 = I3 - np.outer(n0, n0)

        p_prev_local_i = Robj_tm1.T @ (p_prev_world[i] - obj_tm1)
        const_i = Robj_t.T @ (P[i] - obj_t) - p_prev_local_i
        p_ref_t_loc   = Robj_t.T   @ (p_ref_t[i]   - obj_t)
        p_ref_tm1_loc = Robj_tm1.T @ (p_ref_tm1[i] - obj_tm1)
        dp_ref_loc = p_ref_t_loc - p_ref_tm1_loc
        rhs_const = Pi0 @ (dp_ref_loc - const_i)

        res = Pi0 @ (Jloc @ val) - rhs_const
        total += (scale_sq * w) * float(res @ res)

    # Floor channel.
    for i in np.where(active_flr)[0]:
        gamma = gamma_flr[i]
        w = gamma / Nk[i]
        Ji = jacs[idx_to_pos[i]]
        n0 = np.asarray(fflr.direction[i], dtype=float)
        Pi0 = I3 - np.outer(n0, n0)

        const_i   = P[i] - p_prev_world[i]
        dp_ref_i  = dp_ref_world[i]
        rhs_const = Pi0 @ (dp_ref_i - const_i)

        res = Pi0 @ (Ji @ val) - rhs_const
        total += (scale_sq * w) * float(res @ res)

    return total


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_dx_vectorized_matches_pointwise_numpy():
    """Vectorized D/X objective equals independent per-point numpy sum."""
    rt = _rt()
    _skip_if_missing(rt)

    from HoloNew.src.test_socp.interaction import build_dx_terms

    q_pin = rt.pin.qpos_mj_to_q_pin(rt.q_init_full[:36])
    obj_pose = (rt._obj_poses_raw[0]
                if getattr(rt, "_obj_poses_raw", None) is not None
                else np.array([1., 0, 0, 0, 0, 0, 0]))

    rng = np.random.default_rng(0)
    val = 0.01 * rng.standard_normal(rt.nv_a)

    dqa = cp.Variable(rt.nv_a)
    dqa.value = val

    lambda_D, lambda_X = 2.0, 3.0
    terms = build_dx_terms(rt, q_pin, dqa, 0, obj_pose,
                           lambda_D=lambda_D, lambda_X=lambda_X)

    if len(terms) == 0:
        pytest.skip("no active D/X points at frame 0 — cannot test equivalence")

    # Evaluate each cvxpy term at dqa.value and sum.
    vec_obj = float(sum(float(tm.value) for tm in terms))

    # Independent ground truth.
    gt = _pointwise_dx_objective(rt, q_pin, val, 0, obj_pose, lambda_D, lambda_X)

    np.testing.assert_allclose(vec_obj, gt, rtol=1e-8, atol=1e-10,
                               err_msg="vectorized D/X objective mismatch vs numpy ground truth")


def test_p_vectorized_matches_pointwise_numpy():
    """Vectorized P objective equals independent per-point numpy sum at t=1."""
    rt = _rt()
    _skip_if_missing(rt)

    from HoloNew.src.test_socp.interaction import (
        build_p_terms, robot_control_points, query_entities, _activation,
    )

    q_pin = rt.pin.qpos_mj_to_q_pin(rt.q_init_full[:36])
    L = rt.smplx_ground_probe.margin
    M = rt.correspondence.link_idx.shape[0]

    obj_pose = (rt._obj_poses_raw[1]
                if getattr(rt, "_obj_poses_raw", None) is not None
                else np.array([1., 0, 0, 0, 0, 0, 0]))
    obj_prev_pose = (rt._obj_poses_raw[0]
                     if getattr(rt, "_obj_poses_raw", None) is not None
                     else np.array([1., 0, 0, 0, 0, 0, 0]))

    P = robot_control_points(rt, q_pin)
    fobj, fflr = query_entities(rt, P, obj_pose, margin=L)

    # Build a plausible _p_state (same as existing test_p_terms_assemble).
    rt._p_state = {
        "p_prev_world": P.copy(),
        "obj_prev": obj_prev_pose.copy(),
        "d_prev_obj": np.asarray(fobj.distance, dtype=np.float64),
        "d_prev_flr": np.asarray(fflr.distance, dtype=np.float64),
        "a_prev_obj": np.array([_activation(float(fobj.distance[i]), L) for i in range(M)]),
        "a_prev_flr": np.array([_activation(float(fflr.distance[i]), L) for i in range(M)]),
    }

    rng = np.random.default_rng(42)
    val = 0.01 * rng.standard_normal(rt.nv_a)

    dqa = cp.Variable(rt.nv_a)
    dqa.value = val

    lambda_P, sigma_v, dt = 2.0, 0.05, 1.0 / 30.0
    terms = build_p_terms(rt, q_pin, dqa, t=1, obj_pose=obj_pose,
                          lambda_P=lambda_P, sigma_v=sigma_v, dt=dt)

    if len(terms) == 0:
        pytest.skip("no active P points — cannot test equivalence")

    vec_obj = float(sum(float(tm.value) for tm in terms))

    gt = _pointwise_p_objective(rt, q_pin, val, 1, obj_pose,
                                lambda_P, sigma_v, dt)

    np.testing.assert_allclose(vec_obj, gt, rtol=1e-8, atol=1e-10,
                               err_msg="vectorized P objective mismatch vs numpy ground truth")
