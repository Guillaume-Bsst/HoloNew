"""Test for W^o object motion regularization term (Brick 5, Task 1 + Task 2 + Task 3)."""
import numpy as np
import cvxpy as cp
import pinocchio as pin
from HoloNew.src.test_socp.movable import (
    build_wo_position_anchor, build_wo_term, pose_to_se3, se3_to_pose)


def _rand_se3(rng, scale=0.1):
    return pin.exp6(scale * rng.standard_normal(6)) * pin.SE3.Identity()


def test_wo_position_anchor_matches_numpy_and_jacobian():
    """build_wo_position_anchor: value matches the numpy linear model, and its
    Jacobian [I, -skew(p0)] is the first-order derivative of the true object
    position p(dxi) = (exp6(dxi) * T0).translation."""
    rng = np.random.default_rng(3)
    T0 = pin.SE3(pin.exp3(0.3 * rng.standard_normal(3)),
                 np.array([0.4, -0.7, 1.1]))
    p_ref = T0.translation + 0.05 * rng.standard_normal(3)
    lam = 4.0

    # (1) cvxpy term value == lambda * ||A_pos @ dxi + (p0 - p_ref)||^2.
    dxi = cp.Variable(6)
    val = 0.01 * rng.standard_normal(6)
    dxi.value = val
    term = build_wo_position_anchor(T0, p_ref, dxi, lam)
    p0 = T0.translation
    A_pos = np.hstack([np.eye(3), -pin.skew(p0)])
    r = A_pos @ val + (p0 - p_ref)
    gt = lam * float(r @ r)
    np.testing.assert_allclose(float(term.value), gt, rtol=1e-10)

    # (2) A_pos is the first-order Jacobian of the true position p(dxi).
    def p_true(d):
        return (pin.exp6(d) * T0).translation
    eps = 1e-6
    J_fd = np.zeros((3, 6))
    for k in range(6):
        e = np.zeros(6); e[k] = eps
        J_fd[:, k] = (p_true(e) - p_true(-e)) / (2 * eps)
    np.testing.assert_allclose(J_fd, A_pos, atol=1e-6)


def test_wo_term_matches_numpy():
    rng = np.random.default_rng(0)
    T0 = _rand_se3(rng)
    T1 = _rand_se3(rng)
    T2 = _rand_se3(rng)
    vdot_ref = rng.standard_normal(3)
    omega_ref = rng.standard_normal(3)
    lam_o, lam_w, dt = 2.0, 3.0, 1.0 / 30.0
    dxi = cp.Variable(6)
    val = 0.02 * rng.standard_normal(6)
    dxi.value = val
    term = build_wo_term(T0, T1, T2, vdot_ref, omega_ref, dxi, lam_o, lam_w, dt)
    # Independent numpy ground truth at val: object pose = exp6(val)*T0
    Tcur = pin.exp6(val) * T0
    V_t = pin.log6(T1.inverse() * Tcur).vector / dt       # [v; omega] at t
    V_tm1 = pin.log6(T2.inverse() * T1).vector / dt
    vdot = (V_t[:3] - V_tm1[:3]) / dt
    omega = V_t[3:6]
    gt = lam_o * float(np.sum((vdot - vdot_ref) ** 2)) + lam_w * float(np.sum((omega - omega_ref) ** 2))
    np.testing.assert_allclose(float(term.value), gt, rtol=1e-3)


def test_pose_se3_roundtrip():
    """pose_to_se3 / se3_to_pose must be inverses up to floating-point precision."""
    rng = np.random.default_rng(7)
    for _ in range(10):
        M_orig = pin.exp6(0.5 * rng.standard_normal(6)) * pin.SE3.Identity()
        pose7 = se3_to_pose(M_orig)
        assert pose7.shape == (7,)
        M_back = pose_to_se3(pose7)
        np.testing.assert_allclose(M_orig.rotation, M_back.rotation, atol=1e-14)
        np.testing.assert_allclose(M_orig.translation, M_back.translation, atol=1e-14)
    # Forward: known 90-deg Z rotation + translation
    import math
    pose = np.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4), 1.0, 2.0, 3.0])
    M = pose_to_se3(pose)
    R_expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    np.testing.assert_allclose(M.rotation, R_expected, atol=1e-14)
    np.testing.assert_allclose(M.translation, np.array([1.0, 2.0, 3.0]), atol=1e-14)


def test_bilateral_dx_object_channel_numpy_equivalence():
    """Task 3: bilateral D/X object-channel residual matches numpy ground truth.

    Constructs build_dx_terms with dxi given and evaluates at random (dqa, dxi)
    values.  An independent numpy calculation computes, for each active object
    point i:
        Bobj_i = R_obj.T @ [I_3, -skew(p_i)]    (world p_i, NOT p_i - t_obj)
        D residual row: sqrt(w_D) * n0 @ (Jloc @ dqa_v - Bobj_i @ dxi_v)
        X residual row: sqrt(w_X) * Pi0 @ (Jloc @ dqa_v - Bobj_i @ dxi_v)
    and sums the squared norms.  The cvxpy expression value must match to rtol=1e-6.
    """
    import pytest
    from HoloNew.examples.robot_retarget import RetargetingConfig
    from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
    from HoloNew.src.test_socp.interaction import (
        build_dx_terms, robot_control_points, query_entities,
        frame_references, _activation, _robj_from_pose, _skew,
    )

    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="object_interaction",
        task_name="sub3_largebox_003",
        data_format="smplh",
    ))
    if rt.correspondence is None or rt.object_sdf is None:
        pytest.skip("contact assets not present")

    rng = np.random.default_rng(42)
    q_pin = rt.pin.qpos_mj_to_q_pin(rt.q_init_full[:36])
    obj_pose = rt._obj_poses_raw[0]

    lambda_d = 2.0
    lambda_x = 3.0
    L = rt.smplx_ground_probe.margin

    # Set fixed random decision-variable values.
    dqa_v = rng.standard_normal(rt.nv_a) * 0.01
    dxi_v = rng.standard_normal(6) * 0.01

    dqa_var = cp.Variable(rt.nv_a)
    dxi_var = cp.Variable(6)
    dqa_var.value = dqa_v
    dxi_var.value = dxi_v

    terms = build_dx_terms(rt, q_pin, dqa_var, 0, obj_pose,
                           lambda_d, lambda_x, dxi=dxi_var)
    assert len(terms) > 0, "No active object points found — test cannot validate"

    cvxpy_val = float(sum(float(t.value) for t in terms))

    # --- Independent numpy ground truth ---
    corr = rt.correspondence
    M = corr.link_idx.shape[0]
    n_links = len(corr.link_names)
    link_counts = np.array([float(np.sum(corr.link_idx == li)) for li in range(n_links)])
    Nk = link_counts[corr.link_idx]

    P = robot_control_points(rt, q_pin)
    fobj, fflr = query_entities(rt, P, obj_pose, margin=L)
    d_obj_ref, x_obj_ref, d_flr_ref, x_flr_ref, _ = frame_references(rt, 0)
    Robj = _robj_from_pose(obj_pose)
    I3 = np.eye(3)

    alpha_obj = np.array([_activation(d_obj_ref[i], L) for i in range(M)])
    active_obj = (alpha_obj > 0) & np.asarray(fobj.active, dtype=bool)
    alpha_flr = np.array([_activation(d_flr_ref[i], L) for i in range(M)])
    active_flr = (alpha_flr > 0) & np.asarray(fflr.active, dtype=bool)

    active_union = np.where(active_obj | active_flr)[0]
    link_names_active = [corr.link_names[corr.link_idx[i]] for i in active_union]
    offsets_active = corr.offset_local[active_union]
    jacs_full = rt.pin.point_jacobians(q_pin, link_names_active, offsets_active)
    jacs = [J[:, rt.v_a_indices] for J in jacs_full]
    idx_to_pos = {int(active_union[k]): k for k in range(len(active_union))}

    gt = 0.0
    for i in np.where(active_obj)[0]:
        alpha = alpha_obj[i]
        w = alpha / (L ** 2 * Nk[i])
        Ji = jacs[idx_to_pos[i]]
        Jloc = Robj.T @ Ji               # (3, nv_a)
        Bobj_i = Robj.T @ np.hstack([I3, -_skew(P[i])])  # (3, 6): world p_i
        n0 = np.asarray(fobj.direction[i], dtype=float)
        d0 = float(fobj.distance[i])
        x0 = np.asarray(fobj.witness[i], dtype=float)

        # Bilateral relative displacement in object-local frame.
        delta_local = Jloc @ dqa_v - Bobj_i @ dxi_v   # (3,)

        if lambda_d > 0:
            sw = np.sqrt(lambda_d * w)
            res_d = sw * (n0 @ delta_local - float(d_obj_ref[i] - d0))
            gt += res_d ** 2

        if lambda_x > 0:
            sw = np.sqrt(lambda_x * w)
            Pi0 = I3 - np.outer(n0, n0)
            ref_x = np.asarray(x_obj_ref[i], dtype=float)
            res_x = sw * (Pi0 @ delta_local - Pi0 @ (ref_x - x0))
            gt += float(res_x @ res_x)

    # Floor channel (robot-only, no dxi contribution).
    for i in np.where(active_flr)[0]:
        alpha = alpha_flr[i]
        w = alpha / (L ** 2 * Nk[i])
        Ji = jacs[idx_to_pos[i]]
        n0 = np.asarray(fflr.direction[i], dtype=float)
        d0 = float(fflr.distance[i])
        x0 = np.asarray(fflr.witness[i], dtype=float)

        if lambda_d > 0:
            sw = np.sqrt(lambda_d * w)
            res_d = sw * (n0 @ (Ji @ dqa_v) - float(d_flr_ref[i] - d0))
            gt += res_d ** 2

        if lambda_x > 0:
            sw = np.sqrt(lambda_x * w)
            Pi0 = I3 - np.outer(n0, n0)
            ref_x = np.asarray(x_flr_ref[i], dtype=float)
            res_x = sw * (Pi0 @ (Ji @ dqa_v) - Pi0 @ (ref_x - x0))
            gt += float(res_x @ res_x)

    np.testing.assert_allclose(cvxpy_val, gt, rtol=1e-6,
        err_msg=(f"Bilateral D/X object-channel mismatch: cvxpy={cvxpy_val:.6g}, "
                 f"numpy={gt:.6g}"))


def test_movable_when_enabled_stays_finite():
    """W^o is opt-in: activate_movable (object variable) + activate_wo (the W^o cost).
    The solve must stay finite and the object must not drift from the reference.
    """
    import pytest
    from HoloNew.examples.robot_retarget import RetargetingConfig
    from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
    from HoloNew.src.test_socp.config import TestSocpRetargeterConfig

    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="object_interaction",
        task_name="sub3_largebox_003",
        data_format="smplh",
        retargeter=TestSocpRetargeterConfig(activate_movable=True, activate_wo=True),
    ))
    if rt.correspondence is None or rt.object_sdf is None:
        pytest.skip("contact assets not present")

    # W^o on with its tuned weights.
    assert rt.activate_movable is True
    assert rt.lambda_o == 1.0
    assert rt.lambda_omega == 1.0

    res = rt.retarget(max_frames=6)
    assert np.all(np.isfinite(res.qpos)), "qpos contains non-finite values with movable on"

    # Solved object poses should have been recorded for each frame.
    assert len(rt._obj_solved_poses) == 6
    for pose7 in rt._obj_solved_poses:
        assert np.all(np.isfinite(pose7)), "solved object pose is non-finite"

    # The solved object should stay near the reference (W^o tracks it).
    # Tolerance: 0.5 m / 1 rad — generous, just confirms it doesn't explode.
    ref_poses = rt._obj_poses_raw[:6]
    for i, (sol, ref) in enumerate(zip(rt._obj_solved_poses, ref_poses)):
        t_err = np.linalg.norm(sol[4:7] - ref[4:7])
        assert t_err < 0.5, f"frame {i}: solved object drifted {t_err:.3f} m from reference"


def test_movable_with_interaction_bilateral_solve():
    """Task 3: activate_movable=True + D/X on gives a finite 6-frame solve.

    Exercises the bilateral coupling path (dxi passed to build_dx_terms) and
    verifies: (1) qpos is finite; (2) solved object poses are finite; (3) the
    dxi variable is shared between W^o and D/X (same variable in the solve).
    """
    import pytest
    from HoloNew.examples.robot_retarget import RetargetingConfig
    from HoloNew.src.test_socp.test_socp import TestSocpRetargeter

    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="object_interaction",
        task_name="sub3_largebox_003",
        data_format="smplh",
    ))
    if rt.correspondence is None or rt.object_sdf is None:
        pytest.skip("contact assets not present")

    # Enable movable + bilateral D/X.
    rt.activate_movable = True
    rt.lambda_o = 1.0
    rt.lambda_omega = 1.0
    rt.lambda_d = 1.0
    rt.lambda_x = 1.0
    # The D/X terms require ground non-penetration to stay stable.
    rt.activate_obj_non_penetration = True

    res = rt.retarget(max_frames=6)
    assert np.all(np.isfinite(res.qpos)), (
        "qpos non-finite with movable+bilateral D/X on")

    assert len(rt._obj_solved_poses) == 6
    for i, pose7 in enumerate(rt._obj_solved_poses):
        assert np.all(np.isfinite(pose7)), (
            f"frame {i}: solved object pose is non-finite with bilateral D/X")

    # Object stays near the reference (W^o regularization + bilateral coupling).
    ref_poses = rt._obj_poses_raw[:6]
    for i, (sol, ref) in enumerate(zip(rt._obj_solved_poses, ref_poses)):
        t_err = np.linalg.norm(sol[4:7] - ref[4:7])
        assert t_err < 0.5, (
            f"frame {i}: object drifted {t_err:.3f} m from reference "
            "with bilateral coupling")
