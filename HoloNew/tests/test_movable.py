"""Test for W^o object motion regularization term (Brick 5, Task 1 + Task 2)."""
import numpy as np
import cvxpy as cp
import pinocchio as pin
from HoloNew.src.test_socp.movable import build_wo_term, pose_to_se3, se3_to_pose


def _rand_se3(rng, scale=0.1):
    return pin.exp6(scale * rng.standard_normal(6)) * pin.SE3.Identity()


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


def test_movable_default_off_and_runs_on():
    """activate_movable=False by default; True runs the W^o solve and stays finite."""
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

    # Default: movable is off.
    assert rt.activate_movable is False

    # Enable movable with W^o weights.
    rt.activate_movable = True
    rt.lambda_o = 1.0
    rt.lambda_omega = 1.0

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
