"""Object-as-carrier interaction (movable object <-> environment): the object-floor
D/X with independent weights + the new object-floor persistence (P), symmetric to
the robot D/X/P. Tested with a mock rt (only object_surface_local is needed)."""
from types import SimpleNamespace

import cvxpy as cp
import numpy as np
import pinocchio as pin


def _near_floor_object():
    """Synthetic object: surface points straddling z=0, mock rt + a near-floor pose."""
    rng = np.random.default_rng(0)
    # Local points; the pose places them around z in [-0.04, 0.04] (within margin 0.1).
    p_local = rng.uniform(-0.05, 0.05, size=(40, 3))
    rt = SimpleNamespace(object_surface_local=p_local)
    return rt, p_local


def _pose7(M):
    from HoloNew.src.test_socp.movable import se3_to_pose
    return se3_to_pose(M)


def test_object_floor_dx_independent_weights():
    """build_object_floor_terms: D scales with lambda_d_obj, X with lambda_x_obj."""
    from HoloNew.src.test_socp.movable import build_object_floor_terms
    rt, _ = _near_floor_object()
    margin = 0.1
    obj_pose = _pose7(pin.SE3(np.eye(3), np.array([0.2, -0.1, 0.0])))
    dxi = cp.Variable(6); dxi.value = 0.01 * np.random.default_rng(1).standard_normal(6)
    # D-only
    d1 = build_object_floor_terms(rt, dxi, obj_pose, 1.0, 0.0, margin)
    d2 = build_object_floor_terms(rt, dxi, obj_pose, 2.0, 0.0, margin)
    s1 = sum(float(t.value) for t in d1); s2 = sum(float(t.value) for t in d2)
    assert s1 > 0
    np.testing.assert_allclose(s2, 2.0 * s1, rtol=1e-9)
    # X-only
    x1 = build_object_floor_terms(rt, dxi, obj_pose, 0.0, 1.0, margin)
    x2 = build_object_floor_terms(rt, dxi, obj_pose, 0.0, 2.0, margin)
    sx1 = sum(float(t.value) for t in x1); sx2 = sum(float(t.value) for t in x2)
    assert sx1 > 0
    np.testing.assert_allclose(sx2, 2.0 * sx1, rtol=1e-9)


def test_object_floor_persistence_matches_numpy():
    """build_object_floor_persistence: tangential no-slip residual matches an
    independent numpy ground truth at a fixed dxi."""
    from HoloNew.src.test_socp.movable import (
        build_object_floor_persistence, pose_to_se3)
    from HoloNew.src.test_socp.interaction import _activation, _skew, _p_scale_sq
    rt, p_local = _near_floor_object()
    M = p_local.shape[0]
    margin, sigma_v, dt = 0.1, 0.05, 1 / 30.0
    lam = 3.0
    obj_pose = _pose7(pin.SE3(np.eye(3), np.array([0.2, -0.1, 0.0])))
    obj_prev = _pose7(pin.SE3(pin.exp3(np.array([0.0, 0.0, 0.02])), np.array([0.19, -0.1, 0.0])))
    ref_t = _pose7(pin.SE3(np.eye(3), np.array([0.21, -0.1, 0.0])))
    ref_tm1 = _pose7(pin.SE3(np.eye(3), np.array([0.20, -0.1, 0.0])))
    dxi = cp.Variable(6); val = 0.01 * np.random.default_rng(2).standard_normal(6); dxi.value = val
    terms = build_object_floor_persistence(
        rt, dxi, obj_pose, obj_prev, ref_t, ref_tm1, lam, sigma_v, margin, dt)
    assert len(terms) == 1
    # Independent numpy ground truth.
    T0 = pose_to_se3(obj_pose); Tp = pose_to_se3(obj_prev)
    Trt = pose_to_se3(ref_t); Trtm1 = pose_to_se3(ref_tm1)
    p_w0 = p_local @ T0.rotation.T + T0.translation
    p_prev = p_local @ Tp.rotation.T + Tp.translation
    p_ref_t = p_local @ Trt.rotation.T + Trt.translation
    p_ref_tm1 = p_local @ Trtm1.rotation.T + Trtm1.translation
    z = np.array([0.0, 0.0, 1.0]); Pi0 = np.eye(3) - np.outer(z, z)
    scale_sq = _p_scale_sq(lam, sigma_v, dt)
    gt = 0.0
    for i in range(M):
        a = _activation(float(p_w0[i, 2]), margin)
        if a <= 0:
            continue
        Bi = np.hstack([np.eye(3), -_skew(p_w0[i])])
        dp_obj = (p_w0[i] - p_prev[i]) + Bi @ val
        dp_ref = p_ref_t[i] - p_ref_tm1[i]
        r = np.sqrt(scale_sq * a / M) * (Pi0 @ (dp_obj - dp_ref))
        gt += float(r @ r)
    np.testing.assert_allclose(float(terms[0].value), gt, rtol=1e-9)
