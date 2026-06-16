"""Feed-forward object warm-start: linearization point for the next frame is the
previous SOLVED object pose advanced by the reference's per-frame increment.

    T_warm = (T_ref[t] · T_ref[t-1]^{-1}) · T_solved[t-1]

This accumulates the grasp correction (it rides on T_solved, not the reference) AND
pre-applies the reference motion (no lag on fast object motion). It reduces to the
raw reference when the previous solve matched the reference.
"""
import numpy as np
import pinocchio as pin


def _pose7(M: pin.SE3) -> np.ndarray:
    from HoloNew.src.test_socp.movable import se3_to_pose
    return se3_to_pose(M)


def test_warmstart_constant_velocity_no_lag():
    """Constant-velocity reference, previous solve == reference: warm-start lands
    exactly on the current reference (the feed-forward predicts the motion)."""
    from HoloNew.src.test_socp.movable import feedforward_object_warmstart, pose_to_se3
    rng = np.random.default_rng(0)
    # Constant SE(3) increment dT applied each frame.
    dT = pin.exp6(np.array([0.3, -0.1, 0.2, 0.0, 0.0, 0.4]))  # fast motion
    T0 = pin.SE3(pin.exp3(0.2 * rng.standard_normal(3)), rng.standard_normal(3))
    ref_tm1 = T0
    ref_t = dT * ref_tm1
    solved_tm1 = ref_tm1  # previous solve matched the reference
    warm = feedforward_object_warmstart(_pose7(ref_t), _pose7(ref_tm1), _pose7(solved_tm1))
    # warm should equal ref_t
    np.testing.assert_allclose(warm, _pose7(ref_t), atol=1e-10)


def test_warmstart_preserves_grasp_offset():
    """When the previous solve carried a body-frame offset E (solved = ref·E), the
    warm-start re-applies that offset to the current reference: warm = ref[t]·E."""
    from HoloNew.src.test_socp.movable import feedforward_object_warmstart, pose_to_se3
    rng = np.random.default_rng(1)
    dT = pin.exp6(np.array([0.1, 0.05, -0.2, 0.1, 0.0, 0.0]))
    ref_tm1 = pin.SE3(pin.exp3(0.1 * rng.standard_normal(3)), rng.standard_normal(3))
    ref_t = dT * ref_tm1
    E = pin.exp6(np.array([0.02, -0.01, 0.0, 0.05, 0.03, -0.02]))  # grasp offset (body frame)
    solved_tm1 = ref_tm1 * E
    warm = feedforward_object_warmstart(_pose7(ref_t), _pose7(ref_tm1), _pose7(solved_tm1))
    expected = _pose7(ref_t * E)
    np.testing.assert_allclose(warm, expected, atol=1e-10)


def test_loop_tracks_constant_velocity_object_no_lag():
    """End-to-end wiring: with a synthetic constant-velocity object reference and the
    object driven purely by W^o + the feed-forward warm-start (contacts off), the
    solved object trajectory tracks the reference without lag."""
    import pytest
    from HoloNew.examples.robot_retarget import RetargetingConfig
    from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
    from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
    from HoloNew.src.test_socp.movable import se3_to_pose

    cfg = TestSocpRetargeterConfig(
        activate_tm=True, activate_wo=True,           # object variable + W^o motion reg
        activate_wd=False, activate_wx=False,          # no contact channel
        activate_persistence=False, activate_wo_pos=False)
    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh",
        retargeter=cfg))
    # Inject a synthetic constant-SE(3)-velocity object reference of the clip length.
    T = rt.human_quat.shape[0]
    dT = pin.exp6(np.array([0.15, -0.05, 0.1, 0.0, 0.0, 0.3]))  # fast: ~0.18 m + 0.3 rad/frame
    M = pin.SE3(pin.exp3(np.array([0.1, 0.0, 0.0])), np.array([0.2, -0.1, 0.9]))
    ref = []
    for _ in range(T):
        ref.append(se3_to_pose(M)); M = dT * M
    rt._obj_poses_raw = np.asarray(ref)

    n = 5
    res = rt.retarget(max_frames=n)
    sp = getattr(res, "solved_object_poses", None)
    assert sp is not None and len(sp) >= n, "solved object poses not collected"
    # Solved object tracks the reference (no lag): compare positions over the run.
    ref_pos = np.asarray(ref)[:len(sp), 4:7]
    sol_pos = sp[:len(sp), 4:7]
    max_lag = np.abs(sol_pos - ref_pos).max()
    assert max_lag < 1e-2, f"object lags the reference by {max_lag:.4f} m"
