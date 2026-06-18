import numpy as np
import pytest
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.src.test_socp.solve.spec import ResidualBlock


@pytest.fixture(scope="module")
def rt():
    cfg = RetargetingConfig(task_type="robot_only", task_name="sub3_largebox_003",
                            data_format="smplh")
    return TestSocpRetargeter.from_config(cfg)


def test_tracking_blocks_valid(rt):
    from HoloNew.src.test_socp.tracking import build_tracking_blocks
    from HoloNew.src.test_socp.targets import ground_frame_targets
    from HoloNew.src.test_socp.tables import IK_MATCH_TABLE1
    q = rt.q_init_full.copy()
    tg = ground_frame_targets(rt.gmr_ground["pos"][0], rt.gmr_ground["quat"][0], IK_MATCH_TABLE1)
    blocks = build_tracking_blocks(rt, tg, q, lambda_pos=rt.lambda_pos, sigma_p=rt.sigma_p,
                                   lambda_rot=rt.lambda_rot, sigma_rot=rt.sigma_rot)
    assert blocks and all(isinstance(b, ResidualBlock) for b in blocks)
    for b in blocks:
        assert b.A.shape[1] == rt.nv_a
        assert np.all(np.isfinite(b.A))
        assert np.all(np.isfinite(b.c))


def test_style_blocks_valid(rt):
    from HoloNew.src.test_socp.style import build_style_blocks
    from HoloNew.src.test_socp.targets import ground_frame_targets
    from HoloNew.src.test_socp.tables import IK_MATCH_TABLE1
    q = rt.q_init_full.copy()
    tg = ground_frame_targets(rt.gmr_ground["pos"][0], rt.gmr_ground["quat"][0], IK_MATCH_TABLE1)
    blocks = build_style_blocks(rt, q, tg, lambda_ws=1.0, sigma_R=rt.sigma_R)
    assert blocks, (
        "build_style_blocks returned an empty list with lambda_ws=1.0 — "
        "check that frame_targets contains a pelvis entry (R_Bref lookup) and "
        "that at least one frame has w_r > 0."
    )
    assert all(isinstance(b, ResidualBlock) for b in blocks)
    for b in blocks:
        assert b.A.shape[1] == rt.nv_a
        assert np.all(np.isfinite(b.A))
        assert np.all(np.isfinite(b.c))


def test_temporal_block_valid(rt):
    from HoloNew.src.test_socp.temporal import build_temporal_block
    # build_temporal_block takes pinocchio q_pin configs (not MuJoCo qpos):
    # solve_single_iteration converts with pin.qpos_mj_to_q_pin before calling them.
    q0 = rt.q_init_full.copy()
    q1 = rt.q_init_full.copy()
    q2 = rt.q_init_full.copy()
    # Perturb an actuated joint (index >= 7) so the acceleration residual is nonzero.
    q1[10] += 0.1
    q2[10] += 0.2
    q_pin0 = rt.pin.qpos_mj_to_q_pin(q0[:36])
    q_pin1 = rt.pin.qpos_mj_to_q_pin(q1[:36])
    q_pin2 = rt.pin.qpos_mj_to_q_pin(q2[:36])
    blocks = build_temporal_block(rt, q_pin0, q_pin1, q_pin2,
                                  lambda_r=1.0, sigma_qddot=rt.sigma_qddot,
                                  sigma_Vdot=rt.sigma_Vdot, dt=rt._dt)
    assert blocks and all(isinstance(b, ResidualBlock) for b in blocks)
    for b in blocks:
        assert b.A.shape[1] == rt.nv_a
        assert np.all(np.isfinite(b.A))
        assert np.all(np.isfinite(b.c))
