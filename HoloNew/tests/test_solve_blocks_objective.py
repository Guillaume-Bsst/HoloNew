import numpy as np
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.src.test_socp.solve.spec import ResidualBlock


def _rt():
    cfg = RetargetingConfig(task_type="robot_only", task_name="sub3_largebox_003",
                            data_format="smplh")
    return TestSocpRetargeter.from_config(cfg)


def test_tracking_blocks_match_cvxpy_value():
    import cvxpy as cp
    from HoloNew.src.test_socp.tracking import build_tracking_terms, build_tracking_blocks
    from HoloNew.src.test_socp.targets import ground_frame_targets
    from HoloNew.src.test_socp.tables import IK_MATCH_TABLE1
    rt = _rt()
    q = rt.q_init_full.copy()
    tg = ground_frame_targets(rt.gmr_ground["pos"][0], rt.gmr_ground["quat"][0], IK_MATCH_TABLE1)
    blocks = build_tracking_blocks(rt, tg, q, lambda_pos=rt.lambda_pos, sigma_p=rt.sigma_p,
                                   lambda_rot=rt.lambda_rot, sigma_rot=rt.sigma_rot)
    assert blocks and all(isinstance(b, ResidualBlock) for b in blocks)
    dqa = cp.Variable(rt.nv_a)
    terms = build_tracking_terms(rt, tg, dqa, q, lambda_pos=rt.lambda_pos, sigma_p=rt.sigma_p,
                                 lambda_rot=rt.lambda_rot, sigma_rot=rt.sigma_rot)
    dqa.value = np.zeros(rt.nv_a)
    cvxpy_val = float(sum(t.value for t in terms))
    block_val = float(sum(np.sum((b.c) ** 2) for b in blocks))  # A·0 + c
    np.testing.assert_allclose(block_val, cvxpy_val, rtol=1e-9, atol=1e-12)
