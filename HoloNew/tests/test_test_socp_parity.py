"""Golden parity snapshot for TestSocpRetargeter.

Records the first 3 frames of the solved root (qpos[:3, :7]) and asserts the
default solve stays numerically stable to 1e-6 as default-off constraint code
is added.

Baseline re-recorded after Brick 2 temporal regularization W^r is enabled by
default (lambda_r=0.2, sigma_qddot=20.0, sigma_Vdot=20.0 in
TestSocpRetargeterConfig).  The previous pre-W^r values differed; the new
values below are the deliberate post-W^r reference.
"""
import numpy as np

from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter

_G1_QDIM = 36  # G1 base (7) + 29 joints

# Re-baselined after Brick 2 temporal W^r enabled by default.
# Recorded with: rt.retarget().qpos[:3, :7] on sub3_largebox_003 (smplh, robot_only).
BASELINE = np.array([
    [ 0.62890295,  0.81629908,  0.79091496, -0.709926  ,  0.00307521,
     -0.01027691,  0.70419458],
    [ 0.62318512,  0.78709113,  0.77737864, -0.71326872, -0.02731124,
     -0.03936471,  0.69925121],
    [ 0.62056163,  0.76089994,  0.76375037, -0.71567119, -0.05342398,
     -0.06406978,  0.69343759],
])


def test_default_solve_is_stable():
    # The default from_config build must produce a constraint-free solve that
    # matches this frozen baseline. No flag override needed: TestSocpRetargeterConfig
    # defaults all holosoma-style constraint flags OFF, so the plain default config
    # is already constraint-free.
    rt = TestSocpRetargeter.from_config(
        RetargetingConfig(
            task_type="robot_only",
            task_name="sub3_largebox_003",
            data_format="smplh",
        )
    )
    res = rt.retarget()
    assert res.qpos.shape[1] >= _G1_QDIM
    np.testing.assert_allclose(res.qpos[:3, :7], BASELINE, atol=1e-6)
