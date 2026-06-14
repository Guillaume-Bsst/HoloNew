"""Golden parity snapshot for TestSocpRetargeter.

Records the first 3 frames of the solved root (qpos[:3, :7]) and asserts the
default solve stays numerically stable to 1e-6 as default-off constraint code
is added.

Baseline re-recorded after strengthening the Style pelvis position anchor to
pelvis_anchor_weight=10.0 (2026-06-14).  With the stronger anchor the base
tracks the reference path (mean xy-drift ~0.09 m on sub3_largebox_003 vs
0.243 m at paw=1.0) while Style orientation fidelity is unaffected.
"""
import numpy as np

from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter

_G1_QDIM = 36  # G1 base (7) + 29 joints

# Re-baselined after strengthening the Style pelvis anchor to 10.0.
# Recorded with: rt.retarget().qpos[:3, :7] on sub3_largebox_003 (smplh, robot_only).
BASELINE = np.array([
    [ 0.63471276,  0.83391994,  0.80000567, -0.7108144 , -0.00565931,
     -0.01560232,  0.70318378],
    [ 0.6319457 ,  0.83979543,  0.78959527, -0.71410979, -0.05784619,
     -0.06166082,  0.69490932],
    [ 0.62812757,  0.84882141,  0.77276444, -0.71302065, -0.11616299,
     -0.1131963 ,  0.68212485],
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
