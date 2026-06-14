"""Golden parity snapshot for TestSocpRetargeter.

Records the first 3 frames of the solved root (qpos[:3, :7]) and asserts the
default solve stays numerically stable to 1e-6 as default-off constraint code
is added.

Baseline re-recorded 2026-06-14 after removing the holosoma root-XY scale
(root_xy_scale=1.0 in from_config): the base now sits at the RAW grounded pelvis
XY (~0.93, 1.23) instead of the globally-scaled placement (~0.63, 0.83 = raw*0.68),
so the GMR targets agree with the interaction-field references. Only the base XY
shifted; the base Z, orientation, and all joints are unchanged.
"""
import numpy as np

from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter

_G1_QDIM = 36  # G1 base (7) + 29 joints

# Re-baselined after setting root_xy_scale=1.0 (raw pelvis XY).
# Recorded with: rt.retarget().qpos[:3, :7] on sub3_largebox_003 (smplh, robot_only).
BASELINE = np.array([
    [ 0.93336654,  1.22630751,  0.80000567, -0.71081440, -0.00565931,
     -0.01560232,  0.70318378],
    [ 0.92929775,  1.23494756,  0.78959519, -0.71410985, -0.05784612,
     -0.06166059,  0.69490929],
    [ 0.92367282,  1.24801191,  0.77287837, -0.71302129, -0.11616201,
     -0.11319241,  0.68212499],
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
