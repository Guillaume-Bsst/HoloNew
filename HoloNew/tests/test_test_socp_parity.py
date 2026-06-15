"""Golden parity snapshot for TestSocpRetargeter.

Records the first 3 frames of the solved root (qpos[:3, :7]) and asserts the
default solve stays numerically stable to 1e-6 as default-off constraint code
is added.

Baseline re-recorded 2026-06-14 after removing the holosoma root-XY scale
(scale_xy_robot=1.0): the base now sits at the RAW grounded pelvis
XY (~0.93, 1.23) instead of the globally-scaled placement (~0.63, 0.83 = raw*0.68),
so the GMR targets agree with the interaction-field references. Only the base XY
shifted; the base Z, orientation, and all joints are unchanged.
"""
import numpy as np

from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter

_G1_QDIM = 36  # G1 base (7) + 29 joints

# Re-baselined after adding the SQP plateau step-break (_iterate_step_tol=0.01),
# a ~3x speedup. The base pose is essentially unchanged; the step-break only stops
# the inner loop's joint chatter ~6 mm earlier (max |qpos diff| ~6 mm vs running
# every iteration). Recorded with rt.retarget().qpos[:3, :7] on sub3_largebox_003.
BASELINE = np.array([
    [ 0.93336654,  1.22630751,  0.80000567, -0.71056267, -0.00636412,
     -0.01614609,  0.70341986],
    [ 0.92929709,  1.23494086,  0.78959036, -0.71344381, -0.05943356,
     -0.06284857,  0.69535289],
    [ 0.92367599,  1.24801238,  0.77287924, -0.71196827, -0.11827157,
     -0.11462996,  0.68262214],
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
