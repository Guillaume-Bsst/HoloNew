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

# Re-baselined after the robot-Z scale fix: the builder no longer resolves
# scale_z_robot=None -> smpl_scale; it passes None through to scale() so the robot Z
# uses GMR's native morphological base (HUMAN_SCALE_TABLE[pelvis]*ratio), like gmr_socp.
# The pelvis Z dropped ~4 cm and the orientation/XY rebalanced. Recorded with
# rt.retarget().qpos[:3, :7] on sub3_largebox_003 (solve is deterministic to 0.0).
# See test_scale_z_gmr_parity for the underlying z-placement assertion.
BASELINE = np.array([
    [ 0.93314958,  1.24469752,  0.76197750, -0.70748595, -0.07170524,
     -0.07309210,  0.69927072],
    [ 0.92763404,  1.25388192,  0.74687360, -0.70932002, -0.09050483,
     -0.09135339,  0.69305738],
    [ 0.92314360,  1.26238896,  0.73055882, -0.71104174, -0.10792174,
     -0.10957625,  0.68612360],
])


def test_default_solve_is_stable():
    # The default from_config build must produce a constraint-free solve that
    # matches this frozen baseline. No flag override needed: TestSocpRetargeterConfig
    # defaults all holosoma-style constraint flags OFF, so the plain default config
    # is already constraint-free (only the GMR pos/rot tracking weights are active;
    # the re-tuned lambda_* of the gated bricks do not affect this solve).
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
