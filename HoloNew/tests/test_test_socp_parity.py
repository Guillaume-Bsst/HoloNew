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

# Re-baselined 2026-06-18 on the current default weights (lambda_pos/rot=0.2,
# lambda_d/x=5, activate_wd/wx_obj=True, lambda_d/x_obj=50, L_interaction=0.20) after
# the modular solve-backend refactor (ProblemSpec + CvxpyBackend). The refactor is
# exact-parity with the prior inline-cvxpy solve (head-to-head agreement ~4e-9); this
# snapshot moved only because the active tracking/interaction weights changed, not the
# solver. Recorded with rt.retarget(max_frames=3).qpos[:3, :7] on sub3_largebox_003
# (the solve is causal, so frames 0-2 are identical to the full-clip solve).
BASELINE = np.array([
    [ 0.93038755,  1.23216773,  0.73741994, -0.71427204, -0.07053026,
     -0.05635916,  0.69402059],
    [ 0.92578760,  1.24062264,  0.72503019, -0.71573577, -0.08781662,
     -0.07600913,  0.68864588],
    [ 0.92206065,  1.24852131,  0.71121712, -0.71692265, -0.10405287,
     -0.09557189,  0.68268655],
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
