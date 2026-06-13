"""Golden parity snapshot for TestSocpRetargeter.

Records the first 3 frames of the solved root (qpos[:3, :7]) and asserts the
default solve stays numerically stable to 1e-6 as default-off constraint code
is added.

Baseline re-recorded after Brick 3 pelvis-relative Style enabled by default
(activate_style=True in TestSocpRetargeterConfig). Validation (2026-06-13):
30-frame robot_only sub3_largebox_003 — style pelvis-relative err 0.60 rad vs
world 0.82 rad (~27 % better), pelvis z [0.562, 0.800] m, fully finite.
"""
import numpy as np

from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter

_G1_QDIM = 36  # G1 base (7) + 29 joints

# Re-baselined after Brick 3 pelvis-relative Style enabled by default.
# Recorded with: rt.retarget().qpos[:3, :7] on sub3_largebox_003 (smplh, robot_only).
BASELINE = np.array([
    [ 0.63471276,  0.83391994,  0.80000567, -0.7108144 , -0.00565931,
     -0.01560232,  0.70318378],
    [ 0.63394258,  0.83555328,  0.79711165, -0.71410965, -0.05784666,
     -0.06166139,  0.69490938],
    [ 0.63208578,  0.83994367,  0.78959769, -0.71301853, -0.11616867,
     -0.11320524,  0.68212461],
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
