"""Golden parity snapshot for TestSocpRetargeter.

Records the first 3 frames of the solved root (qpos[:3, :7]) and asserts the
default solve stays numerically stable to 1e-6 as default-off constraint code
is added.
"""
import numpy as np

from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter

_G1_QDIM = 36  # G1 base (7) + 29 joints

# Frozen from the constraint-free solve.  Must not change when default-off
# constraint flags are introduced.
BASELINE = np.array([
    [ 0.63502428,  0.84425885,  0.76797607, -0.7069117 , -0.09206482,
     -0.08866697,  0.69565658],
    [ 0.63158063,  0.84970386,  0.75353206, -0.70796568, -0.11332616,
     -0.10957183,  0.68842995],
    [ 0.62874468,  0.85488169,  0.73783928, -0.70858303, -0.13297777,
     -0.13041967,  0.68060099],
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
