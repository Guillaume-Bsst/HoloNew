"""Golden parity snapshot for TestSocpRetargeter.

Records the first 3 frames of the solved root (qpos[:3, :7]) and asserts the
default solve stays numerically stable to 1e-6 as default-off constraint code
is added.

Baseline re-recorded after Brick 0 pinocchio kinematics migration (tangent-space
dqa, pin.integrate replacing the MuJoCo finite-difference kinematics).  The
previous MuJoCo-based values differed by ~2.4e-5; the new values below are the
deliberate post-migration reference.
"""
import numpy as np

from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter

_G1_QDIM = 36  # G1 base (7) + 29 joints

# Re-baselined after Brick 0 pinocchio kinematics migration.
# Recorded with: rt.retarget().qpos[:3, :7] on sub3_largebox_003 (smplh, robot_only).
BASELINE = np.array([
    [ 0.63502393,  0.84425846,  0.76797592, -0.70690993, -0.09206585,
     -0.08866832,  0.69565808],
    [ 0.63158034,  0.84970298,  0.75353179, -0.70796383, -0.11332746,
     -0.10957337,  0.68843139],
    [ 0.62874597,  0.85487409,  0.73782876, -0.70860376, -0.1329542 ,
     -0.13040042,  0.6805877 ],
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
