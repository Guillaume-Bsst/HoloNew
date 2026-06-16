"""W^qdiag port (native Holosoma Q_diag regularizer on the absolute joint config)."""
import numpy as np

from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter


def _rt(**kw):
    return TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh",
        retargeter=TestSocpRetargeterConfig(**kw)))


def test_qdiag_runs_and_pulls_costed_joints_toward_zero():
    # MANUAL_COST = {19: 0.2, 20: 0.2} -> actuated joints 19, 20 -> qpos 26, 27.
    q_off = _rt(activate_qdiag=False).retarget(max_frames=10).qpos
    q_on = _rt(activate_qdiag=True, lambda_qdiag=50.0).retarget(max_frames=10).qpos
    assert np.all(np.isfinite(q_on)), "non-finite qpos with W^qdiag on"
    off_mag = float(np.abs(q_off[:, [26, 27]]).mean())
    on_mag = float(np.abs(q_on[:, [26, 27]]).mean())
    assert on_mag < off_mag, (
        f"W^qdiag did not pull the costed joints toward 0: on={on_mag:.4f} off={off_mag:.4f}")
