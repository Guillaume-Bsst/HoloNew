"""W^nominal port (native Holosoma nominal-pose tracking with exp-decaying weight)."""
import numpy as np

from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter


def _rt(**kw):
    return TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh",
        retargeter=TestSocpRetargeterConfig(**kw)))


def test_nominal_runs_and_pulls_toward_nominal_pose():
    # NOMINAL_TRACKING_INDICES = 0..18 -> actuated joints 0..18 -> qpos 7..25.
    # Default nominal pose = q_init's joints.
    rt_on = _rt(activate_nominal=True, lambda_nominal=50.0)
    q_off = _rt(activate_nominal=False).retarget(max_frames=10).qpos
    q_on = rt_on.retarget(max_frames=10).qpos
    assert np.all(np.isfinite(q_on)), "non-finite qpos with W^nominal on"
    qpos_sel = np.arange(7, 7 + 19)
    init = rt_on.q_init_full[qpos_sel]
    dev_off = float(np.abs(q_off[:, qpos_sel] - init).mean())
    dev_on = float(np.abs(q_on[:, qpos_sel] - init).mean())
    assert dev_on < dev_off, (
        f"W^nominal did not pull joints toward the nominal pose: on={dev_on:.4f} off={dev_off:.4f}")
