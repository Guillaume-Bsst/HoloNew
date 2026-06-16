"""W^smooth port (native Holosoma smoothness term): step toward the previous frame."""
import numpy as np

from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter


def _rt(**kw):
    return TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh",
        retargeter=TestSocpRetargeterConfig(**kw)))


def test_smooth_runs_finite_and_is_active():
    """W^smooth assembles, the solve stays finite, and the term measurably affects
    the joints (wired + active). Correctness vs the native Holosoma smoothness term
    is covered by the Holosoma-mode parity test, not by an A/B on top of GMR tracking
    (where a regularizer competing with tracking can increase frame-to-frame motion)."""
    q_off = _rt(activate_smooth=False).retarget(max_frames=12).qpos
    q_on = _rt(activate_smooth=True, lambda_smooth=0.2).retarget(max_frames=12).qpos
    assert np.all(np.isfinite(q_on)), "non-finite qpos with W^smooth on"
    assert not np.allclose(q_off[:, 7:], q_on[:, 7:], atol=1e-6), (
        "W^smooth had no effect on the joints — not wired/active")
