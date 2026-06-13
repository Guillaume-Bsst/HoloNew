# tests/test_pin_model_build.py
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.src.test_socp.pin_model import PinModel


def test_pin_model_builds_freeflyer_g1():
    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    pm = PinModel(rt.task_constants.ROBOT_URDF_FILE)
    assert pm.model.nq == 36
    assert pm.model.nv == 35
    assert "left_hip_pitch_joint" in pm.joint_names
