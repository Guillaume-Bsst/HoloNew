import numpy as np, numpy.testing as npt
from HoloNew.src.retarget_result import RetargetResult

def test_retarget_returns_result_matching_golden(golden_qpos):
    from HoloNew.examples.robot_retarget import run_headless
    result = run_headless(
        data_path="demo_data/OMOMO_new", task_type="robot_only",
        task_name="sub3_largebox_003", data_format="smplh",
    )
    assert isinstance(result, RetargetResult)
    npt.assert_allclose(result.qpos, golden_qpos, atol=1e-9)
    assert "mapped" in result.stages and "in_object" in result.stages
