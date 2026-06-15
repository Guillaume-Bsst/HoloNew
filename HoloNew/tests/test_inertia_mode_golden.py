"""Pin the inertia-mode output so future changes are intentional."""
from pathlib import Path
import numpy as np
import numpy.testing as npt
import pytest
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.tests.paper_placement import paper_placement_config

# NOTE: golden recorded under the old inertia_mode preset. After the config flatten the
# paper-placement config is explicit (paper_placement_config); regenerate this golden if
# it drifts.
_GOLD = Path(__file__).parent / "golden" / "inertia_mode_qpos.npz"


@pytest.mark.skipif(not _GOLD.exists(), reason="inertia-mode golden not present")
def test_inertia_mode_matches_golden():
    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="object_interaction", task_name="sub3_largebox_003",
        data_format="smplh", retargeter=paper_placement_config()))
    if rt.correspondence is None:
        pytest.skip("assets not present")
    res = rt.retarget(max_frames=30)
    gold = np.load(_GOLD)["qpos"]
    npt.assert_allclose(res.qpos, gold, atol=1e-6)
