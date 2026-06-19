import numpy as np
import pytest
from pathlib import Path
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.src.data_loaders.facade import normalize_dataset_cfg

_NAME = "subject01_baseball"


def _hodome_seq_present() -> bool:
    try:
        from HoloNew.src.paths import get_path
        r = Path(get_path("hodome"))
        return (r / "smplx" / f"{_NAME}.npz").exists() \
            and (r / "object" / f"{_NAME}.npz").exists() \
            and (r / "scaned_object" / "baseball.tar").exists()
    except Exception:
        return False


@pytest.mark.skipif(not _hodome_seq_present(), reason="HODome baseball sequence not present")
def test_hodome_object_scene_end_to_end():
    cfg = RetargetingConfig(
        dataset="hodome", motion_name=_NAME, task_type="object_interaction",
        retargeter=TestSocpRetargeterConfig(activate_obj_non_penetration=True,
                                            load_object_scene=True))
    normalize_dataset_cfg(cfg)                       # façade: paths, object_name=baseball
    assert cfg.task_config.object_name == "baseball"
    rt = TestSocpRetargeter.from_config(cfg)
    assert rt.object_name == "baseball"
    assert rt.has_dynamic_object is True             # scene swap added the object free joint
    res = rt.retarget(max_frames=3)
    assert np.all(np.isfinite(res.qpos))
    assert res.qpos.shape[1] >= 7 + rt.nq_a          # trailing object DOFs present
