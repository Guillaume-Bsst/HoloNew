import numpy as np
from HoloNew.src.retarget_result import RetargetResult


def test_result_holds_optional_interaction_fields():
    r = RetargetResult(qpos=np.zeros((2, 36)))
    assert r.human_probe_pts is None
    assert r.g1_transport_pts is None
    r2 = RetargetResult(qpos=np.zeros((2, 36)),
                        human_probe_pts=np.zeros((2, 5, 3)),
                        g1_transport_pts=np.zeros((2, 4, 3)))
    assert r2.human_probe_pts.shape == (2, 5, 3)
