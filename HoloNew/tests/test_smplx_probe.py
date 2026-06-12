import numpy as np
from HoloNew.src.test_socp.correspondence.human_body import PointCloudCache
from HoloNew.src.test_socp.contact.smplx_field import ProbeFrame


def test_probeframe_carries_points_and_field():
    pts = np.zeros((5, 3), np.float32)

    class _F:
        distance = np.zeros(5)
    pf = ProbeFrame(points=pts, field=_F())
    assert pf.points.shape == (5, 3)
    assert pf.field.distance.shape == (5,)
