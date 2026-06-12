import numpy as np
from HoloNew.src.test_socp.correspondence.transport import transported_points


def test_transported_points_identity_and_offset():
    transforms = {"a": np.eye(4, dtype=np.float64)}
    link_idx = np.array([0, 0])
    offset_local = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64)
    out = transported_points(transforms, link_idx, offset_local, ["a"])
    np.testing.assert_allclose(out, offset_local)


def test_transported_points_applies_rotation_translation():
    T = np.eye(4)
    T[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    T[:3, 3] = [10.0, 0.0, 0.0]
    out = transported_points({"a": T}, np.array([0]), np.array([[1.0, 0.0, 0.0]]), ["a"])
    np.testing.assert_allclose(out[0], [10.0, 1.0, 0.0], atol=1e-9)
