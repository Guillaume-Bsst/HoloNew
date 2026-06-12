import numpy as np

from HoloNew.src.holosoma.preprocess import scale_object_poses_to_center


def _poses(T=5):
    # layout [qw, qx, qy, qz, x, y, z]
    p = np.zeros((T, 7), np.float64)
    p[:, 0] = 1.0  # identity quat
    p[:, 4] = np.linspace(2.0, 4.0, T)   # x
    p[:, 5] = np.linspace(-1.0, -3.0, T)  # y
    p[:, 6] = np.linspace(0.8, 1.3, T)    # z
    return p


def test_xy_scaled_toward_center():
    p = _poses()
    scale = 0.5
    out = scale_object_poses_to_center(p, scale)
    np.testing.assert_allclose(out[:, 4:6], p[:, 4:6] * scale)


def test_z_keeps_frame0_and_scales_deviation():
    p = _poses()
    scale = 0.5
    out = scale_object_poses_to_center(p, scale)
    z0 = p[0, 6]
    np.testing.assert_allclose(out[0, 6], z0)                       # frame 0 unchanged
    np.testing.assert_allclose(out[:, 6], z0 + (p[:, 6] - z0) * scale)


def test_quaternion_untouched():
    p = _poses()
    out = scale_object_poses_to_center(p, 0.3)
    np.testing.assert_allclose(out[:, :4], p[:, :4])


def test_input_not_mutated():
    p = _poses()
    before = p.copy()
    scale_object_poses_to_center(p, 0.3)
    np.testing.assert_array_equal(p, before)
