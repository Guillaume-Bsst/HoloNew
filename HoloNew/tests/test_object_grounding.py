"""ground_object_pose: a single constant per-clip z-shift drops the object so its lowest
surface point over the clip rests on z=0 (HODome only). Non-HODome / no-surface = no-op."""
import numpy as np

from HoloNew.src.test_socp.movable import ground_object_pose


def _identity_poses(T, z):
    p = np.zeros((T, 7))
    p[:, 0] = 1.0          # qw (identity rotation)
    p[:, 6] = z            # translate z
    return p


def test_hodome_grounds_lowest_surface_to_zero():
    surface = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    poses = _identity_poses(3, 0.5)        # lowest surface world z = 0.5
    grounded, shift = ground_object_pose(poses, surface, "hodome")
    assert np.isclose(shift, -0.5)
    # lowest surface point over the clip now rests on z=0 (identity rot => world = local + t)
    zmin = min((surface + grounded[t, 4:7])[:, 2].min() for t in range(3))
    assert np.isclose(zmin, 0.0, atol=1e-9)
    # only z is shifted; rotation + xy untouched
    np.testing.assert_array_equal(grounded[:, :6], poses[:, :6])


def test_non_hodome_is_noop():
    surface = np.array([[0, 0, 0], [0, 0, 1]], float)
    poses = _identity_poses(2, 0.5)
    grounded, shift = ground_object_pose(poses, surface, "omomo")
    assert shift == 0.0
    np.testing.assert_array_equal(grounded, poses)


def test_no_surface_is_noop():
    poses = _identity_poses(2, 0.5)
    grounded, shift = ground_object_pose(poses, None, "hodome")
    assert shift == 0.0
    np.testing.assert_array_equal(grounded, poses)
