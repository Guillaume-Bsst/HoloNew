import numpy as np
from scipy.spatial.transform import Rotation as R

from HoloNew.evaluation.metrics.style import compute_style, style_series


def _random_motion(T=6, K=5, seed=0):
    rng = np.random.RandomState(seed)
    rot = R.random(T * K, random_state=rng).as_matrix().reshape(T, K, 3, 3)
    pos = rng.randn(T, K, 3)
    tracked = np.ones(K, dtype=bool)
    tracked[0] = False  # treat index 0 as the pelvis
    return rot, pos, tracked


def test_identical_is_zero():
    rot, pos, tracked = _random_motion()
    m = compute_style(rot, pos, rot.copy(), pos.copy(), pelvis_idx=0, tracked=tracked)
    assert m["style_orient_err"] < 1e-9
    assert m["style_shape_err"] < 1e-9


def test_orientation_offset_recovered():
    rot, pos, tracked = _random_motion(seed=1)
    # Add a fixed extra rotation to every non-pelvis link's solved orientation.
    ang = 0.3
    Rd = R.from_rotvec([0.0, ang, 0.0]).as_matrix()
    rot_off = rot.copy()
    for k in np.where(tracked)[0]:
        rot_off[:, k] = rot[:, k] @ Rd
    m = compute_style(rot_off, pos, rot, pos, pelvis_idx=0, tracked=tracked)
    assert abs(m["style_orient_err"] - ang) < 1e-6
    assert m["style_shape_err"] < 1e-9  # positions unchanged


def test_heading_invariance():
    """A global yaw of the WHOLE body must leave both style errors ~0."""
    rot, pos, tracked = _random_motion(seed=2)
    yaw = R.from_rotvec([0.0, 0.0, 0.7]).as_matrix()
    rot_y = np.einsum("ij,tkjl->tkil", yaw, rot)      # rotate every link's frame
    pos_y = np.einsum("ij,tkj->tki", yaw, pos)        # and every link's position
    m = compute_style(rot_y, pos_y, rot, pos, pelvis_idx=0, tracked=tracked)
    assert m["style_orient_err"] < 1e-9
    assert m["style_shape_err"] < 1e-9


def test_series_shapes_and_reduce_parity():
    rot, pos, tracked = _random_motion(seed=3)
    T, K = rot.shape[0], rot.shape[1]
    s = style_series(rot, pos, rot.copy(), pos.copy(), pelvis_idx=0, tracked=tracked)
    assert s["orient"].shape == (T, K)   # per-frame, per-link
    assert s["shape"].shape == (T, K)
    # Reducing over the tracked mask reproduces the scoreboard scalars.
    rot2, pos2, _ = _random_motion(seed=4)
    s2 = style_series(rot, pos, rot2, pos2, pelvis_idx=0, tracked=tracked)
    m = compute_style(rot, pos, rot2, pos2, pelvis_idx=0, tracked=tracked)
    msk = tracked.astype(bool)
    np.testing.assert_allclose(m["style_orient_err"], np.mean(s2["orient"][:, msk]))
    np.testing.assert_allclose(m["style_shape_err"], np.mean(s2["shape"][:, msk]))
