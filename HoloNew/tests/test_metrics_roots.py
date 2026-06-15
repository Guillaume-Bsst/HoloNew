import numpy as np
from scipy.spatial.transform import Rotation as R

from HoloNew.evaluation.metrics.roots import compute_pose_error


def _poses(T=6, seed=0):
    rng = np.random.RandomState(seed)
    pos = rng.randn(T, 3)
    rot = R.random(T, random_state=rng).as_matrix()
    return pos, rot


def test_identical_zero():
    pos, rot = _poses()
    m = compute_pose_error(pos, rot, pos.copy(), rot.copy())
    assert m["pos_err"] < 1e-12
    assert m["rot_err"] < 1e-9


def test_translation_offset():
    pos, rot = _poses(seed=1)
    d = np.array([0.0, 0.0, 0.5])
    m = compute_pose_error(pos + d, rot, pos, rot)
    assert abs(m["pos_err"] - 0.5) < 1e-9
    assert m["rot_err"] < 1e-9


def test_rotation_offset():
    pos, rot = _poses(seed=2)
    ang = 0.4
    Rd = R.from_rotvec([0.0, 0.0, ang]).as_matrix()
    rot_off = np.einsum("tij,jl->til", rot, Rd)
    m = compute_pose_error(pos, rot_off, pos, rot)
    assert abs(m["rot_err"] - ang) < 1e-6
    assert m["pos_err"] < 1e-12
