"""Unit tests for the shared ``pose_cloud`` op (pure, no data). K=1 must be an exact rigid placement;
K>1 must be the weighted average of the per-part rigid placements (the LBS-on-cloud blend)."""
import numpy as np
from scipy.spatial.transform import Rotation as R

from holov2.contracts import PointCloud
from holov2.targets.interaction import pose_cloud


def _rot(axis, ang):
    return R.from_rotvec(np.asarray(axis, float) * ang).as_matrix()


def test_k1_is_exact_rigid_placement():
    rng = np.random.default_rng(0)
    p = 64
    offsets = rng.standard_normal((p, 1, 3))                          # (P,1,3) one part each
    cloud = PointCloud(parts=np.zeros((p, 1), np.int64), weights=np.ones((p, 1), np.float32),
                       offsets=offsets.astype(np.float32))
    rm, t = _rot([0, 0, 1], 0.7), np.array([1.0, -2.0, 0.5])
    out = pose_cloud(cloud, rm[None], t[None])                       # single part transform
    assert out.shape == (p, 3)
    assert np.allclose(out, offsets[:, 0, :] @ rm.T + t, atol=1e-6)


def test_k2_is_weighted_average_of_rigid_placements():
    rng = np.random.default_rng(1)
    r0, t0 = _rot([1, 0, 0], 0.3), rng.standard_normal(3)
    r1, t1 = _rot([0, 1, 0], -0.8), rng.standard_normal(3)
    o0, o1 = rng.standard_normal(3), rng.standard_normal(3)
    w = np.array([0.25, 0.75])
    cloud = PointCloud(parts=np.array([[0, 1]], np.int64), weights=np.array([w], np.float32),
                       offsets=np.array([[o0, o1]], np.float32))
    out = pose_cloud(cloud, np.stack([r0, r1]), np.stack([t0, t1]))
    a, b = r0 @ o0 + t0, r1 @ o1 + t1                                # each part's rigid placement
    assert np.allclose(out[0], w[0] * a + w[1] * b, atol=1e-6)


def test_partition_of_unity_is_rigid():
    # weights summing to 1 over IDENTICAL placements (same part) => the point moves rigidly.
    rng = np.random.default_rng(2)
    o = rng.standard_normal(3)
    cloud = PointCloud(parts=np.array([[3, 3, 3]], np.int64),
                       weights=np.array([[0.2, 0.3, 0.5]], np.float32),
                       offsets=np.array([[o, o, o]], np.float32))
    part_rot = np.tile(np.eye(3), (5, 1, 1))
    part_rot[3] = _rot([0, 1, 1], 1.1)
    part_pos = rng.standard_normal((5, 3))
    out = pose_cloud(cloud, part_rot, part_pos)
    assert np.allclose(out[0], part_rot[3] @ o + part_pos[3], atol=1e-6)
