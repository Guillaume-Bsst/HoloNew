"""Reusable residual ops: unit values + finite-difference (FD) of the linearised model each op
encodes. The builders' correctness reduces to these ops + a contraction, so they carry the FD load."""
import numpy as np

from src.solve.terms._ops import (GeoField, dist_jac, geo_chain, quat_to_rot, scatter_obj,
                                   se3_log_world, so3_log, world_normal)


def _rand_rot(rng):
    a = rng.standard_normal(3); a /= np.linalg.norm(a)
    th = rng.uniform(0.2, 2.5)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def test_world_normal_single_and_batched():
    rng = np.random.default_rng(0)
    R = _rand_rot(rng)
    n = rng.standard_normal((5, 3))
    got = world_normal(R, n)
    assert got.shape == (5, 3)
    assert np.allclose(got, n @ R.T)                       # n_world[m] = R @ n[m]
    Rb = np.stack([_rand_rot(rng) for _ in range(5)])       # per-row rotation
    gb = world_normal(Rb, n)
    assert np.allclose(gb, np.einsum("mij,mj->mi", Rb, n))
    assert np.allclose(world_normal(np.eye(3), n), n)       # identity (ground channel)


def test_dist_jac_matches_directional_derivative():
    # d(v) = nᵀ(p0 + J v - w) is linear in v -> dist_jac(n,J) @ v == d(v) - d(0) exactly.
    rng = np.random.default_rng(1)
    M, nv = 4, 9
    n = rng.standard_normal((M, 3)); J = rng.standard_normal((M, 3, nv))
    A = dist_jac(n, J)
    assert A.shape == (M, nv)
    v = rng.standard_normal(nv)
    d_lin = np.einsum("mi,mij,j->m", n, J, v)               # exact directional derivative
    assert np.allclose(A @ v, d_lin)


def test_geo_chain_is_the_same_contraction():
    rng = np.random.default_rng(2)
    g = rng.standard_normal((3, 3)); J = rng.standard_normal((3, 3, 6))
    assert np.allclose(geo_chain(g, J), dist_jac(g, J))


def test_so3_log_recovers_axis_angle_and_jacobian():
    rng = np.random.default_rng(3)
    R_ref = _rand_rot(rng)
    u = rng.standard_normal(3); u /= np.linalg.norm(u)
    for th in (1e-4, 0.3, 3.0, np.pi - 1e-3):
        K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
        E = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)   # exp(th [u]x), world-left
        R_cur = E @ R_ref                                            # R_cur Rrefᵀ = E
        e = so3_log(R_ref[None], R_cur[None])[0]
        assert np.allclose(e, th * u, atol=1e-6)
    # Jacobian convention: d/dα so3_log(R_ref, exp(α[w]x) R_ref) |0 = w  (world angular vel = jac_rot v)
    w = rng.standard_normal(3)
    eps = 1e-6
    Kw = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    Rp = (np.eye(3) + eps * Kw) @ R_ref
    de = so3_log(R_ref[None], Rp[None])[0] / eps
    assert np.allclose(de, w, atol=1e-4)


def test_se3_log_world_blocks():
    rng = np.random.default_rng(4)
    R_ref = np.stack([_rand_rot(rng), _rand_rot(rng)])
    R_cur = np.stack([_rand_rot(rng), _rand_rot(rng)])
    p_ref = rng.standard_normal((2, 3)); p_cur = rng.standard_normal((2, 3))
    e = se3_log_world(R_ref, p_ref, R_cur, p_cur)
    assert e.shape == (2, 6)
    assert np.allclose(e[:, :3], p_cur - p_ref)
    assert np.allclose(e[:, 3:], so3_log(R_ref, R_cur))
    # identical poses -> zero residual
    z = se3_log_world(R_ref, p_ref, R_ref, p_ref)
    assert np.allclose(z, 0.0)


def test_quat_to_rot_known():
    R = quat_to_rot(np.array([[1.0, 0.0, 0.0, 0.0]]))       # identity wxyz
    assert np.allclose(R[0], np.eye(3))
    R90 = quat_to_rot(np.array([[np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]]))  # +90° about z
    assert np.allclose(R90[0] @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)


def test_scatter_obj_places_block():
    rng = np.random.default_rng(5)
    blk = rng.standard_normal((3, 6))
    A_obj = scatter_obj(blk, object_idx=1, n_obj=3)
    assert A_obj.shape == (3, 18)
    assert np.allclose(A_obj[:, 6:12], blk)
    assert np.allclose(A_obj[:, :6], 0.0) and np.allclose(A_obj[:, 12:], 0.0)


def test_geofield_shapes_and_ground_identity():
    pts = np.zeros((4, 3), np.float32)
    gf = GeoField(tables=(None, None), rot=np.stack([np.eye(3), np.eye(3)]),
                  pos=np.zeros((2, 3)), object_idx=(-1, 0))
    assert gf.rot.shape == (2, 3, 3) and gf.object_idx == (-1, 0)
    assert np.allclose(world_normal(gf.rot[0], pts.astype(float)), pts)  # ground frame = I
