"""Unit tests for prepare/sdf: analytic-shape parity (distance + witness), determinism, cache.

A sphere is used as the ground truth: its signed distance and nearest surface point are analytic,
so the grid can be checked at nodes without involving the (downstream) eval. The only error is the
icosphere's tessellation vs the true sphere — sub-mm at subdivisions=4, hence the few-mm tolerance.
"""
from __future__ import annotations

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from holov2.contracts import SDF, Channel, SdfConfig
from holov2.prepare.sdf.build import SdfBuilder, build_plane_sdf, build_sdf


def _trilinear(sdf: SDF, pts: np.ndarray) -> np.ndarray:
    """Trilinear interpolation of the distance grid at local ``pts`` (test-side reference sampler)."""
    g = (pts - sdf.origin) / sdf.spacing
    i0 = np.floor(g).astype(int)
    t = g - i0
    out = np.zeros(len(pts))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (np.where(dx, t[:, 0], 1 - t[:, 0]) * np.where(dy, t[:, 1], 1 - t[:, 1]) *
                     np.where(dz, t[:, 2], 1 - t[:, 2]))
                out += w * sdf.grid[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
    return out


def _sphere(radius: float = 0.2, subdivisions: int = 4):
    m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)


def _node_coords(sdf: SDF) -> np.ndarray:
    """(Nx, Ny, Nz, 3) local coords of every grid node."""
    nx, ny, nz = sdf.grid.shape
    xs = sdf.origin[0] + sdf.spacing * np.arange(nx)
    ys = sdf.origin[1] + sdf.spacing * np.arange(ny)
    zs = sdf.origin[2] + sdf.spacing * np.arange(nz)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([gx, gy, gz], axis=-1)


def test_sphere_signed_distance_matches_analytic():
    R = 0.2
    sdf = build_sdf(*_sphere(R), spacing=0.04, margin=0.05, name="sphere")
    coords = _node_coords(sdf)
    analytic = np.linalg.norm(coords, axis=-1) - R          # exact signed distance to the sphere
    band = np.abs(analytic) < 0.05
    assert np.abs(sdf.grid[band] - analytic[band]).max() < 2e-3


def test_sphere_witness_on_surface():
    R = 0.2
    sdf = build_sdf(*_sphere(R), spacing=0.04, margin=0.05)
    coords = _node_coords(sdf)
    r = np.linalg.norm(coords, axis=-1)
    band = np.abs(r - R) < 0.05
    w = sdf.witness[band]
    assert np.abs(np.linalg.norm(w, axis=-1) - R).max() < 2e-3       # witnesses lie on the surface
    radial = coords[band] / r[band][..., None]
    wdir = w / np.linalg.norm(w, axis=-1, keepdims=True)
    assert (np.sum(radial * wdir, axis=-1) > 0.99).all()            # along their node's radius


def test_inside_is_negative():
    sdf = build_sdf(*_sphere(0.2), spacing=0.04, margin=0.05)
    coords = _node_coords(sdf)
    centre = np.unravel_index(np.argmin(np.linalg.norm(coords, axis=-1)), sdf.grid.shape)
    assert sdf.grid[centre] < 0


def test_post_init_rejects_witness_shape_mismatch():
    with pytest.raises(ValueError):
        SDF(grid=np.zeros((3, 3, 3), np.float32), witness=np.zeros((3, 3, 2, 3), np.float32),
            origin=np.zeros(3), spacing=0.01, name="x")


def test_determinism():
    v, f = _sphere(0.2)
    a = build_sdf(v, f, 0.04, 0.05)
    b = build_sdf(v, f, 0.04, 0.05)
    assert np.array_equal(a.grid, b.grid)
    assert np.array_equal(a.witness, b.witness)


def test_cache_roundtrip(tmp_path):
    v, f = _sphere(0.2)
    builder = SdfBuilder()
    sdf = builder.build(SdfConfig(spacing=0.04, margin=0.05), v, f, name="sphere")
    p = tmp_path / "sphere.npz"
    builder.save(sdf, p)
    loaded = builder.load(p)
    assert np.array_equal(sdf.grid, loaded.grid)
    assert np.array_equal(sdf.witness, loaded.witness)
    assert np.allclose(sdf.origin, loaded.origin)
    assert sdf.spacing == loaded.spacing
    assert sdf.name == loaded.name == "sphere"


def test_cache_key_sensitivity():
    v, f = _sphere(0.2)
    b = SdfBuilder()
    base = SdfConfig(spacing=0.02, margin=0.05)
    k = b.cache_key(base, v, f)
    assert k == b.cache_key(base, v, f)
    assert k != b.cache_key(SdfConfig(spacing=0.01, margin=0.05), v, f)   # spacing matters
    assert k != b.cache_key(SdfConfig(spacing=0.02, margin=0.08), v, f)   # margin matters
    v2 = v.copy(); v2[0] += 0.1
    assert k != b.cache_key(base, v2, f)                                  # geometry matters


def test_plane_sdf_is_exact_affine():
    sdf = build_plane_sdf([-0.5, -0.5], [0.5, 0.5], spacing=0.1, margin=0.05, name="ground")
    coords = _node_coords(sdf)
    assert np.allclose(sdf.grid, coords[..., 2], atol=1e-6)               # node distance == z
    assert np.allclose(sdf.witness[..., 2], 0.0, atol=1e-6)              # witness on the z=0 plane
    assert np.allclose(sdf.witness[..., :2], coords[..., :2], atol=1e-6)  # witness xy == node xy
    # trilinear sampling at random interior points reproduces z EXACTLY (the field is affine)
    lo = sdf.origin
    hi = sdf.origin + sdf.spacing * (np.array(sdf.grid.shape) - 1)
    p = np.random.default_rng(0).uniform(lo + 1e-6, hi - 1e-6, size=(200, 3))
    assert np.abs(_trilinear(sdf, p) - p[:, 2]).max() < 1e-6


def test_ground_channel_requires_an_sdf():
    # homogeneity: every channel carries an SDF — the flat-ground "None" case is gone
    sdf = build_plane_sdf([-0.3, -0.3], [0.3, 0.3], spacing=0.1, margin=0.05)
    assert Channel("ground", None, sdf).sdf is sdf
    with pytest.raises(TypeError):
        Channel("ground", None)                                          # sdf is now mandatory
