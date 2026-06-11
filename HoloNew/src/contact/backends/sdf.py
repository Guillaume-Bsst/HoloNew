# src/test_pipe_retargeting/test_pipe_retargeting/fields/backends/sdf.py
"""Precomputed signed-distance field of a rigid object, in the object-local frame.

The grid is filled by the same Coal + trimesh.contains pipeline as coal.surface_field,
so it equals Coal at grid nodes by construction; only trilinear interpolation between nodes
approximates. Sampling returns distance + analytic gradient (the contact direction) with no
Coal query, for cheap repeated evaluation inside an optimisation loop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contact_field import ContactField
from .coal import build_bvh, surface_field


@dataclass(frozen=True)
class ObjectSDF:
    origin: np.ndarray   # (3,)        object-local coords of node (0, 0, 0)
    spacing: float       #             isotropic voxel size (metres)
    dims: np.ndarray     # (3,) int    nx, ny, nz
    data: np.ndarray     # (nx,ny,nz) float32  signed distance at nodes
    witness: np.ndarray | None = None         # (nx,ny,nz,3) float32  nearest surface point; None for legacy grids
    active_grid: np.ndarray | None = None     # (nx,ny,nz) bool      Coal active flag per node; None for legacy grids

    def sample(self, pts_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Trilinear distance + normalised analytic gradient at object-local points.

        Returns (dist (N,), grad (N, 3)). Out-of-grid points get dist=+inf, grad=0."""
        pts = np.asarray(pts_local, dtype=np.float64)
        n = len(pts)
        g = (pts - self.origin) / self.spacing          # continuous grid index
        i0 = np.floor(g).astype(np.int64)
        t = g - i0                                       # in [0, 1) per axis
        # Both i0 and i0+1 must be valid nodes -> i0 in [0, dim-2].
        in_grid = np.all((i0 >= 0) & (i0 < self.dims - 1), axis=1)

        dist = np.full(n, np.inf, dtype=np.float64)
        grad = np.zeros((n, 3), dtype=np.float64)
        if in_grid.any():
            ii, tt = i0[in_grid], t[in_grid]
            c = self._corners(ii)                        # (m, 2, 2, 2)
            x0, x1 = 1 - tt[:, 0], tt[:, 0]
            y0, y1 = 1 - tt[:, 1], tt[:, 1]
            z0, z1 = 1 - tt[:, 2], tt[:, 2]

            v = (c[:, 0, 0, 0] * x0 * y0 * z0 + c[:, 1, 0, 0] * x1 * y0 * z0 +
                 c[:, 0, 1, 0] * x0 * y1 * z0 + c[:, 1, 1, 0] * x1 * y1 * z0 +
                 c[:, 0, 0, 1] * x0 * y0 * z1 + c[:, 1, 0, 1] * x1 * y0 * z1 +
                 c[:, 0, 1, 1] * x0 * y1 * z1 + c[:, 1, 1, 1] * x1 * y1 * z1)

            dvdx = ((c[:, 1, 0, 0] - c[:, 0, 0, 0]) * y0 * z0 +
                    (c[:, 1, 1, 0] - c[:, 0, 1, 0]) * y1 * z0 +
                    (c[:, 1, 0, 1] - c[:, 0, 0, 1]) * y0 * z1 +
                    (c[:, 1, 1, 1] - c[:, 0, 1, 1]) * y1 * z1) / self.spacing
            dvdy = ((c[:, 0, 1, 0] - c[:, 0, 0, 0]) * x0 * z0 +
                    (c[:, 1, 1, 0] - c[:, 1, 0, 0]) * x1 * z0 +
                    (c[:, 0, 1, 1] - c[:, 0, 0, 1]) * x0 * z1 +
                    (c[:, 1, 1, 1] - c[:, 1, 0, 1]) * x1 * z1) / self.spacing
            dvdz = ((c[:, 0, 0, 1] - c[:, 0, 0, 0]) * x0 * y0 +
                    (c[:, 1, 0, 1] - c[:, 1, 0, 0]) * x1 * y0 +
                    (c[:, 0, 1, 1] - c[:, 0, 1, 0]) * x0 * y1 +
                    (c[:, 1, 1, 1] - c[:, 1, 1, 0]) * x1 * y1) / self.spacing

            g_raw = np.stack([dvdx, dvdy, dvdz], axis=1)
            norm = np.linalg.norm(g_raw, axis=1, keepdims=True)
            # Guard the denominator so flat regions (zero gradient, e.g. interior clamped
            # to +margin) don't raise a divide-by-zero; they keep grad = 0.
            safe = np.where(norm > 1e-9, norm, 1.0)
            dist[in_grid] = v
            grad[in_grid] = np.where(norm > 1e-9, g_raw / safe, 0.0)
        return dist, grad

    def _corners(self, i0: np.ndarray) -> np.ndarray:
        """(m, 2, 2, 2) array of the 8 node values around each cell index i0."""
        ix, iy, iz = i0[:, 0], i0[:, 1], i0[:, 2]
        d = self.data
        c = np.empty((len(i0), 2, 2, 2), dtype=np.float64)
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    c[:, di, dj, dk] = d[ix + di, iy + dj, iz + dk]
        return c

    def query(self, pts_local: np.ndarray, margin: float) -> ContactField:
        """Coal's 4-channel ContactField reconstructed from the cached grid.

        Trilinearly interpolates distance, witness, and the active flag from the
        cached grids; direction is derived from the interpolated witness as
        (probe - witness)/||.|| (a unit vector everywhere, consistent with how
        sdf_surface_field derives direction from the gradient). Exact at nodes;
        only approximates between them. Requires a witness-carrying grid
        (build_object_field)."""
        if self.witness is None:
            raise ValueError("query() needs a witness grid; build with build_object_field")
        pts = np.asarray(pts_local, dtype=np.float64)
        n = len(pts)
        g = (pts - self.origin) / self.spacing
        i0 = np.floor(g).astype(np.int64)
        t = g - i0
        in_grid = np.all((i0 >= 0) & (i0 < self.dims - 1), axis=1)

        dist = np.full(n, float(margin), dtype=np.float64)
        witness = np.zeros((n, 3), dtype=np.float64)
        active = np.zeros(n, dtype=bool)

        if in_grid.any():
            ii, tt = i0[in_grid], t[in_grid]
            dist[in_grid] = self._trilinear(self.data, ii, tt)
            for k in range(3):
                witness[in_grid, k] = self._trilinear(self.witness[..., k], ii, tt)
            # active_grid: stored from Coal's own float64 comparison at build time,
            # so trilinear interpolation is exact at nodes (avoids float32 rounding
            # of the clamped distance grid misidentifying margin-boundary nodes).
            if self.active_grid is not None:
                active_interp = self._trilinear(self.active_grid.astype(np.float64), ii, tt)
                active[in_grid] = active_interp >= 0.5
            else:
                active[in_grid] = dist[in_grid] < margin

        # Direction is derived from the interpolated witness, giving a true unit
        # vector everywhere (interpolating a raw direction grid would yield shrunken,
        # non-unit vectors near surface-normal discontinuities like box corners). The
        # 1e-6 guard absorbs the float32 round-trip residual at on-surface nodes
        # (probe == witness), where Coal reports a zero direction.
        d = pts - witness
        norm = np.linalg.norm(d, axis=1, keepdims=True)
        direction = np.where(norm > 1e-6, d / norm, 0.0)

        out_dist = np.where(active, dist, float(margin)).astype(np.float32)
        out_dir = np.where(active[:, None], direction, 0.0).astype(np.float32)
        out_wit = np.where(active[:, None], witness, 0.0).astype(np.float32)
        return ContactField(distance=out_dist, direction=out_dir,
                            witness=out_wit, active=active)

    @staticmethod
    def _trilinear(grid: np.ndarray, i0: np.ndarray, t: np.ndarray) -> np.ndarray:
        ix, iy, iz = i0[:, 0], i0[:, 1], i0[:, 2]
        x0, x1 = 1 - t[:, 0], t[:, 0]
        y0, y1 = 1 - t[:, 1], t[:, 1]
        z0, z1 = 1 - t[:, 2], t[:, 2]
        return (grid[ix, iy, iz] * x0 * y0 * z0 + grid[ix + 1, iy, iz] * x1 * y0 * z0 +
                grid[ix, iy + 1, iz] * x0 * y1 * z0 + grid[ix + 1, iy + 1, iz] * x1 * y1 * z0 +
                grid[ix, iy, iz + 1] * x0 * y0 * z1 + grid[ix + 1, iy, iz + 1] * x1 * y0 * z1 +
                grid[ix, iy + 1, iz + 1] * x0 * y1 * z1 + grid[ix + 1, iy + 1, iz + 1] * x1 * y1 * z1)


def build_object_sdf(mesh, margin: float, resolution: float = 0.01) -> ObjectSDF:
    """Voxel SDF over the object's AABB padded by (margin + 2*resolution), object-local frame.

    Nodes are filled with the exact signed distance from contact_field.surface_field, which
    clamps values >= margin to +margin (far/outside nodes) and keeps true negatives inside."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    pad = float(margin) + 2.0 * float(resolution)
    lo = verts.min(0) - pad
    hi = verts.max(0) + pad
    dims = (np.ceil((hi - lo) / resolution).astype(np.int64) + 1)
    xs = lo[0] + resolution * np.arange(dims[0])
    ys = lo[1] + resolution * np.arange(dims[1])
    zs = lo[2] + resolution * np.arange(dims[2])
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    nodes = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    bvh = build_bvh(verts, np.asarray(mesh.faces, dtype=np.int64))
    f = surface_field(nodes, bvh, mesh, float(margin))
    data = f.distance.reshape(tuple(int(d) for d in dims)).astype(np.float32)
    return ObjectSDF(origin=lo.astype(np.float64), spacing=float(resolution),
                     dims=dims, data=data)


def build_object_field(mesh, margin: float, resolution: float = 0.01) -> ObjectSDF:
    """Like build_object_sdf, but also caches Coal's witness (proximal point) and active flag per node."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    pad = float(margin) + 2.0 * float(resolution)
    lo = verts.min(0) - pad
    hi = verts.max(0) + pad
    dims = (np.ceil((hi - lo) / resolution).astype(np.int64) + 1)
    xs = lo[0] + resolution * np.arange(dims[0])
    ys = lo[1] + resolution * np.arange(dims[1])
    zs = lo[2] + resolution * np.arange(dims[2])
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    nodes = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    bvh = build_bvh(verts, np.asarray(mesh.faces, dtype=np.int64))
    f = surface_field(nodes, bvh, mesh, float(margin))
    shape = tuple(int(d) for d in dims)
    return ObjectSDF(
        origin=lo.astype(np.float64), spacing=float(resolution), dims=dims,
        data=f.distance.reshape(shape).astype(np.float32),
        witness=f.witness.reshape(shape + (3,)).astype(np.float32),
        active_grid=f.active.reshape(shape),
    )


def band_points(sdf: ObjectSDF, margin: float) -> tuple[np.ndarray, np.ndarray]:
    """Grid nodes within the |signed distance| < margin band, for visualising the SDF.

    Returns (pts (M, 3) float32 in the object-local frame, dist (M,) signed distance).
    Only the near-surface shell is kept; nodes beyond margin are clamped flat at +margin
    and carry no useful gradient/colour, so they are dropped."""
    nx, ny, nz = (int(d) for d in sdf.dims)
    xs = sdf.origin[0] + sdf.spacing * np.arange(nx)
    ys = sdf.origin[1] + sdf.spacing * np.arange(ny)
    zs = sdf.origin[2] + sdf.spacing * np.arange(nz)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    dist = sdf.data.reshape(-1)
    band = np.abs(dist) < margin
    return pts[band].astype(np.float32), dist[band]


def sdf_surface_field(probe_pts: np.ndarray, sdf: ObjectSDF, margin: float) -> ContactField:
    """ContactField from an ObjectSDF, drop-in for contact_field.surface_field.

    Same contract: signed distance (clamped to +margin when inactive), direction in the
    Coal convention (probe - witness)/||.||.

    If the SDF carries a witness grid (build_object_field), we use trilinear
    interpolation of the witnesses; this is stable and avoids the aberrant points
    that an analytic gradient would produce near medial axes or sharp features.
    Otherwise, we fall back to gradient-based reconstruction."""
    pts = np.asarray(probe_pts, dtype=np.float64)
    if sdf.witness is not None:
        return sdf.query(pts, margin)

    # Fallback for legacy distance-only grids:
    dist, grad = sdf.sample(pts)
    finite = np.isfinite(dist)
    dist_f = np.where(finite, dist, float(margin))
    active = finite & (dist_f < margin)

    direction = np.sign(dist_f)[:, None] * grad
    witness = pts - dist_f[:, None] * grad

    out_dist = np.where(active, dist_f, float(margin)).astype(np.float32)
    out_dir = np.where(active[:, None], direction, 0.0).astype(np.float32)
    out_witness = np.where(active[:, None], witness, 0.0).astype(np.float32)
    return ContactField(distance=out_dist, direction=out_dir,
                        witness=out_witness, active=active)


def sdf_surface_distance_torch(probe_pts, sdf: ObjectSDF, margin: float):
    """Differentiable signed distance from an ObjectSDF: value from the field, gradient =
    contact direction (sign(dist)*grad), zeroed for inactive probes. Mirrors
    contact_field.surface_distance_torch so it is a swap-in for any torch consumer."""
    import torch

    class _SDFDistance(torch.autograd.Function):
        @staticmethod
        def forward(ctx, pts):
            f = sdf_surface_field(pts.detach().cpu().numpy(), sdf, margin)
            ctx.save_for_backward(
                torch.from_numpy(np.ascontiguousarray(f.direction)).to(pts),
                torch.from_numpy(np.ascontiguousarray(f.active)).to(pts.device),
            )
            return torch.from_numpy(np.ascontiguousarray(f.distance)).to(pts)

        @staticmethod
        def backward(ctx, grad_out):
            direction, active = ctx.saved_tensors
            grad_p = grad_out[:, None] * direction
            return grad_p * active[:, None].to(grad_p)

    return _SDFDistance.apply(probe_pts)


def save_object_sdf(sdf: ObjectSDF, path) -> None:
    """Serialise an ObjectSDF to a compressed .npz (compute once, cache on disk).

    Saves witness and active_grid when present; legacy distance-only files remain
    loadable (missing keys are restored as None)."""
    arrays = dict(origin=sdf.origin, spacing=np.float64(sdf.spacing),
                  dims=sdf.dims, data=sdf.data)
    if sdf.witness is not None:
        arrays["witness"] = sdf.witness
    if sdf.active_grid is not None:
        arrays["active_grid"] = sdf.active_grid
    np.savez_compressed(str(path), **arrays)


def load_object_sdf(path) -> ObjectSDF:
    d = np.load(str(path))
    return ObjectSDF(
        origin=d["origin"], spacing=float(d["spacing"]),
        dims=d["dims"], data=d["data"],
        witness=d["witness"] if "witness" in d.files else None,
        active_grid=d["active_grid"] if "active_grid" in d.files else None,
    )
