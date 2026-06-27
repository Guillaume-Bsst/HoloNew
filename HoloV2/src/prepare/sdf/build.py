"""prepare/sdf — builds the signed-distance + witness grid of a surface: an object/terrain MESH
(``build_sdf``) or the flat GROUND plane (``build_plane_sdf``, analytic, no mesh).

The mesh SDF is offline, geometry-keyed, cached once (``SdfBuilder``): a subject-/robot-free asset
shared by every sequence touching the same mesh. Mesh-agnostic — ONE builder for objects AND terrain.
The flat ground is ALSO an SDF (not a special analytic case): a plane is affine, so a tiny grid
reproduces it exactly, keeping every eval channel homogeneous (no flat-ground branch).

Build pipeline (trimesh, no Coal — this runs once and is cached, so per-probe speed is irrelevant):
fill each grid node with the signed distance to the surface. ``trimesh.proximity.closest_point``
returns the nearest surface point (the WITNESS, stored first-class) AND the unsigned distance in one
vectorised call; ``mesh.contains`` gives the inside/outside sign. The grid stores the TRUE signed
distance everywhere (no clamp): the activation band (``dist < margin``) is applied later, at eval.

Sampling the grid (probe -> ContactField) is NOT here — it lives in ``targets/interaction/eval.py``
(the online, q-independent consumer). This module only builds + caches the asset.

Ported from HoloNew ``contact/backends/sdf.py`` (the grid build), with the Coal distance backend
replaced by ``trimesh.proximity`` and the witness kept as a first-class grid.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..contracts import SDF
from ..config import SdfConfig


def build_sdf(vertices: np.ndarray, faces: np.ndarray, spacing: float, margin: float,
              name: str = "") -> SDF:
    """Signed-distance + witness grid over the mesh AABB padded by ``margin + 2*spacing``.

    The pad makes the whole activation band (half-width ``margin``) representable with a 2-voxel
    cushion for stable trilinear gradients at the band edge. Nodes hold the TRUE signed distance
    (negative inside) and the nearest surface point (witness); both are valid across the whole grid
    (a closest-point query is not band-limited). Pure: builds a trimesh internally, no I/O, leaves
    its inputs untouched, returns a frozen ``SDF``."""
    import trimesh

    verts = np.asarray(vertices, np.float64)
    tris = np.asarray(faces, np.int64)
    pad = float(margin) + 2.0 * float(spacing)
    lo = verts.min(0) - pad
    hi = verts.max(0) + pad
    dims = np.ceil((hi - lo) / spacing).astype(np.int64) + 1          # (3,) nodes per axis
    xs = lo[0] + spacing * np.arange(dims[0])
    ys = lo[1] + spacing * np.arange(dims[1])
    zs = lo[2] + spacing * np.arange(dims[2])
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    nodes = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)    # (M, 3)

    mesh = trimesh.Trimesh(vertices=verts, faces=tris, process=False)
    witness, unsigned, _ = trimesh.proximity.closest_point(mesh, nodes)  # (M, 3), (M,)
    inside = mesh.contains(nodes)                                        # (M,) bool
    signed = np.where(inside, -unsigned, unsigned)                       # (M,)

    shape = tuple(int(d) for d in dims)
    return SDF(
        grid=signed.reshape(shape).astype(np.float32),
        witness=witness.reshape(shape + (3,)).astype(np.float32),
        origin=lo.astype(np.float64),
        spacing=float(spacing),
        name=name,
    )


def build_plane_sdf(xy_min: np.ndarray, xy_max: np.ndarray, spacing: float, margin: float,
                    name: str = "ground") -> SDF:
    """Flat-ground SDF: the ``z = 0`` plane baked into a grid, so the ground is an ORDINARY ``SDF``
    (homogeneous with objects/terrain — the eval samples every channel the same way, no flat-ground
    special case). A plane's signed distance ``z`` and witness ``(x, y, 0)`` are AFFINE, so trilinear
    sampling reproduces them EXACTLY at any resolution; the grid only has to SPAN the probe region.

    Covers the scene's horizontal extent ``[xy_min, xy_max]`` (each (2,)) padded by ``margin +
    2*spacing``, with ``z`` in ``+-(margin + 2*spacing)`` — the contact band around the plane (a probe
    beyond it is out-of-grid -> inactive, exactly like any object SDF). Pure + analytic: no mesh, no
    trimesh, just a broadcast fill."""
    pad = float(margin) + 2.0 * float(spacing)
    lo = np.array([float(xy_min[0]) - pad, float(xy_min[1]) - pad, -pad], np.float64)
    hi = np.array([float(xy_max[0]) + pad, float(xy_max[1]) + pad, pad], np.float64)
    dims = np.ceil((hi - lo) / spacing).astype(np.int64) + 1          # (3,) nodes per axis
    xs = lo[0] + spacing * np.arange(dims[0])
    ys = lo[1] + spacing * np.arange(dims[1])
    zs = lo[2] + spacing * np.arange(dims[2])
    shape = tuple(int(d) for d in dims)

    grid = np.broadcast_to(zs, shape)                                # signed distance = z (plane z=0)
    gx, gy, _ = np.meshgrid(xs, ys, zs, indexing="ij")
    witness = np.stack([gx, gy, np.zeros(shape)], axis=-1)           # nearest surface point (x, y, 0)
    return SDF(grid=np.ascontiguousarray(grid, np.float32),
               witness=np.ascontiguousarray(witness, np.float32),
               origin=lo, spacing=float(spacing), name=name)


class SdfBuilder:
    """``AssetBuilder`` producing the ``SDF`` of one rigid mesh (object or terrain ground). Scoped to
    GEOMETRY (+ the ``SdfConfig``): two sequences sharing a mesh share the cached grid, independent
    of subject/robot. The runner wraps ``build``/``load`` in a ``prof.span("sdf")``."""

    def cache_key(self, config: SdfConfig, vertices: np.ndarray, faces: np.ndarray) -> str:
        """Stable hash of the geometry + the SDF resolution (spacing) and band (margin). Geometry
        only — no subject/robot term, so the grid is shared across every sequence using the mesh."""
        h = hashlib.sha1()
        h.update(f"{config.spacing}|{config.margin}".encode())
        h.update(np.ascontiguousarray(vertices, np.float32).tobytes())
        h.update(np.ascontiguousarray(faces, np.int64).tobytes())
        return h.hexdigest()

    def build(self, config: SdfConfig, vertices: np.ndarray, faces: np.ndarray,
              name: str = "") -> SDF:
        return build_sdf(vertices, faces, config.spacing, config.margin, name=name)

    def save(self, sdf: SDF, path: Path) -> None:
        return save_sdf(sdf, path)

    def load(self, path: Path) -> SDF:
        return load_sdf(path)


# =============================================================================
# Persistence — save/load co-located (the builder delegates here in one line)
# =============================================================================
def save_sdf(sdf: SDF, path: Path) -> None:
    """Serialise an ``SDF`` to ``path`` (``np.savez_compressed`` — the grids are large), creating
    parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), grid=sdf.grid, witness=sdf.witness,
                        origin=sdf.origin, spacing=np.float64(sdf.spacing),
                        name=np.array(sdf.name))


def load_sdf(path: Path) -> SDF:
    """Inverse of ``save_sdf``: load an ``SDF`` from ``path``."""
    d = np.load(str(path), allow_pickle=False)
    return SDF(grid=d["grid"], witness=d["witness"], origin=d["origin"],
               spacing=float(d["spacing"]), name=str(d["name"]))
