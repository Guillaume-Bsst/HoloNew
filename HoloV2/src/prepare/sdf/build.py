"""prepare/sdf — construit la grille distance-signée + witness d'une surface : un MESH objet/terrain
(``build_sdf``) ou le plan GROUND plat (``build_plane_sdf``, analytique, pas de mesh).

Le SDF mesh est offline, geometry-keyed, caché une fois (``SdfBuilder``) : un asset subject-/robot-free
partagé par chaque séquence touchant le même mesh. Mesh-agnostic — UN builder pour objets ET terrain.
Le sol plat est AUSSI un SDF (pas un cas analytique spécial) : un plan est affine, donc une petite
grille le reproduit exactement, gardant chaque canal eval homogène (pas de branche flat-ground).

Pipeline build (trimesh, pas Coal — ce code tourne une fois et est caché, donc la vitesse par-probe
est sans importance) : remplit chaque nœud grille avec la distance signée à la surface.
``trimesh.proximity.closest_point`` retourne le point de surface le plus proche (le WITNESS, stocké
first-class) ET la distance non-signée en un appel vectorisé ; ``mesh.contains`` donne le signe
intérieur/extérieur. La grille stocke la TRUE distance signée partout (pas de clamp) : la bande
d'activation (``dist < margin``) est appliquée plus tard, à l'eval.

Échantillonner la grille (probe → ContactField) n'est PAS ici — ça vit dans ``targets/interaction/eval.py``
(le consumer online, q-indépendant). Ce module construit l'asset ; sa persistence ``.npz`` vit dans
``cache.py`` (le builder délègue).

Porté de HoloNew ``contact/backends/sdf.py`` (le grid build), avec le backend distance Coal remplacé
par ``trimesh.proximity`` et le witness gardé comme grille first-class.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..contracts import SDF
from ..config import SdfConfig
from .cache import load_sdf, save_sdf


def build_sdf(vertices: np.ndarray, faces: np.ndarray, spacing: float, margin: float,
              name: str = "") -> SDF:
    """Grille distance-signée + witness sur l'AABB mesh rembourré de ``margin + 2*spacing``.

    Le rembourrage rend la bande d'activation entière (demi-largeur ``margin``) représentable avec
    un coussin 2-voxel pour des gradients trilinéaires stables à la bordure bande. Les nœuds tiennent
    la TRUE distance signée (négative intérieur) et le point de surface le plus proche (witness) ;
    les deux sont valides sur la grille entière (une requête closest-point n'est pas band-limitée).
    Pur : construit un trimesh en interne, pas I/O, laisse ses inputs intacts, retourne un ``SDF`` gelé."""
    import trimesh

    verts = np.asarray(vertices, np.float64)
    tris = np.asarray(faces, np.int64)
    pad = float(margin) + 2.0 * float(spacing)
    lo = verts.min(0) - pad
    hi = verts.max(0) + pad
    dims = np.ceil((hi - lo) / spacing).astype(np.int64) + 1          # (3,) nœuds par axe
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
    """SDF sol-plat : le plan ``z = 0`` cuit dans une grille, donc le sol est un ``SDF`` ORDINAIRE
    (homogène avec objets/terrain — l'eval échantillonne chaque canal de la même façon, pas de cas
    spécial sol-plat). La distance signée ``z`` d'un plan et le witness ``(x, y, 0)`` sont AFFINE,
    donc l'interpolation trilinéaire les reproduit EXACTEMENT à n'importe quelle résolution ; la grille
    n'a que COUVRIR la région probe.

    Couvre l'étendue horizontale ``[xy_min, xy_max]`` de la scène (chacun (2,)) rembourré de
    ``margin + 2*spacing``, avec ``z`` dans ``+-(margin + 2*spacing)`` — la bande contact autour du
    plan (une probe au-delà est out-of-grid → inactive, exactement comme tout SDF objet). Pur +
    analytique : pas de mesh, pas de trimesh, juste un remplissage broadcast."""
    pad = float(margin) + 2.0 * float(spacing)
    lo = np.array([float(xy_min[0]) - pad, float(xy_min[1]) - pad, -pad], np.float64)
    hi = np.array([float(xy_max[0]) + pad, float(xy_max[1]) + pad, pad], np.float64)
    dims = np.ceil((hi - lo) / spacing).astype(np.int64) + 1          # (3,) nœuds par axe
    xs = lo[0] + spacing * np.arange(dims[0])
    ys = lo[1] + spacing * np.arange(dims[1])
    zs = lo[2] + spacing * np.arange(dims[2])
    shape = tuple(int(d) for d in dims)

    grid = np.broadcast_to(zs, shape)                                # distance signée = z (plan z=0)
    gx, gy, _ = np.meshgrid(xs, ys, zs, indexing="ij")
    witness = np.stack([gx, gy, np.zeros(shape)], axis=-1)           # point de surface le plus proche (x, y, 0)
    return SDF(grid=np.ascontiguousarray(grid, np.float32),
               witness=np.ascontiguousarray(witness, np.float32),
               origin=lo, spacing=float(spacing), name=name)


class SdfBuilder:
    """``AssetBuilder`` produisant le ``SDF`` d'un mesh rigide (objet ou terrain ground). Scopé à
    la GÉOMÉTRIE (+ le ``SdfConfig``) : deux séquences partageant un mesh partagent la grille cachée,
    indépendamment du sujet/robot. Le runner enveloppe ``build``/``load`` dans un ``prof.span("sdf")``."""

    def cache_key(self, config: SdfConfig, vertices: np.ndarray, faces: np.ndarray) -> str:
        """Hash stable de la géométrie + la résolution SDF (spacing) et bande (margin). Géométrie
        uniquement — pas de terme sujet/robot, donc la grille est partagée entre chaque séquence utilisant le mesh."""
        h = hashlib.sha1()
        h.update(f"{config.spacing}|{config.margin}".encode())
        h.update(np.ascontiguousarray(vertices, np.float32).tobytes())
        h.update(np.ascontiguousarray(faces, np.int64).tobytes())
        return h.hexdigest()

    def build(self, config: SdfConfig, vertices: np.ndarray, faces: np.ndarray,
              name: str = "") -> SDF:
        return build_sdf(vertices, faces, config.spacing, config.margin, name=name)

    def save(self, sdf: SDF, path: Path) -> None:
        return save_sdf(sdf, path)             # la persistence vit dans cache.py

    def load(self, path: Path) -> SDF:
        return load_sdf(path)
