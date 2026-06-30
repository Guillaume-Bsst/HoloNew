"""Visualiseur SDF — débogage visuel de la construction ``prepare/sdf``.

Rend tout ``SDF`` (objet / terrain / sol plat) dans son cadre local :
  - une tranche COUPE-TRANSVERSALE mobile de la grille, coloriée par distance signée (bleu = intérieur/négatif,
    blanc = surface/zéro, rouge = extérieur/positif) : la traversée zéro doit tracer la surface,
  - la coquille BANDE près-surface (|d| < marge) coloriée de la même façon,
  - lignes WITNESS (nœud bande → son point de surface le plus proche stocké) : chaque ligne doit finir SUR la surface,
  - la maille source fantôme quand il y en a une (un SDF plan sol-plat n'a pas de maille).

Consommateur pur : pilote le builder une fois pour récupérer l'asset, puis le lit uniquement ; pas de hooks de calcul.

Exécution :
    python -m src.viz.sdf --mesh <path.obj>   [--spacing 0.02] [--margin 0.05] [--port 8080]
    python -m src.viz.sdf --plane <size_m>    [--spacing 0.05] [--margin 0.05] [--port 8080]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ..prepare.contracts import SDF
from ..prepare.load.mesh import load_mesh
from ..prepare.sdf.build import build_plane_sdf, build_sdf


def _node_coords(sdf: SDF) -> np.ndarray:
    """(Nx, Ny, Nz, 3) coords locales de chaque nœud grille."""
    nx, ny, nz = sdf.grid.shape
    xs = sdf.origin[0] + sdf.spacing * np.arange(nx)
    ys = sdf.origin[1] + sdf.spacing * np.arange(ny)
    zs = sdf.origin[2] + sdf.spacing * np.arange(nz)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([gx, gy, gz], axis=-1)


def _diverging(dist: np.ndarray, vmax: float) -> np.ndarray:
    """Distance signée → (N,3) uint8 RGB. -vmax = bleu, 0 = blanc, +vmax = rouge (serré)."""
    t = np.clip(dist / max(vmax, 1e-9), -1.0, 1.0)
    col = np.ones((len(t), 3), np.float64)                      # blanc à t=0
    neg = t < 0
    a = (-t[neg])[:, None]
    col[neg] = (1 - a) * np.array([1, 1, 1]) + a * np.array([0.20, 0.35, 1.0])   # → bleu
    b = (t[~neg])[:, None]
    col[~neg] = (1 - b) * np.array([1, 1, 1]) + b * np.array([1.0, 0.25, 0.20])  # → rouge
    return (col * 255).astype(np.uint8)


def view_sdf(sdf: SDF, margin: float, *, verts: np.ndarray | None = None,
             faces: np.ndarray | None = None, port: int = 8080) -> None:
    import viser

    spacing = sdf.spacing
    coords = _node_coords(sdf)                                   # (Nx,Ny,Nz,3)
    nx, ny, nz = sdf.grid.shape
    inside_pct = 100.0 * float((sdf.grid < 0).mean())

    # coquille bande (|d| < marge) : coords, dist signée, witness stocké — pour la coquille + lignes witness
    mask = np.abs(sdf.grid) < margin
    band_xyz = coords[mask]; band_d = sdf.grid[mask]; band_w = sdf.witness[mask]
    rng = np.random.default_rng(0)
    sub = rng.choice(len(band_xyz), min(500, len(band_xyz)), replace=False) if len(band_xyz) else []

    print(f"SDF '{sdf.name}': grid {nx}x{ny}x{nz} ({nx*ny*nz} nodes), spacing={spacing}, "
          f"inside%={inside_pct:.1f}, band nodes={int(mask.sum())}")

    srv = viser.ViserServer(port=port)
    srv.scene.add_grid("/grid", width=2.0, height=2.0)

    with srv.gui.add_folder("Layers"):
        show_mesh = srv.gui.add_checkbox("mesh ghost", verts is not None)
        show_slice = srv.gui.add_checkbox("slice", True)
        show_band = srv.gui.add_checkbox("band shell", False)
        show_wit = srv.gui.add_checkbox("witness lines", False)
    with srv.gui.add_folder("Slice"):
        axis = srv.gui.add_dropdown("axis", ("X", "Y", "Z"), initial_value="Y")
        idx = srv.gui.add_slider("index", 0, ny - 1, 1, ny // 2)
    info = srv.gui.add_markdown("")

    band_h = srv.scene.add_point_cloud("/band", band_xyz.astype(np.float32),
                                       _diverging(band_d, margin), point_size=spacing * 0.6)
    if len(sub):
        seg = np.stack([band_xyz[sub], band_w[sub]], axis=1).astype(np.float32)   # (S,2,3) node->witness
        wcol = np.where((band_d[sub] < 0)[:, None, None],
                        np.array([[[60, 90, 255]]]), np.array([[[255, 80, 60]]]))
        wcol = np.broadcast_to(wcol, (len(sub), 2, 3)).astype(np.uint8)
    else:
        seg = np.zeros((1, 2, 3), np.float32); wcol = np.zeros((1, 2, 3), np.uint8)
    wit_h = srv.scene.add_line_segments("/witness", seg, wcol, line_width=1.5)

    def render(_=None):
        a = {"X": 0, "Y": 1, "Z": 2}[axis.value]
        idx.max = sdf.grid.shape[a] - 1
        i = min(int(idx.value), sdf.grid.shape[a] - 1)
        sl = [slice(None)] * 3; sl[a] = i
        pts = coords[tuple(sl)].reshape(-1, 3)
        dist = sdf.grid[tuple(sl)].reshape(-1)
        srv.scene.add_point_cloud("/slice", pts.astype(np.float32), _diverging(dist, margin),
                                  point_size=spacing * 0.9, visible=show_slice.value)
        if verts is not None:
            srv.scene.add_mesh_simple("/mesh", verts.astype(np.float32), faces, color=(150, 150, 160),
                                      opacity=0.35 if show_mesh.value else 0.0, side="double")
        band_h.visible = show_band.value
        wit_h.visible = show_wit.value
        info.content = (
            f"**{sdf.name}** · grille {nx}×{ny}×{nz} · espacement {spacing} · marge {margin}\n\n"
            f"intérieur **{inside_pct:.1f}%**\n\n"
            f"tranche **{axis.value}={i}**  ·  bleu=intérieur (−) · blanc=surface (0) · rouge=extérieur (+)")

    for h in (show_mesh, show_slice, show_band, show_wit, axis, idx):
        h.on_update(render)
    render()
    print(f"viser ready -> http://localhost:{port}")
    while True:
        time.sleep(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--mesh", type=Path, help="object/terrain mesh -> SDF")
    src.add_argument("--plane", type=float, metavar="SIZE", help="flat ground: SIZE×SIZE m plane SDF")
    ap.add_argument("--spacing", type=float, default=0.02)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    if args.plane is not None:
        h = args.plane / 2.0
        sdf = build_plane_sdf([-h, -h], [h, h], args.spacing, args.margin, name="ground")
        view_sdf(sdf, args.margin, port=args.port)
    else:
        verts, faces = load_mesh(args.mesh)
        t0 = time.time()
        sdf = build_sdf(verts, faces, args.spacing, args.margin, name=args.mesh.stem)
        print(f"built mesh SDF in {time.time() - t0:.1f}s")
        view_sdf(sdf, args.margin, verts=verts, faces=faces, port=args.port)


if __name__ == "__main__":
    main()
