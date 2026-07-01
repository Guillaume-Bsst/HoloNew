"""Visualiseur SDF — débogage visuel de la construction ``prepare/sdf`` (réécrit sur ``viz/core``).

Rend tout ``SDF`` (objet / terrain / sol plat) dans son cadre local :
  - une tranche COUPE-TRANSVERSALE mobile de la grille, coloriée par distance signée (bleu = intérieur/négatif,
    blanc = surface/zéro, rouge = extérieur/positif) : la traversée zéro doit tracer la surface,
  - la coquille BANDE près-surface (|d| < marge) coloriée de la même façon,
  - lignes WITNESS (nœud bande → son point de surface le plus proche stocké) : chaque ligne doit finir SUR la surface,
  - la maille source fantôme quand il y en a une (un SDF plan sol-plat n'a pas de maille).

Viewer de débogage avec son propre parser ``--mesh``/``--plane`` (il visualise un seul SDF dans son cadre
local, pas un mouvement). Il pilote le builder ``prepare/sdf`` une fois puis lit uniquement l'asset. Il
n'y a PAS d'axe temps → ce viewer n'utilise PAS ``core/Player`` ; il consomme ``core/colors.diverging`` +
``core/viser_ops`` + le helper pur ``core.geometry.node_coords``. viser confiné à ``core/viser_ops`` + ce module.

Exécution :
    python -m src.viz.debug.sdf --mesh <path.obj>   [--spacing 0.02] [--margin 0.05] [--port 8080]
    python -m src.viz.debug.sdf --plane <size_m>    [--spacing 0.05] [--margin 0.05] [--port 8080]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ...prepare.contracts import SDF
from ...prepare.load.mesh import load_mesh
from ...prepare.sdf.build import build_plane_sdf, build_sdf
from ..core.colors import diverging
from ..core.geometry import node_coords
from ..core.viser_ops import add_line_segments, add_point_cloud


def view_sdf(sdf: SDF, margin: float, *, verts: np.ndarray | None = None,
             faces: np.ndarray | None = None, port: int = 8080) -> None:
    """Lance le visualiseur SDF interactif pour un ``SDF`` donné.

    Crée un serveur viser, affiche la tranche mobile, la bande shell, les lignes witness et
    éventuellement la maille source fantôme. Bloque indéfiniment (keep-alive minimal sans axe temps).

    Paramètres
    ----------
    sdf :
        Grille distance-signée à visualiser.
    margin :
        Demi-largeur de la bande shell |d| < margin (en mètres).
    verts :
        Sommets de la maille source (V, 3) — ``None`` pour un SDF plan.
    faces :
        Triangles de la maille source (F, 3) — ``None`` pour un SDF plan.
    port :
        Port viser (défaut 8080).
    """
    import viser

    spacing = sdf.spacing
    coords = node_coords(sdf)                               # (Nx, Ny, Nz, 3) via core.geometry
    nx, ny, nz = sdf.grid.shape
    inside_pct = 100.0 * float((sdf.grid < 0).mean())

    # Coquille bande (|d| < marge) : coords, dist signée, witness stocké — pour shell + lignes witness
    mask = np.abs(sdf.grid) < margin
    band_xyz = coords[mask]; band_d = sdf.grid[mask]; band_w = sdf.witness[mask]
    rng = np.random.default_rng(0)
    sub = rng.choice(len(band_xyz), min(500, len(band_xyz)), replace=False) if len(band_xyz) else []

    print(f"SDF '{sdf.name}': grid {nx}x{ny}x{nz} ({nx * ny * nz} nodes), spacing={spacing}, "
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

    # Bande shell : nuage de points coloriés par distance signée via core.colors.diverging
    band_h = add_point_cloud(srv, "/band", band_xyz.astype(np.float32),
                             diverging(band_d, margin), point_size=spacing * 0.6)
    # Lignes witness : chaque segment va du nœud bande vers son point surface stocké
    if len(sub):
        seg = np.stack([band_xyz[sub], band_w[sub]], axis=1).astype(np.float32)   # (S, 2, 3)
        wcol = np.where((band_d[sub] < 0)[:, None, None],
                        np.array([[[60, 90, 255]]]), np.array([[[255, 80, 60]]]))
        wcol = np.broadcast_to(wcol, (len(sub), 2, 3)).astype(np.uint8)
    else:
        seg = np.zeros((1, 2, 3), np.float32); wcol = np.zeros((1, 2, 3), np.uint8)
    wit_h = add_line_segments(srv, "/witness", seg, wcol, line_width=1.5)

    def render(_=None) -> None:
        """Rafraîchit la tranche et les toggles de visibilité."""
        a = {"X": 0, "Y": 1, "Z": 2}[axis.value]
        idx.max = sdf.grid.shape[a] - 1
        i = min(int(idx.value), sdf.grid.shape[a] - 1)
        sl = [slice(None)] * 3; sl[a] = i
        pts = coords[tuple(sl)].reshape(-1, 3)
        dist = sdf.grid[tuple(sl)].reshape(-1)
        # La géométrie de la tranche change à chaque axis/index : re-add sous le même nom + visible.
        sh = add_point_cloud(srv, "/slice", pts.astype(np.float32), diverging(dist, margin),
                             point_size=spacing * 0.9)
        sh.visible = show_slice.value
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
    while True:                        # viewer sans axe temps : keep-alive minimal, pas de Player
        time.sleep(1)


def main() -> None:
    """Point d'entrée CLI : parse les arguments et lance ``view_sdf``."""
    ap = argparse.ArgumentParser(description="Visualiseur SDF — débogage étape prepare/sdf")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--mesh", type=Path, help="maille objet/terrain -> SDF")
    src.add_argument("--plane", type=float, metavar="SIZE", help="sol plat : plan SIZE×SIZE m")
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
