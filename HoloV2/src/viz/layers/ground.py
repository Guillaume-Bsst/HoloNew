"""Couche sol — rend la vraie surface du canal ground depuis son SDF (plan OU terrain), remplaçant
l'ancienne box plate codée en dur à z=0. ``ground_surface_mesh`` est un helper pur (SDF -> heightfield)
directement testé unitairement ; la couche ne fait qu'ajouter le mesh une fois au setup et bascule
sa visibilité."""
from __future__ import annotations

import numpy as np

from ..core.layer import UiState
from ..model import VizContext, VizFrame


def ground_surface_mesh(sdf) -> tuple[np.ndarray, np.ndarray]:
    """``SDF`` -> (verts (Nx*Ny, 3) f32, faces (M, 3) int64). Heightfield : par colonne (x, y), la
    surface z est le zéro-crossing de la distance signée selon z (interpolation linéaire entre les
    nœuds encadrants ; repli = nœud de |d| minimal). Un SDF de plan plat -> quad plat sur son
    étendue ; un SDF terrain -> sa surface. Lit uniquement le SDF (pur, sans viser)."""
    grid = np.asarray(sdf.grid, np.float64)                       # (Nx, Ny, Nz)
    nx, ny, nz = grid.shape
    ox, oy, oz = (float(v) for v in sdf.origin)
    h = float(sdf.spacing)
    xs = ox + h * np.arange(nx)
    ys = oy + h * np.arange(ny)
    zs = oz + h * np.arange(nz)

    zsurf = np.empty((nx, ny), np.float64)
    for i in range(nx):
        for j in range(ny):
            d = grid[i, j]                                        # (Nz,) distance signée selon z
            sign = np.signbit(d)
            cross = np.where(sign[:-1] != sign[1:])[0]           # intervalles de zéro-crossing
            if len(cross):
                k = int(cross[0])
                d0, d1 = d[k], d[k + 1]
                t = 0.0 if d1 == d0 else d0 / (d0 - d1)         # interpolation vers d == 0
                zsurf[i, j] = zs[k] + t * h
            else:
                zsurf[i, j] = zs[int(np.argmin(np.abs(d)))]     # pas de crossing -> nœud le plus proche

    gx, gy = np.meshgrid(xs, ys, indexing="ij")                  # (Nx, Ny)
    verts = np.stack([gx, gy, zsurf], axis=-1).reshape(-1, 3).astype(np.float32)

    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a = i * ny + j
            b = a + 1
            c = a + ny
            d_ = c + 1
            faces.append([a, c, b])
            faces.append([b, c, d_])
    faces_arr = np.asarray(faces, np.int64) if faces else np.zeros((0, 3), np.int64)
    return verts, faces_arr


class GroundLayer:
    """Surface sol statique lue depuis le SDF du canal ground (ajoutée une fois ; checkbox la bascule)."""

    folder = "Static"

    def setup(self, server, gui, ctx: VizContext) -> None:
        """Construit le mesh heightfield depuis le SDF sol et crée la checkbox de visibilité."""
        verts, faces = ground_surface_mesh(ctx.ground_sdf)
        self._handle = server.scene.add_mesh_simple(
            "/ground", verts, faces, color=(170, 170, 178), side="double")
        self._cb = gui.add_checkbox("ground", True)
        self._cb.on_update(lambda _: setattr(self._handle, "visible", self._cb.value))

    def update(self, frame: VizFrame, ui: UiState) -> None:
        """Rafraîchit la visibilité du mesh sol d'après la checkbox."""
        self._handle.visible = self._cb.value
