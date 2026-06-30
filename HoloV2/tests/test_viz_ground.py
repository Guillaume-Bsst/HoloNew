"""ground_surface_mesh — extraction de heightfield depuis le SDF (pur). Un SDF de plan plat -> un quad
plat à la hauteur du plan sur l'étendue de la grille (le principe : la couche ground lit le SDF,
pas une box codée en dur)."""
import numpy as np

from src.prepare.contracts import SDF
from src.viz.core.layer import Layer
from src.viz.layers.ground import GroundLayer, ground_surface_mesh


def _plane_sdf(plane_z: float) -> SDF:
    """Grille 3x3xNz dont la distance signée est (z - plane_z) : zéro-crossing à z = plane_z."""
    nx = ny = 3
    nz = 5
    spacing = 0.1
    origin = np.array([0.0, 0.0, -0.2])
    zs = origin[2] + spacing * np.arange(nz)                     # -0.2 .. 0.2
    grid = np.zeros((nx, ny, nz))
    for k in range(nz):
        grid[:, :, k] = zs[k] - plane_z                          # distance au plan selon z
    witness = np.zeros((nx, ny, nz, 3))
    return SDF(grid=grid, witness=witness, origin=origin, spacing=spacing, name="ground")


def test_flat_plane_heightfield():
    """Plan à z=0 : tous les sommets à z=0, grille 3x3=9 sommets, 8 triangles."""
    verts, faces = ground_surface_mesh(_plane_sdf(0.0))
    assert verts.shape == (9, 3) and verts.dtype == np.float32
    assert np.allclose(verts[:, 2], 0.0, atol=1e-6)              # tous à la hauteur du plan
    assert faces.shape == (8, 3)                                  # 2 tris * (3-1)*(3-1) cellules
    assert faces.dtype == np.int64 and faces.max() < 9


def test_plane_at_offset_height():
    """Plan décalé à z=0.1 : tous les sommets à z=0.1."""
    verts, _ = ground_surface_mesh(_plane_sdf(0.1))
    assert np.allclose(verts[:, 2], 0.1, atol=1e-6)


def test_ground_layer_is_a_layer():
    """GroundLayer implémente le protocole Layer et déclare le dossier 'Static'."""
    assert isinstance(GroundLayer(), Layer)
    assert GroundLayer().folder == "Static"
