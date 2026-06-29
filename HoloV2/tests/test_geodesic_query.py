"""Helper de query géodésique (numpy-only) : nearest_index (snap), et geo_value_grad par MLS degré-1.
Reproduction EXACTE d'un champ linéaire (le fit degré-1 est exact dessus), gradient vs différences
finies sur un champ non-linéaire, et import via la surface publique targets."""
import numpy as np

from src.prepare.contracts import GeodesicTable
from src.targets.interaction.geodesic import nearest_index, geo_value_grad


def _table_from_field(points, field):
    P = len(points)
    geo = np.tile(field.astype(np.float32), (P, 1))     # geo[src] = field pour toute source (test)
    return GeodesicTable(points=points.astype(np.float32), normals=np.tile([0, 0, 1.0], (P, 1)).astype(np.float32),
                         geo=geo, name="t")


def test_nearest_index():
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    assert nearest_index(pts, np.array([0.9, 0.1, 0])) == 1
    assert list(nearest_index(pts, np.array([[0.1, 0.1, 0], [0.1, 0.9, 0]]))) == [0, 2]


def test_linear_field_reproduced_exactly():
    # champ f(x) = a·x + c → MLS degré-1 doit rendre g=f(query) et grad=a EXACTEMENT.
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1, 1, size=(200, 3)); pts[:, 2] = 0.0      # surface plate (2D dans z=0)
    a = np.array([2.0, -3.0, 0.0]); c = 0.5
    field = pts @ a + c
    table = _table_from_field(pts, field)
    q = np.array([[0.2, -0.1, 0.0], [-0.3, 0.4, 0.0]])
    g, grad = geo_value_grad(table, source_idx=np.array([0, 0]), query_xyz=q, k=8)
    assert np.allclose(g, q @ a + c, atol=1e-6)
    assert np.allclose(grad[:, :2], a[:2], atol=1e-6)            # gradient (composantes dans le plan)


def test_gradient_matches_finite_difference():
    rng = np.random.default_rng(1)
    pts = rng.uniform(-1, 1, size=(400, 3)); pts[:, 2] = 0.0
    p0 = np.array([0.0, 0.0, 0.0])
    field = np.linalg.norm(pts - p0, axis=1)                    # non-linéaire (cône)
    table = _table_from_field(pts, field)
    q = np.array([[0.3, 0.2, 0.0]])
    g, grad = geo_value_grad(table, np.array([0]), q, k=12)
    eps = 1e-4
    fd = np.array([(geo_value_grad(table, np.array([0]), q + d, k=12)[0][0]
                    - geo_value_grad(table, np.array([0]), q - d, k=12)[0][0]) / (2 * eps)
                   for d in (np.array([[eps, 0, 0]]), np.array([[0, eps, 0]]))])
    assert np.allclose(grad[0, :2], fd, atol=2e-2)


def test_public_surface_import():
    from src.targets import geo_value_grad as g1, nearest_index as g2
    assert callable(g1) and callable(g2)
