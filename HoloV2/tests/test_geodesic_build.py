"""prepare/geodesic build (fonctions pures) : parité du sampling avec le object_cloud, géométrie
géodésique sur formes connues (plan ≈ euclidien, sphère ≫ euclidien), invariants matrice, et les
garde-fous (graphe disconnecté, max_points, gating normales au niveau du graphe)."""
import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from src.prepare.config import CloudConfig, GeodesicConfig
from src.prepare.point_cloud.objects import sample_object_surface
from src.prepare.geodesic.build import (sample_surface_with_normals, build_knn_graph,
                                        all_pairs_geodesic, build_geodesic_table,
                                        _bridge_disconnected)


def _sphere(sub=3, r=0.2):
    m = trimesh.creation.icosphere(subdivisions=sub, radius=r)      # fermé, bien connecté, normales sortantes
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)


def _grid(n=30, side=1.0):
    """Nappe plane (points + normales +z) — graphe/géodésique contrôlés sans confondre avec le sampling."""
    xs = np.linspace(0.0, side, n)
    xx, yy = np.meshgrid(xs, xs)
    pts = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1)
    nrm = np.tile([0.0, 0.0, 1.0], (pts.shape[0], 1))
    return pts, nrm


def test_sampling_matches_object_cloud():
    # Parité EXACTE des points avec le object_cloud (même densité/seed/topo) : un seul sampling canonique.
    v, f = _sphere()
    pts_geo, nrm = sample_surface_with_normals(v, f, density=500.0, seed=0)
    pts_cloud = sample_object_surface(v, f, density=500.0, seed=0)
    assert np.array_equal(pts_geo, pts_cloud)
    assert nrm.shape == pts_geo.shape
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-5)   # normales unitaires


def test_plane_geodesic_close_to_euclidean():
    # Surface plate (nappe) : la géodésique de graphe ≈ euclidien (sur-estime un peu, jamais en dessous).
    pts, nrm = _grid(n=30, side=1.0)
    D = all_pairs_geodesic(build_knn_graph(pts, nrm, k=8, normal_gate=-0.5))
    eucl = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    far = eucl > 0.3
    ratio = D[far] / eucl[far]
    assert (ratio >= 0.999).all()                                    # jamais sous l'euclidien
    assert np.median(ratio) < 1.2                                    # sur-estime modérément


def test_sphere_geodesic_exceeds_euclidean():
    # Sphère (convexe, SANS raccourci) : la paire ~antipodale a une géodésique ≈ grand cercle (π·r)
    # ≫ la corde euclidienne (2r). NB un cylindre PLEIN ne marche pas : ses capuchons plats offrent un
    # raccourci à travers la face (ratio ~1.1), donc on prend une sphère.
    v, f = _sphere(sub=3, r=0.2)
    pts, nrm = sample_surface_with_normals(v, f, density=1500.0, seed=0)
    D = all_pairs_geodesic(build_knn_graph(pts, nrm, k=10, normal_gate=-1.0))
    eucl = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    i, j = np.unravel_index(np.argmax(eucl), eucl.shape)            # paire la plus écartée (~antipodale)
    assert D[i, j] > 1.3 * eucl[i, j]                               # le chemin suit la surface (grand cercle)


def test_matrix_invariants():
    pts, nrm = _grid(n=20)
    D = all_pairs_geodesic(build_knn_graph(pts, nrm, k=8, normal_gate=-0.5))
    assert np.allclose(np.diag(D), 0.0)
    assert np.allclose(D, D.T, atol=1e-5)
    assert D.dtype == np.float32


def test_disconnected_graph_bridged():
    # Deux amas éloignés, k=1 → 2 composantes. _bridge_disconnected les relie → all-pairs FINI
    # (plus d'abort), la distance inter-amas ≈ l'écart euclidien (~10) traversé par le pont.
    a = np.zeros((5, 3)); b = np.zeros((5, 3)) + [[10.0, 0, 0]]
    pts = np.concatenate([a, b]) + np.random.default_rng(0).normal(0, 1e-3, (10, 3))
    nrm = np.tile([0.0, 0, 1.0], (10, 1))
    graph = _bridge_disconnected(pts, build_knn_graph(pts, nrm, k=1, normal_gate=-1.0))
    D = all_pairs_geodesic(graph)
    assert np.isfinite(D).all()
    assert D[0, 5] > 9.0                                # traverse le pont (~10)


def test_all_pairs_raises_on_raw_disconnected():
    # garde-fou interne : all_pairs_geodesic SANS bridging lève toujours (ne stocke pas d'inf).
    pts = np.concatenate([np.zeros((3, 3)), np.zeros((3, 3)) + [[10.0, 0, 0]]])
    nrm = np.tile([0.0, 0, 1.0], (6, 1))
    with pytest.raises(ValueError):
        all_pairs_geodesic(build_knn_graph(pts, nrm, k=1, normal_gate=-1.0))


def test_normal_gate_cuts_opposite_normal_edges():
    # 2 points proches à normales ~opposées (faces d'une plaque) : l'arête directe existe SANS gating
    # et disparaît AVEC (dot≈-0.995 < -0.5). Une arête à même normale reste. Test au niveau du graphe.
    pts = np.array([[0.0, 0, 0.001], [0.0, 0, -0.001], [0.05, 0, 0.001], [0.05, 0, -0.001]])
    nrm = np.array([[0.0, 0, 1.0], [0.0, 0.1, -1.0], [0.0, 0, 1.0], [0.0, 0.1, -1.0]])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)              # bas ≈ (0, 0.0995, -0.995)
    g_open = build_knn_graph(pts, nrm, k=3, normal_gate=-1.0)      # garde tout sauf dot=-1 exact
    g_gated = build_knn_graph(pts, nrm, k=3, normal_gate=-0.5)
    assert g_open[0, 1] > 0                                        # 0-1 (opposés, proches) présent
    assert g_gated[0, 1] == 0                                      # coupé par le gating
    assert g_gated[0, 2] > 0                                       # 0-2 (même normale) conservé


def test_max_points_guard():
    v, f = _sphere()
    with pytest.raises(ValueError):
        build_geodesic_table(v, f, CloudConfig(object_density=800.0, seed=0),
                             GeodesicConfig(max_points=10))         # P >> 10


def test_build_table_shapes():
    v, f = _sphere()
    t = build_geodesic_table(v, f, CloudConfig(object_density=500.0, seed=0),
                             GeodesicConfig(normal_gate=-1.0), name="obj0")
    assert t.geo.shape == (t.n_points, t.n_points)
    assert t.points.dtype == np.float32 and t.geo.dtype == np.float32
    assert t.name == "obj0" and t.sampling_id != ""
