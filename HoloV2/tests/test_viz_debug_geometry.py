"""Helpers de géométrie debug purs : entrée connue → valeur connue (sans viser, headless).

``node_coords`` n'est PAS testé ici — il réside dans ``src/viz/core/geometry.py``
(signature différente : prend un objet ``SDF``) et n'est pas dupliqué dans
``debug/_geometry``. Voir ``tests/test_viz_core_geometry.py`` pour sa couverture."""
import numpy as np

from src.viz.debug._geometry import (
    lowest_point,
    object_world_lowz,
    parity_error,
    surface_points,
)


def test_object_world_lowz_identity_translation():
    # 8 coins du cube unité dans [-1,1]^3, rotation identité, décalé de +5 en z
    # → coordonnées z monde dans {4, 6}, donc minimum = 4.
    c = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], np.float64)
    rot = np.eye(3)[None]                                    # (1, 3, 3)
    pos = np.array([[0.0, 0.0, 5.0]])                        # (1, 3)
    mz, lp = object_world_lowz(c, rot, pos, cap=8000)
    assert mz.shape == (1,) and lp.shape == (1, 3)
    assert np.isclose(mz[0], 4.0) and np.isclose(lp[0, 2], 4.0)


def test_object_world_lowz_cap_shape_only():
    # vérifie que le sous-échantillonnage (cap < V) conserve les bonnes formes de sortie
    v = np.random.default_rng(0).normal(size=(50, 3))
    mz, lp = object_world_lowz(v, np.eye(3)[None], np.zeros((1, 3)), cap=4)
    assert mz.shape == (1,) and lp.shape == (1, 3)


def test_lowest_point():
    # frame 0 : z ∈ {3, 1, 2} → min = 1 ; frame 1 : z ∈ {-1, 5, 0.5} → min = -1
    pts = np.array([[[0, 0, 3.0], [0, 0, 1.0], [0, 0, 2.0]],
                    [[0, 0, -1.0], [0, 0, 5.0], [0, 0, 0.5]]])   # (2, 3, 3)
    mz, lp = lowest_point(pts)
    assert np.allclose(mz, [1.0, -1.0])
    assert np.allclose(lp[:, 2], [1.0, -1.0])


def test_surface_points_centroid_and_vertex():
    # triangle : sommets (0,0,0), (1,0,0), (0,1,0)
    # barycentrique (1/3, 1/3, 1/3) → centroïde (1/3, 1/3, 0)
    # barycentrique (1, 0, 0) → premier sommet (0, 0, 0)
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float64)
    tri = np.array([[0, 1, 2]])
    assert np.allclose(surface_points(verts, tri, np.array([[1 / 3, 1 / 3, 1 / 3]])), [[1 / 3, 1 / 3, 0]])
    assert np.allclose(surface_points(verts, tri, np.array([[1.0, 0.0, 0.0]])), [[0, 0, 0]])


def test_parity_error():
    # vecteur (3, 4, 0) vs origine → norme L2 = 5
    assert np.allclose(parity_error(np.array([[3.0, 4.0, 0.0]]), np.zeros((1, 3))), [5.0])
