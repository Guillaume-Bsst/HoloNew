"""GeodesicBuilder : déterminisme (build x2 identique), round-trip cache (save->load == build), et
sensibilité de la clé (sampling + knobs graphe + géométrie). Calqué sur test_sdf."""
import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from src.prepare.config import CloudConfig, GeodesicConfig
from src.prepare.geodesic.build import GeodesicBuilder


def _sphere():
    m = trimesh.creation.icosphere(subdivisions=3, radius=0.2)     # fermé → graphe connecté
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)


_CC = CloudConfig(object_density=500.0, seed=0)
_GC = GeodesicConfig(normal_gate=-1.0)


def test_determinism():
    v, f = _sphere()
    b = GeodesicBuilder()
    a = b.build(_CC, _GC, v, f)
    c = b.build(_CC, _GC, v, f)
    assert np.array_equal(a.points, c.points)
    assert np.array_equal(a.geo, c.geo)


def test_cache_roundtrip(tmp_path):
    v, f = _sphere()
    b = GeodesicBuilder()
    t = b.build(_CC, _GC, v, f, name="obj0")
    p = tmp_path / "obj0.npz"
    b.save(t, p)
    g = b.load(p)
    assert np.array_equal(t.geo, g.geo)
    assert np.array_equal(t.points, g.points)
    assert g.name == "obj0"


def test_cache_key_sensitivity():
    v, f = _sphere()
    b = GeodesicBuilder()
    k = b.cache_key(_CC, _GC, v, f)
    assert k == b.cache_key(_CC, _GC, v, f)
    assert k != b.cache_key(CloudConfig(object_density=800.0, seed=0), _GC, v, f)   # densité
    assert k != b.cache_key(CloudConfig(object_density=500.0, seed=1), _GC, v, f)   # seed
    assert k != b.cache_key(_CC, GeodesicConfig(k_neighbors=12, normal_gate=-1.0), v, f)  # k
    assert k != b.cache_key(_CC, GeodesicConfig(normal_gate=0.0), v, f)             # gate
    v2 = v.copy(); v2[0] += 0.1
    assert k != b.cache_key(_CC, _GC, v2, f)                                        # géométrie
