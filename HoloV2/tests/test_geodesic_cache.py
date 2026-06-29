"""Round-trip .npz de GeodesicTable : save->load reproduit exactement (writer/reader co-localisés)."""
import numpy as np

from src.prepare.contracts import GeodesicTable
from src.prepare.geodesic.cache import save_geo, load_geo


def test_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    t = GeodesicTable(points=rng.standard_normal((6, 3)).astype(np.float32),
                      normals=rng.standard_normal((6, 3)).astype(np.float32),
                      geo=rng.standard_normal((6, 6)).astype(np.float32),
                      name="obj0", sampling_id="deadbeef")
    p = tmp_path / "sub" / "obj0.npz"           # parents créés à la volée
    save_geo(t, p)
    g = load_geo(p)
    assert np.array_equal(t.points, g.points)
    assert np.array_equal(t.normals, g.normals)
    assert np.array_equal(t.geo, g.geo)
    assert g.name == "obj0" and g.sampling_id == "deadbeef"
