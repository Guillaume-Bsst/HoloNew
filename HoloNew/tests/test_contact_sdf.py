import numpy as np
import trimesh
from HoloNew.src.gmr_socp_v2.contact.backends.sdf import (
    build_object_field, sdf_surface_field, save_object_sdf, load_object_sdf,
)

def test_build_and_query_box_sdf(tmp_path):
    box = trimesh.creation.box(extents=(0.4, 0.4, 0.4))
    sdf = build_object_field(box, margin=0.05, resolution=0.02)
    f = sdf_surface_field(np.array([[0.25, 0.0, 0.0], [5.0, 5.0, 5.0]], float), sdf, margin=0.05)
    assert f.distance.shape == (2,)
    assert f.active[0] and not f.active[1]
    assert f.distance[1] == np.float32(0.05)
    p = tmp_path / "sdf.npz"
    save_object_sdf(sdf, p)
    r = load_object_sdf(p)
    np.testing.assert_array_equal(r.data, sdf.data)
