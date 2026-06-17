import numpy as np
import trimesh
from HoloNew.src.test_socp.contact.backends.sdf import (
    build_object_field, sdf_surface_field, save_object_sdf, load_object_sdf,
    load_or_build_object_sdf,
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


def test_load_or_build_caches_keyed_by_L_and_resolution(tmp_path):
    """On-demand SDF bake: builds + caches under an (L, resolution) key, reloads on a
    second call, and a larger L yields a wider band + extent (the field stores the true
    distance out to L). Distinct keys never collide."""
    box = trimesh.creation.box(extents=(0.4, 0.4, 0.4))
    mesh_file = tmp_path / "box.obj"
    box.export(mesh_file)

    sdf = load_or_build_object_sdf(str(mesh_file), margin=0.05, resolution=0.02,
                                   cache_dir=tmp_path)
    cache = tmp_path / "box_L0.05_r0.020.npz"
    assert cache.exists(), "first call must bake and cache under the (L, res) key"
    assert abs(float(sdf.data.max()) - 0.05) < 1e-3, "band should clamp at L=0.05"

    # Second call hits the cache (identical grid, no rebuild).
    again = load_or_build_object_sdf(str(mesh_file), margin=0.05, resolution=0.02,
                                     cache_dir=tmp_path)
    np.testing.assert_array_equal(again.data, sdf.data)

    # Larger L -> distinct cache key, wider band + more nodes.
    wide = load_or_build_object_sdf(str(mesh_file), margin=0.15, resolution=0.02,
                                    cache_dir=tmp_path)
    assert (tmp_path / "box_L0.15_r0.020.npz").exists()
    assert abs(float(wide.data.max()) - 0.15) < 1e-3
    assert np.prod(wide.dims) > np.prod(sdf.dims), "wider band must enlarge the grid"
