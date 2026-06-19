import numpy as np
from HoloNew.src.test_socp.contact.probes import make_floor_grid


def test_make_floor_grid():
    g = make_floor_grid(center_xy=(0.0, 0.0))
    assert g.ndim == 2 and g.shape[1] == 3
    assert np.allclose(g[:, 2], 0.0)


def test_coal_build_bvh_smoke():
    import trimesh
    from HoloNew.src.test_socp.contact.backends.coal import build_bvh
    box = trimesh.creation.box(extents=(1, 1, 1))
    bvh = build_bvh(np.asarray(box.vertices), np.asarray(box.faces))
    assert bvh is not None
