import numpy as np
from HoloNew.src.contact.object_input import parse_obj_name
from HoloNew.src.contact.probes import make_floor_grid, make_object_grid


def test_parse_obj_name():
    assert parse_obj_name("sub3_largebox_003") == "largebox"


def test_make_floor_grid():
    g = make_floor_grid(center_xy=(0.0, 0.0))
    assert g.ndim == 2 and g.shape[1] == 3
    assert np.allclose(g[:, 2], 0.0)


def test_coal_build_bvh_smoke():
    import trimesh
    from HoloNew.src.contact.backends.coal import build_bvh
    box = trimesh.creation.box(extents=(1, 1, 1))
    bvh = build_bvh(np.asarray(box.vertices), np.asarray(box.faces))
    assert bvh is not None
