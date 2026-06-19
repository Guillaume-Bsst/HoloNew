import numpy as np
import trimesh

from HoloNew.src.test_socp.movable import sample_object_surface


def test_sample_density_deterministic(tmp_path):
    box = trimesh.creation.box(extents=(1, 1, 1))
    f = tmp_path / "box.obj"
    box.export(f)
    a = sample_object_surface(str(f), density=200.0, seed=0)
    b = sample_object_surface(str(f), density=200.0, seed=0)
    assert a.shape[1] == 3 and len(a) >= 64
    assert np.allclose(a, b)   # deterministic at fixed seed
