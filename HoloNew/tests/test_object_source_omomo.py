import numpy as np
import torch
from HoloNew.src.data_loaders.omomo import OmomoMixedLoader


def _fake_pt(path, T=4):
    # InterMimic layout: joints [162:318], object pose [318:325].
    row = np.zeros((T, 400), np.float32)
    # Within [318:325]: src order [x,y,z,qx,qy,qz,qw]; load_intermimic_data emits
    # [qw,qx,qy,qz,x,y,z]. Set qw=1 (src idx 6 -> col 324) and x=arange (col 318).
    row[:, 324] = 1.0
    row[:, 318] = np.arange(T)
    torch.save(torch.from_numpy(row), str(path))


def test_omomo_object_source_robot_only_empty(tmp_path):
    p = tmp_path / "sub3_largebox_003.pt"
    _fake_pt(p)
    srcs = OmomoMixedLoader().object_source(
        motion_path=p, obj_path=None, model_path=None, task_type="robot_only",
        constants=None, motion_data_config=None)
    assert srcs == []


def test_omomo_object_source_unknown_object_empty(tmp_path):
    # An object with no bundled mesh and no omomo_dir -> no source.
    p = tmp_path / "sub3_nosuchobj_003.pt"
    _fake_pt(p)
    srcs = OmomoMixedLoader().object_source(
        motion_path=p, obj_path=p, model_path=None, task_type="object_interaction",
        constants=None, motion_data_config=None)
    assert srcs == []


def test_omomo_object_source_bundled_largebox(tmp_path):
    # largebox is bundled in the package; resolved regardless of cwd, model_path=None.
    p = tmp_path / "sub3_largebox_003.pt"
    _fake_pt(p, T=5)
    srcs = OmomoMixedLoader().object_source(
        motion_path=p, obj_path=p, model_path=None, task_type="object_interaction",
        constants=None, motion_data_config=None)
    assert len(srcs) == 1
    assert srcs[0].poses_raw.shape == (5, 7)
    assert np.allclose(srcs[0].poses_raw[:, 0], 1.0)            # qw
    assert np.allclose(srcs[0].poses_raw[:, 4], np.arange(5))   # x
    assert srcs[0].mesh_path.name == "largebox.obj"
