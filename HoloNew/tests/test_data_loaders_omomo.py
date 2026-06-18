from types import SimpleNamespace

import numpy as np
import torch
import joblib
import pytest

from HoloNew.src.data_loaders.omomo import OmomoMixedLoader


def _make_pt(path, T=4):
    # InterMimic packed tensor: load_intermimic_data reads joints at [162:162+52*3]
    # and object pose at [318:325]. Width 325 is enough.
    arr = np.zeros((T, 325), dtype=np.float32)
    arr[:, 162:162 + 52 * 3] = np.random.rand(T, 156)
    arr[:, 318:325] = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0])  # xyzw quat + pos slots
    torch.save(torch.from_numpy(arr), path)


def _make_pickle(path, seq_name):
    entry = {"seq_name": np.array(seq_name), "betas": np.zeros((1, 16), dtype=np.float32),
             "gender": np.array("neutral")}
    joblib.dump({0: entry}, path)


def test_omomo_loader_robot_only(tmp_path):
    seq = "sub3_largebox_003"
    pt = tmp_path / f"{seq}.pt"; _make_pt(pt)
    pk = tmp_path / "train.p"; _make_pickle(pk, seq)
    constants = SimpleNamespace(ROBOT_HEIGHT=1.32)

    hj, op, scale = OmomoMixedLoader().load(
        model_path=pk, motion_path=pt, obj_path=None,
        task_type="robot_only", constants=constants, motion_data_config=None)

    assert hj.shape == (4, 52, 3)
    assert op.shape == (4, 7)
    assert np.allclose(op[0], [1, 0, 0, 0, 0, 0, 0])  # dummy object for robot_only
    assert scale > 0
