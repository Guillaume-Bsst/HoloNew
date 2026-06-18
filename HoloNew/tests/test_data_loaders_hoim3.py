import numpy as np
import pytest

from HoloNew.src.data_loaders.hoim3 import HoiM3Loader, hoim3_object_poses


def _make_object_npz(path, T=3):
    R = np.tile(np.eye(3), (T, 1, 1)).astype(np.float64)
    Tt = np.zeros((T, 1, 3), dtype=np.float64)
    Tt[:, 0, 0] = np.arange(T)  # moving in X
    np.savez(path, object_R=R, object_T=Tt, mocap_frame_rate=60)


def test_hoim3_object_poses_identity(tmp_path):
    p = tmp_path / "seq_object.npz"; _make_object_npz(p, T=3)
    op = hoim3_object_poses(p)
    assert op.shape == (3, 7)
    assert np.allclose(op[:, :4], [1, 0, 0, 0])      # identity R -> wxyz unit quat
    assert np.allclose(op[:, 4], [0, 1, 2])          # translation X
