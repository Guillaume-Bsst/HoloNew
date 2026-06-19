import tarfile
from pathlib import Path

import numpy as np
import pytest
import trimesh

from HoloNew.src.data_loaders.hodome import HoDomeLoader, hodome_object_poses


def _make_scaned_tar(scaned_dir, token):
    scaned_dir.mkdir(parents=True, exist_ok=True)
    work = scaned_dir / token
    work.mkdir()
    trimesh.creation.box(extents=(0.2, 0.2, 0.2)).export(work / f"{token}.obj")
    with tarfile.open(scaned_dir / f"{token}.tar", "w") as t:
        t.add(work / f"{token}.obj", arcname=f"{token}/{token}.obj")


def _make_object_npz(path, T=3):
    R = np.tile(np.eye(3), (T, 1, 1)).astype(np.float64)
    Tt = np.zeros((T, 1, 3), dtype=np.float64)
    Tt[:, 0, 0] = np.arange(T)  # moving in X
    np.savez(path, object_R=R, object_T=Tt, mocap_frame_rate=60)


def test_hodome_object_poses_identity(tmp_path):
    p = tmp_path / "seq_object.npz"; _make_object_npz(p, T=3)
    op = hodome_object_poses(p)
    assert op.shape == (3, 7)
    assert np.allclose(op[:, :4], [1, 0, 0, 0])      # identity R -> wxyz unit quat
    assert np.allclose(op[:, 4], [0, 1, 2])          # translation X


def test_hodome_object_source(tmp_path):
    token = "box"
    obj_dir = tmp_path / "object"
    obj_dir.mkdir()
    _make_object_npz(obj_dir / f"sub3_{token}.npz", T=3)
    _make_scaned_tar(tmp_path / "scaned_object", token)
    srcs = HoDomeLoader().object_source(
        motion_path=tmp_path / "smplx" / f"sub3_{token}.npz",
        obj_path=obj_dir / f"sub3_{token}.npz", model_path=None,
        task_type="object_interaction", constants=None, motion_data_config=None)
    assert len(srcs) == 1
    assert srcs[0].poses_raw.shape == (3, 7)
    assert Path(srcs[0].mesh_path).exists()


def test_hodome_object_source_robot_only_empty(tmp_path):
    srcs = HoDomeLoader().object_source(
        motion_path=tmp_path / "m.npz", obj_path=None, model_path=None,
        task_type="robot_only", constants=None, motion_data_config=None)
    assert srcs == []
