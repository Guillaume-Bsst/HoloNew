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
    from scipy.spatial.transform import Rotation as R
    from HoloNew.src.data_loaders.hodome import _YUP_TO_ZUP
    p = tmp_path / "seq_object.npz"; _make_object_npz(p, T=3)
    op = hodome_object_poses(p)
    assert op.shape == (3, 7)
    # Identity native R, expressed in the physically Y->Z rotated scene, is the world
    # rotation Q itself (NOT identity): the object's native frame stands up Z-up.
    q_Q = R.from_matrix(_YUP_TO_ZUP).as_quat()[[3, 0, 1, 2]]   # xyzw -> wxyz
    assert np.allclose(op[:, :4], q_Q)
    assert np.allclose(op[:, 4], [0, 1, 2])          # translation X (unchanged by Q here)


def test_hodome_object_poses_is_world_rotation_not_conjugation(tmp_path):
    # The object mesh is used in its NATIVE (Y-up) local frame (sample_object_surface /
    # trimesh load apply no Y->Z swap), so the returned Z-up pose must reproduce the
    # rigidly Y->Z rotated scene: applying it to a native point == Q @ (its native Y-up
    # placement). That holds for a world rotation (Q R), NOT for conjugation (Q R Q^T),
    # which extra-rotates the object's local axes and mis-orients it (cf. the human
    # global_orientations_zup fix).
    from scipy.spatial.transform import Rotation as R
    from HoloNew.src.data_loaders.hodome import _YUP_TO_ZUP
    rng = np.random.default_rng(2)
    T = 4
    Rm = R.from_rotvec(rng.normal(scale=0.7, size=(T, 3))).as_matrix()   # native Y-up
    Tt = rng.normal(size=(T, 1, 3))
    p = tmp_path / "seq_object.npz"
    np.savez(p, object_R=Rm, object_T=Tt, mocap_frame_rate=60)
    op = hodome_object_poses(p)
    quat, trans = op[:, :4], op[:, 4:7]
    Q = _YUP_TO_ZUP
    v = rng.normal(size=(5, 3))                                          # native object-local points
    for i in range(T):
        Rz = R.from_quat(quat[i][[1, 2, 3, 0]]).as_matrix()
        world_zup = v @ Rz.T + trans[i]                                 # returned Z-up pose on native pts
        world_yup = v @ Rm[i].T + Tt[i, 0]                              # native Y-up placement
        assert np.allclose(world_zup, world_yup @ Q.T, atol=1e-9)      # == rigid Q rotation of the scene


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
