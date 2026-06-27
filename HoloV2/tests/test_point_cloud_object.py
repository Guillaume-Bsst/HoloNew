"""Object cloud bake: rigid K=1 sampling, deterministic + cache round-trip. The synthetic-box unit
test needs no external data; an integration case samples a real HODome object mesh when present."""
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from src.contracts import PointCloud, RobotSpec, SceneSpec
from config_types import CloudConfig
from src.prepare.point_cloud import (assemble_rigid_cloud, build_object_cloud,
                                        sample_object_surface)
from src.prepare.point_cloud.objects import ObjectCloudBuilder
from src.targets.interaction import pose_cloud

_DATA = Path("/home/vboxuser/Documents/wbt_rl/data/00_raw_datasets")
_HODOME = _DATA / "HODome"
_SMPLX = _DATA / "models" / "models_smplx_v1_1" / "models" / "smplx"


def test_assemble_rigid_cloud_is_k1_and_poses_rigidly():
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((128, 3))
    cloud = assemble_rigid_cloud(pts)
    assert cloud.n_points == 128 and cloud.n_influences == 1
    assert np.array_equal(cloud.parts, np.zeros((128, 1), np.int64))
    assert np.allclose(cloud.weights, 1.0)
    assert np.allclose(cloud.offsets[:, 0, :], pts)
    # posing with one part transform == one rigid placement of the whole cloud.
    rm, t = R.from_rotvec([0, 0, 0.5]).as_matrix(), np.array([1.0, 2.0, 3.0])
    assert np.allclose(pose_cloud(cloud, rm[None], t[None]), pts @ rm.T + t, atol=1e-6)


def test_build_object_cloud_on_box_determinism_roundtrip(tmp_path):
    import trimesh
    extents = np.array([0.4, 0.6, 0.8])
    box = trimesh.creation.box(extents=extents)
    v, f = np.asarray(box.vertices, np.float64), np.asarray(box.faces, np.int64)
    cfg = CloudConfig()

    pts = sample_object_surface(v, f, cfg.object_density, cfg.seed)
    assert pts.shape[0] >= 64
    # every even sample lies on a box face: at least one coord on a +-half-extent plane.
    assert np.isclose(np.abs(pts), extents / 2, atol=1e-6).any(axis=1).all()

    cloud = build_object_cloud(v, f, cfg)
    assert cloud.n_influences == 1 and cloud.sampling_id == ""

    builder = ObjectCloudBuilder()
    again = builder.build(cfg, v, f)                                # determinism (seeded sampling)
    assert np.array_equal(cloud.offsets, again.offsets)

    path = tmp_path / "obj_cloud.npz"                               # cache round-trip
    builder.save(cloud, path)
    loaded = builder.load(path)
    assert np.array_equal(cloud.parts, loaded.parts)
    assert np.array_equal(cloud.offsets, loaded.offsets)


def _pick() -> Path | None:
    sm, ob = _HODOME / "smplx", _HODOME / "object"
    if not (sm.is_dir() and ob.is_dir() and _SMPLX.is_dir()):
        return None
    shared = {p.stem for p in sm.glob("*.npz")} & {p.stem for p in ob.glob("*.npz")}
    return sm / f"{sorted(shared)[0]}.npz" if shared else None


_SEQ = _pick()


@pytest.mark.skipif(_SEQ is None, reason="HODome data not available")
def test_object_cloud_on_real_mesh_poses_near_world_pose():
    from src.prepare.load import load
    from src.prepare.load.mesh import load_mesh

    spec = SceneSpec(
        dataset="hodome", motion_path=_SEQ,
        robot=RobotSpec(name="g1", urdf_path=Path("g1.urdf"), link_names=("a",), dof=29, height=1.3),
        smpl_model_dir=_SMPLX,
    )
    raw = load(spec)
    assert len(raw.object_mesh_paths) >= 1
    v, f = load_mesh(raw.object_mesh_paths[0])
    cloud = build_object_cloud(v, f, CloudConfig())
    assert cloud.n_influences == 1 and cloud.n_points >= 64

    pose = np.asarray(raw.object_poses_raw[0][0], np.float64)        # frame 0: [x,y,z,qw,qx,qy,qz]
    rm = R.from_quat(pose[[4, 5, 6, 3]]).as_matrix()                 # wxyz -> xyzw
    world = pose_cloud(cloud, rm[None], pose[:3][None])
    assert np.isfinite(world).all()
    # the rigid object spans < ~1.5 m, so every posed point sits near its world translation.
    assert np.linalg.norm(world - pose[:3], axis=1).max() < 1.5
