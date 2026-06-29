"""_build_channels rattache une GeodesicTable par objet (et None au sol plat). Scène minimale montée à
la main (un petit cube exporté), force=True pour bâtir sans cache. _build_channels n'utilise ni la
calibration ni le robot → valeurs factices acceptables."""
import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from src.obs import NULL
from src.prepare.config import PrepareConfig, SdfConfig, CloudConfig, GeodesicConfig
from src.prepare.contracts import (GroundedScene, Calibration, SceneSpec, RobotSpec, GeodesicTable)
from src.prepare.runner import _build_channels


def _spec(tmp_path, ground=None):
    robot = RobotSpec(name="g1", urdf_path=tmp_path / "x.urdf", link_names=("a",), dof=1, height=1.2)
    return SceneSpec(dataset="demo", motion_path=tmp_path, robot=robot,
                     ground_mesh_path=ground, cache_dir=tmp_path)


def _grounded(mesh_path):
    pose = np.tile([0.0, 0, 0, 1, 0, 0, 0], (2, 1))
    calib = Calibration(human_offset=0.0, object_offset=0.0, root_frame=np.eye(4))
    return GroundedScene(joint_pos=np.zeros((2, 3, 3)), joint_names=("a", "b", "c"),
                         object_poses=(pose,), object_mesh_paths=(mesh_path,), calibration=calib,
                         fps=30.0)


def _cfg():
    # coarse + clairsemé → test rapide
    return PrepareConfig(sdf=SdfConfig(spacing=0.05, margin=0.05),
                         cloud=CloudConfig(object_density=200.0, seed=0),
                         geodesic=GeodesicConfig(normal_gate=-1.0))


def test_object_channel_gets_geodesic_flat_ground_none(tmp_path):
    box = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    mesh_path = tmp_path / "box.obj"; box.export(mesh_path)
    channels = _build_channels(_grounded(mesh_path), _spec(tmp_path), _cfg(), tmp_path, NULL, force=True)
    assert channels[0].name == "ground" and channels[0].geodesic is None       # sol plat
    assert isinstance(channels[1].geodesic, GeodesicTable)                      # objet
    assert channels[1].geodesic.n_points == channels[1].geodesic.geo.shape[0]


def test_terrain_ground_gets_geodesic(tmp_path):
    # Sol-TERRAIN (spec.ground_mesh_path != None) → le canal ground reçoit une GeodesicTable
    # (contrairement au sol plat qui reste None). Couvre la branche terrain de _build_channels.
    obj = trimesh.creation.box(extents=(0.3, 0.3, 0.1))
    obj_path = tmp_path / "obj.obj"; obj.export(obj_path)
    terrain = trimesh.creation.box(extents=(1.0, 1.0, 0.05))
    terrain_path = tmp_path / "terrain.obj"; terrain.export(terrain_path)
    channels = _build_channels(_grounded(obj_path), _spec(tmp_path, ground=terrain_path),
                               _cfg(), tmp_path, NULL, force=True)
    assert channels[0].name == "ground" and channels[0].object_idx is None
    assert isinstance(channels[0].geodesic, GeodesicTable)   # terrain ground HAS a table
