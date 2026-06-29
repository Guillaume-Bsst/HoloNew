"""runner._build_channels mesh-dedup unit test (synthetic, no heavy deps): the per-object meshes are
loaded ONCE upstream and passed in, so _build_channels must NOT reload them (zero load_mesh per
object); a flat ground needs no mesh at all. The SDF / geodesic builders pull trimesh/coal, so they
are faked here — we assert the load_mesh WIRING (no double read), not the SDF math.
"""
from pathlib import Path

import numpy as np

from src.obs import NULL
from src.prepare import runner
from src.prepare.config import PrepareConfig
from src.prepare.contracts import Calibration, GroundedScene, RobotSpec, SceneSpec


class _FakeBuilder:
    """Stand-in for SdfBuilder / GeodesicBuilder: trivial cache_key/build/save/load, no heavy deps."""
    def cache_key(self, *a):
        return "k"

    def build(self, *a, **k):
        return object()

    def save(self, asset, path):
        pass

    def load(self, path):
        return object()


def _grounded(paths):
    obj = np.tile([0.0, 0.0, 0.0, 1, 0, 0, 0], (1, 1)).astype(np.float32)
    return GroundedScene(joint_pos=np.zeros((1, 1, 3), np.float32), joint_names=("a",),
                         object_poses=tuple(obj for _ in paths), object_mesh_paths=tuple(paths),
                         calibration=Calibration(0.0, 0.0, np.eye(4)), fps=30.0,
                         smpl_params=None, body=None)


def _spec():
    return SceneSpec(dataset="x", motion_path=Path("m"),
                     robot=RobotSpec(name="g1", urdf_path=Path("u"), link_names=(), dof=1, height=1.0),
                     ground_mesh_path=None)


def test_build_channels_does_not_reload_provided_object_meshes(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(runner, "load_mesh",
                        lambda p: calls.append(p) or (np.zeros((3, 3)), np.zeros((1, 3), int)))
    monkeypatch.setattr(runner, "SdfBuilder", _FakeBuilder)
    monkeypatch.setattr(runner, "GeodesicBuilder", _FakeBuilder)
    monkeypatch.setattr(runner, "build_plane_sdf", lambda *a, **k: object())

    paths = [Path("a.obj"), Path("b.obj")]
    meshes = [(np.zeros((3, 3)), np.zeros((1, 3), int)) for _ in paths]
    channels = runner._build_channels(_grounded(paths), _spec(), PrepareConfig(), tmp_path, NULL,
                                      force=True, object_meshes=meshes)

    assert calls == []                          # flat ground (no mesh) + objects from provided meshes
    assert len(channels) == 3                    # ground + 2 objects
    assert channels[0].name == "ground" and channels[0].object_idx is None
    assert [c.object_idx for c in channels[1:]] == [0, 1]
