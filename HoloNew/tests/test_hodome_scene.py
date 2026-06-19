from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import trimesh
import mujoco
from HoloNew.src.data_loaders.hodome_scene import build_hodome_scene_xml, ensure_object_scene_xml
from HoloNew.src.data_loaders.base import ObjectSource

_G1 = Path("models/g1/g1_29dof.xml")


def test_scene_content(tmp_path):
    mesh = tmp_path / "box.obj"
    trimesh.creation.box(extents=(0.2, 0.2, 0.2)).export(mesh)
    out = tmp_path / "scene.xml"
    p = build_hodome_scene_xml(_G1, "baseball", mesh, output_path=out)
    txt = Path(p).read_text()
    assert '<mesh name="baseball_mesh"' in txt
    assert str(mesh.resolve()) in txt              # absolute object-mesh path
    assert '<body name="baseball_link">' in txt
    assert "<freejoint/>" in txt
    assert 'diaginertia="0.002 0.002 0.002"' in txt


def test_scene_parses_and_adds_free_joint(tmp_path):
    # Written next to the robot meshes so meshdir="assets/" resolves; cleaned up after.
    mesh = tmp_path / "box.obj"
    trimesh.creation.box(extents=(0.3, 0.3, 0.3)).export(mesh)
    base_nq = mujoco.MjModel.from_xml_path(str(_G1)).nq
    out = _G1.with_name("g1_29dof_w_pytesttoken.xml")
    try:
        build_hodome_scene_xml(_G1, "pytesttoken", mesh, output_path=out)
        m = mujoco.MjModel.from_xml_path(str(out))
        assert m.nq == base_nq + 7                 # object free joint adds 7 qpos
    finally:
        out.unlink(missing_ok=True)


# --- Task 3: ensure_object_scene_xml ---

def _make_cfg(dataset, task_type="object_interaction", obj_path=None,
              motion_path=None, model_path=None, smpl_model_dir=None):
    """Minimal cfg-like namespace for ensure_object_scene_xml tests."""
    return SimpleNamespace(
        dataset=dataset,
        task_type=task_type,
        obj_path=obj_path,
        motion_path=motion_path,
        model_path=model_path,
        smpl_model_dir=smpl_model_dir,
        motion_data_config=SimpleNamespace(),
    )


def _make_constants(token="baseball", robot_urdf="models/g1/g1_29dof.urdf"):
    """Constants namespace with OBJECT_NAME and ROBOT_URDF_FILE."""
    return SimpleNamespace(OBJECT_NAME=token, ROBOT_URDF_FILE=robot_urdf)


def test_ensure_returns_none_for_non_hodome():
    """ensure_object_scene_xml is a no-op for non-hodome datasets (e.g. omomo)."""
    cfg = _make_cfg(dataset="omomo")
    constants = _make_constants()
    result = ensure_object_scene_xml(cfg, constants)
    assert result is None


def test_ensure_returns_none_when_dataset_is_none():
    """ensure_object_scene_xml returns None when no dataset is configured."""
    cfg = _make_cfg(dataset=None)
    constants = _make_constants()
    result = ensure_object_scene_xml(cfg, constants)
    assert result is None


def test_ensure_returns_none_for_robot_only():
    """ensure_object_scene_xml returns None for robot_only task type (no object)."""
    cfg = _make_cfg(dataset="hodome", task_type="robot_only")
    constants = _make_constants()
    result = ensure_object_scene_xml(cfg, constants)
    assert result is None


def test_ensure_returns_none_when_no_object_source(tmp_path):
    """ensure_object_scene_xml returns None when the loader returns no object sources."""
    import numpy as np
    cfg = _make_cfg(dataset="hodome", task_type="object_interaction",
                    obj_path=tmp_path / "s01_baseball.npz")
    constants = _make_constants()
    # Mock resolve_loader to return a loader that yields no object sources
    mock_loader = SimpleNamespace(
        object_source=lambda **kwargs: []
    )
    import HoloNew.src.data_loaders.hodome_scene as _mod
    with patch.object(_mod, "resolve_loader", return_value=mock_loader):
        result = ensure_object_scene_xml(cfg, constants)
    assert result is None


def test_ensure_creates_scene_xml_and_returns_path(tmp_path):
    """ensure_object_scene_xml calls build_hodome_scene_xml and returns the scene path.

    The scene xml is written next to the robot xml (models/g1/); cleaned up in finally.
    """
    import numpy as np
    mesh = tmp_path / "baseball.obj"
    trimesh.creation.box(extents=(0.1, 0.1, 0.1)).export(mesh)
    fake_poses = np.zeros((10, 7), dtype=np.float32)
    fake_poses[:, 0] = 1.0  # qw = 1

    cfg = _make_cfg(dataset="hodome", task_type="object_interaction",
                    obj_path=tmp_path / "s01_baseball.npz")
    constants = _make_constants(token="baseball", robot_urdf="models/g1/g1_29dof.urdf")

    mock_loader = SimpleNamespace(
        object_source=lambda **kwargs: [ObjectSource(mesh_path=mesh, poses_raw=fake_poses)]
    )
    expected_out = Path("models/g1/g1_29dof_w_baseball.xml")
    import HoloNew.src.data_loaders.hodome_scene as _mod
    try:
        with patch.object(_mod, "resolve_loader", return_value=mock_loader):
            result = ensure_object_scene_xml(cfg, constants)
        assert result is not None
        assert result == expected_out
        assert result.exists()
        txt = result.read_text()
        assert '<body name="baseball_link">' in txt
        assert "<freejoint/>" in txt
    finally:
        expected_out.unlink(missing_ok=True)
