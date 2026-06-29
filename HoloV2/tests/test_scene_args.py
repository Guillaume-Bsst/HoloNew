import argparse
from pathlib import Path

from src.viz._scene_args import add_scene_args, scene_from_args


def _toml(tmp_path: Path) -> Path:
    p = tmp_path / "paths.toml"
    p.write_text('smplx = "/models/smplx"\n[roots]\nhodome = "/data/HODome"\n')
    return p


def _parse(argv):
    ap = argparse.ArgumentParser()
    add_scene_args(ap)
    return ap.parse_args(argv)


def test_defaults_filled_from_paths(tmp_path):
    a = _parse(["--dataset", "hodome", "--motion-path", "smplx/s01.npz"])
    spec = scene_from_args(a, paths_file=_toml(tmp_path))
    assert spec.smpl_model_dir == Path("/models/smplx")
    assert spec.motion_path == Path("/data/HODome/smplx/s01.npz")
    assert spec.dataset_root == Path("/data/HODome")


def test_explicit_args_override_paths(tmp_path):
    a = _parse(["--dataset", "hodome", "--motion-path", "/abs/seq.npz",
                "--model-dir", "/m", "--dataset-root", "/r"])
    spec = scene_from_args(a, paths_file=_toml(tmp_path))
    assert spec.motion_path == Path("/abs/seq.npz")
    assert spec.smpl_model_dir == Path("/m")
    assert spec.dataset_root == Path("/r")


def test_object_names_split(tmp_path):
    a = _parse(["--dataset", "hodome", "--motion-path", "/abs/seq.npz",
                "--model-dir", "/m", "--dataset-root", "/r", "--object-names", "box,case"])
    spec = scene_from_args(a, paths_file=_toml(tmp_path))
    assert spec.object_names == ("box", "case")


def test_robot_urdf_is_real_repo_model(tmp_path):
    a = _parse(["--dataset", "hodome", "--motion-path", "/abs/seq.npz",
                "--model-dir", "/m", "--dataset-root", "/r"])
    spec = scene_from_args(a, paths_file=_toml(tmp_path))
    assert spec.robot.urdf_path.name == "g1_29dof.urdf"
    assert spec.robot.urdf_path.is_absolute()
