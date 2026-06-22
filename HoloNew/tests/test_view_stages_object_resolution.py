"""The GMR/TEST solve must resolve a dataset's object the same way for HODome as for
OMOMO. view_stages clears cfg.dataset (so holosoma's main() loads via the normalized
legacy fields); the object resolver then falls onto the legacy OBJECT_MESH_FILE + .pt
path, which can't reach a dataset whose object lives outside a .pt (HODome: scanned tar +
separate npz). _solve_dataset_key re-exposes the dataset to the solve ONLY when there is
no legacy .pt, so HODome resolves its object via the loader while OMOMO keeps its float32
legacy path (preserving solve numerics)."""
import types
from pathlib import Path

import pytest

from HoloNew.examples.view_stages import _solve_dataset_key


def _cfg(data_path, task_name):
    return types.SimpleNamespace(data_path=Path(data_path), task_name=task_name)


def test_no_dataset_stays_none():
    # An ad-hoc (no --dataset) run: nothing to re-expose.
    assert _solve_dataset_key(_cfg("/whatever", "clip"), None) is None


def test_omomo_with_pt_keeps_legacy_path(tmp_path):
    # OMOMO has a .pt next to its motion -> keep the legacy (float32) resolution, so the
    # solve numerics / golden tests are untouched: the key must come back None.
    (tmp_path / "seq.pt").write_bytes(b"")
    assert _solve_dataset_key(_cfg(tmp_path, "seq"), "omomo") is None


def test_hodome_without_pt_reexposes_dataset(tmp_path):
    # HODome (smplx) has no .pt -> the legacy path can't see the object, so the dataset
    # key must be re-exposed for the solve's loader-based object_source.
    assert _solve_dataset_key(_cfg(tmp_path, "subject01_baseball"), "hodome") == "hodome"


# --- integration: the real HODome run now resolves its object in the solve ---
from HoloNew.src.paths import get_path  # noqa: E402

_HAVE_BASEBALL = (get_path("hodome") / "scaned_object" / "baseball.tar").exists()


@pytest.mark.skipif(not _HAVE_BASEBALL, reason="HODome baseball mesh not present")
def test_hodome_solve_resolves_object():
    """End-to-end at the resolution layer: replicate view_stages' cfg handling for HODome,
    apply _solve_dataset_key, and assert resolve_object_inputs now returns a real mesh +
    poses (=> the solve will build object_sdf -> object_surface_local, so the viewer's
    'Object->Floor' / 'Object surface pts' overlays get data)."""
    from HoloNew.examples.view_stages import ViewStagesConfig
    from HoloNew.examples.robot_retarget import DEFAULT_DATA_FORMATS, create_task_constants
    from HoloNew.config_types.data_type import MotionDataConfig
    from HoloNew.config_types.robot import RobotConfig
    from HoloNew.src.data_loaders.facade import normalize_dataset_cfg
    from HoloNew.src.test_socp.builder import resolve_object_inputs

    cfg = ViewStagesConfig(dataset="hodome", motion_name="subject01_baseball",
                           methods=("test_socp",))
    normalize_dataset_cfg(cfg)
    dataset = cfg.dataset
    cfg.dataset = None                                   # exactly what view_stages does

    # Bug repro: with dataset cleared the legacy path finds nothing.
    data_format = cfg.data_format or DEFAULT_DATA_FORMATS[cfg.task_type]
    cfg.motion_data_config = MotionDataConfig(data_format=data_format, robot_type=cfg.robot)
    if cfg.robot_config.robot_type != cfg.robot:
        cfg.robot_config = RobotConfig(robot_type=cfg.robot)
    constants = create_task_constants(
        robot_config=cfg.robot_config, motion_data_config=cfg.motion_data_config,
        task_config=cfg.task_config, task_type=cfg.task_type)
    _pt = cfg.data_path / f"{cfg.task_name}.pt"
    _pt = _pt if _pt.exists() else None
    assert resolve_object_inputs(cfg, constants, _pt) == (None, None)

    # Fix: re-expose the dataset for the solve -> the object resolves via the loader.
    cfg.dataset = _solve_dataset_key(cfg, dataset)
    assert cfg.dataset == "hodome"
    mesh_file, poses = resolve_object_inputs(cfg, constants, _pt)
    assert mesh_file is not None and Path(mesh_file).exists()
    assert poses is not None and poses.ndim == 2 and poses.shape[1] == 7
