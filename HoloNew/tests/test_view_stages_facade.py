"""End-to-end smoke: the façade normalization feeds view_stages' load path."""
from pathlib import Path

import pytest

from HoloNew.config_types.data_type import MotionDataConfig
from HoloNew.config_types.robot import RobotConfig
from HoloNew.examples.robot_retarget import RetargetingConfig, create_task_constants, load_motion_data
from HoloNew.src.data_loaders.facade import normalize_dataset_cfg

_REPO = Path(__file__).resolve().parents[5]
_OMOMO_PICKLE = _REPO / "data/00_raw_datasets/OMOMO/data/train_diffusion_manip_seq_joints24.p"
_OMOMO_PT = Path("demo_data/OMOMO_new/sub3_largebox_003.pt")  # shipped, relative to pkg dir
_HODOME_NPZ = _REPO / "data/00_raw_datasets/HODome/smplx/subject01_baseball.npz"
_SMPLX_DIR = _REPO / "data/00_raw_datasets/models/models_smplx_v1_1/models/smplx"


def _load_like_view(cfg):
    normalize_dataset_cfg(cfg)
    mdc = MotionDataConfig(data_format=cfg.data_format, robot_type="g1")
    constants = create_task_constants(RobotConfig(robot_type="g1"), mdc, cfg.task_config, cfg.task_type)
    return load_motion_data(cfg.task_type, cfg.data_format, cfg.data_path, cfg.task_name, constants, mdc)


@pytest.mark.skipif(not _OMOMO_PT.exists(), reason="OMOMO_new demo .pt not present")
def test_facade_omomo_feeds_load(monkeypatch, tmp_path):
    cfg = RetargetingConfig(dataset="omomo", task_type="robot_only",
                            model_path=_OMOMO_PICKLE, motion_path=_OMOMO_PT.resolve())
    hj, op, scale = _load_like_view(cfg)
    assert cfg.data_format == "smplh"
    assert hj.shape[1:] == (52, 3)
    assert scale > 0


@pytest.mark.skipif(not (_HODOME_NPZ.exists() and _SMPLX_DIR.exists()),
                    reason="HODome data / SMPL-X model not present")
def test_facade_hodome_feeds_load():
    cfg = RetargetingConfig(dataset="hodome", task_type="robot_only",
                            model_path=_SMPLX_DIR, motion_path=_HODOME_NPZ)
    hj, op, scale = _load_like_view(cfg)
    assert cfg.data_format == "smplx"
    assert hj.shape[1:] == (22, 3)
    assert scale > 0


_OMOMO_ROOT = _REPO / "data/00_raw_datasets/OMOMO"
_OMOMO_NEW_ROOT = _REPO / "data/00_raw_datasets/OMOMO_new/OMOMO_new"
_SMPLH_DIR = _REPO / "data/00_raw_datasets/models/smplh"
_HODOME_ROOT = _REPO / "data/00_raw_datasets/HODome"
_SMPLX_ROOT = _REPO / "data/00_raw_datasets/models/models_smplx_v1_1/models"


@pytest.mark.skipif(not ((_OMOMO_NEW_ROOT / "sub3_largebox_003.pt").exists() and _SMPLH_DIR.exists()),
                    reason="OMOMO global roots not present")
def test_facade_by_name_omomo(monkeypatch):
    monkeypatch.setenv("WBT_OMOMO_NEW_DIR", str(_OMOMO_NEW_ROOT))
    monkeypatch.setenv("WBT_OMOMO_DIR", str(_OMOMO_ROOT))
    monkeypatch.setenv("WBT_SMPLH_DIR", str(_SMPLH_DIR))
    cfg = RetargetingConfig(dataset="omomo", task_type="robot_only", motion_name="sub3_largebox_003")
    hj, op, scale = _load_like_view(cfg)
    assert cfg.task_name == "sub3_largebox_003" and hj.shape[1:] == (52, 3) and scale > 0


@pytest.mark.skipif(not ((_HODOME_ROOT / "smplx/subject01_baseball.npz").exists() and _SMPLX_DIR.exists()),
                    reason="HODome global roots not present")
def test_facade_by_name_hodome(monkeypatch):
    monkeypatch.setenv("WBT_HODOME_DIR", str(_HODOME_ROOT))
    monkeypatch.setenv("WBT_SMPLX_DIR", str(_SMPLX_ROOT))
    # robot_only here: the resolver still fills obj_path by name (object_interaction
    # loading for smplx is a separate, not-yet-wired path).
    cfg = RetargetingConfig(dataset="hodome", task_type="robot_only", motion_name="subject01_baseball")
    hj, op, scale = _load_like_view(cfg)
    assert cfg.data_format == "smplx" and hj.shape[1:] == (22, 3)
    assert cfg.obj_path is not None and cfg.obj_path.name == "subject01_baseball.npz"
