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
_HOIM3_NPZ = _REPO / "data/00_raw_datasets/HOI-M3/smplx/subject01_baseball.npz"
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


@pytest.mark.skipif(not (_HOIM3_NPZ.exists() and _SMPLX_DIR.exists()),
                    reason="HOI-M3 data / SMPL-X model not present")
def test_facade_hoim3_feeds_load():
    cfg = RetargetingConfig(dataset="hoim3", task_type="robot_only",
                            model_path=_SMPLX_DIR, motion_path=_HOIM3_NPZ)
    hj, op, scale = _load_like_view(cfg)
    assert cfg.data_format == "smplx"
    assert hj.shape[1:] == (22, 3)
    assert scale > 0
