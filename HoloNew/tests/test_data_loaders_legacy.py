from types import SimpleNamespace
from pathlib import Path

import numpy as np

import HoloNew.src.data_loaders as dl
from HoloNew.src.data_loaders.legacy import LegacyLoader


def test_legacy_wraps_load_motion_data(monkeypatch):
    captured = {}

    def fake_load_motion_data(task_type, data_format, data_path, task_name,
                              constants, motion_data_config):
        captured.update(dict(task_type=task_type, data_format=data_format,
                             data_path=data_path, task_name=task_name))
        return np.zeros((3, 22, 3)), np.zeros((3, 7)), 0.7

    monkeypatch.setattr("HoloNew.src.data_loaders.legacy.load_motion_data",
                        fake_load_motion_data)

    mdc = SimpleNamespace(data_format="lafan")
    hj, op, scale = LegacyLoader("lafan").load(
        model_path=None, motion_path=Path("/data/walk1.npy"), obj_path=None,
        task_type="robot_only", constants=SimpleNamespace(), motion_data_config=mdc)

    assert captured["data_path"] == Path("/data")
    assert captured["task_name"] == "walk1"
    assert captured["data_format"] == "lafan"
    assert hj.shape == (3, 22, 3) and scale == 0.7


def test_legacy_loaders_registered():
    for name in ("lafan", "sfu", "climbing"):
        assert name in dl.DATASET_LOADERS
