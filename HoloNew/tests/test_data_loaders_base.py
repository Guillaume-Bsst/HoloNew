import numpy as np
import pytest
from HoloNew.src.data_loaders.base import (
    MotionLoader, DATASET_TO_FORMAT, DATASET_LOADERS, register_loader, resolve_loader,
)


def test_dataset_to_format_map():
    assert DATASET_TO_FORMAT == {
        "omomo": "smplh", "hodome": "smplx", "sfu": "smplx",
        "lafan": "lafan", "climbing": "mocap",
    }


def test_register_and_resolve_loader():
    @register_loader("dummy")
    class DummyLoader(MotionLoader):
        def load(self, *, model_path, motion_path, obj_path, task_type,
                 constants, motion_data_config):
            return np.zeros((2, 22, 3)), np.zeros((2, 7)), 1.0

    assert "dummy" in DATASET_LOADERS
    loader = resolve_loader("dummy")
    hj, op, scale = loader.load(model_path=None, motion_path=None, obj_path=None,
                                task_type="robot_only", constants=None,
                                motion_data_config=None)
    assert hj.shape == (2, 22, 3) and op.shape == (2, 7) and scale == 1.0
    DATASET_LOADERS.pop("dummy")


def test_resolve_unknown_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        resolve_loader("nope")
