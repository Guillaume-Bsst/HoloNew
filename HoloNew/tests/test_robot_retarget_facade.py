from pathlib import Path

import numpy as np
import pytest

from HoloNew.examples.robot_retarget import RetargetingConfig, _validate_dataset_paths


def test_new_fields_default_none():
    cfg = RetargetingConfig()
    assert cfg.dataset is None
    assert cfg.model_path is None and cfg.motion_path is None and cfg.obj_path is None


def test_validation_requires_paths_when_dataset_set():
    cfg = RetargetingConfig(dataset="hodome", task_type="robot_only")
    with pytest.raises(ValueError, match="requires --model-path"):
        _validate_dataset_paths(cfg)


def test_validation_obj_required_for_interaction():
    cfg = RetargetingConfig(dataset="omomo", task_type="object_interaction",
                            model_path=Path("a"), motion_path=Path("b"))
    with pytest.raises(ValueError, match="--obj-path"):
        _validate_dataset_paths(cfg)


def test_validation_passes_robot_only_without_obj():
    cfg = RetargetingConfig(dataset="omomo", task_type="robot_only",
                            model_path=Path("a"), motion_path=Path("b"))
    _validate_dataset_paths(cfg)  # no raise
