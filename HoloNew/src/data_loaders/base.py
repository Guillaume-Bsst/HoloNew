"""Common interface and registry for dataset motion loaders.

Each loader turns three explicit paths (model / motion / object) into the unified
motion contract consumed by robot_retarget.main():
    human_joints  (T, J, 3)  Z-up metres
    object_poses  (T, 7)     [qw, qx, qy, qz, x, y, z]
    smpl_scale    float       robot_height / human_height
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

# Dataset key -> internal data_format (drives joint registry / mapping / toe names).
DATASET_TO_FORMAT: dict[str, str] = {
    "omomo": "smplh",
    "hodome": "smplx",
    "sfu": "smplx",
    "lafan": "lafan",
    "climbing": "mocap",
}


class MotionLoader(ABC):
    """Turns (model_path, motion_path, obj_path) into the unified motion contract.

    `smpl_model_dir` is an explicit, no-default body-model directory used only by
    loaders that need forward kinematics from a separate model (OMOMO, for its
    betas-based height). Loaders that don't need it ignore the argument.
    """

    @abstractmethod
    def load(self, *, model_path: Path | None, motion_path: Path,
             obj_path: Path | None, task_type: str, constants,
             motion_data_config,
             smpl_model_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray, float]:
        ...


DATASET_LOADERS: dict[str, type[MotionLoader]] = {}


def register_loader(name: str):
    """Class decorator registering a MotionLoader subclass under `name`."""
    def _inner(cls: type[MotionLoader]) -> type[MotionLoader]:
        DATASET_LOADERS[name] = cls
        return cls
    return _inner


def resolve_loader(dataset: str) -> MotionLoader:
    """Instantiate the loader registered for `dataset`."""
    if dataset not in DATASET_LOADERS:
        known = ", ".join(sorted(DATASET_LOADERS)) or "(none registered)"
        raise ValueError(f"Unknown dataset {dataset!r}. Known datasets: {known}")
    return DATASET_LOADERS[dataset]()
