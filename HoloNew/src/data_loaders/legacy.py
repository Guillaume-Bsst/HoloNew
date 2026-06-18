"""Legacy loaders: thin wrappers reconstructing data_path/task_name from motion_path
and delegating to the existing load_motion_data() (behaviour-preserving)."""
from __future__ import annotations

from pathlib import Path

from HoloNew.examples.robot_retarget import load_motion_data
from HoloNew.src.data_loaders.base import DATASET_TO_FORMAT, MotionLoader, register_loader


class LegacyLoader(MotionLoader):
    def __init__(self, dataset: str):
        self.dataset = dataset

    def load(self, *, model_path, motion_path, obj_path, task_type,
             constants, motion_data_config, smpl_model_dir=None):
        motion_path = Path(motion_path)
        return load_motion_data(
            task_type,
            DATASET_TO_FORMAT[self.dataset],
            motion_path.parent,
            motion_path.stem,
            constants,
            motion_data_config,
        )


@register_loader("lafan")
class LafanLoader(LegacyLoader):
    def __init__(self): super().__init__("lafan")


@register_loader("sfu")
class SfuLoader(LegacyLoader):
    def __init__(self): super().__init__("sfu")


@register_loader("climbing")
class ClimbingLoader(LegacyLoader):
    def __init__(self): super().__init__("climbing")
