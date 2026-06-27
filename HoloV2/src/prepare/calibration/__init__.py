"""Calibration: ground the human onto the floor + characterise the subject (robot-free stature).
SceneSpec/RawMotion -> ``Calibration``. Logic lives in ``build.py``."""
from .build import (CalibrationBuilder, build_calibration, foot_floor_offset,
                    human_stature, object_floor_offset)

__all__ = ["CalibrationBuilder", "build_calibration", "human_stature", "foot_floor_offset",
           "object_floor_offset"]
