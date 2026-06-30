"""Calibration : ancrer l'humain + objets au sol (robot-free ET body-free).
RawMotion → ``Calibration`` (décalages d'ancrage). Logique build dans ``build.py``, I/O .npz dans ``cache.py``."""
from .build import (CalibrationBuilder, build_calibration, foot_floor_offset,
                    object_floor_offset)
from .cache import load_calibration, save_calibration

__all__ = ["CalibrationBuilder", "build_calibration", "foot_floor_offset",
           "object_floor_offset", "save_calibration", "load_calibration"]
