"""Config TYPES for HoloV2 — the dataclass SCHEMAS (what knobs exist), grouped by step.

Kept apart from the DATA contracts (``src.contracts``, the artifacts that flow through the
pipeline) and from the VALUES (``config_values``, named presets you pick). One module per
step (``prepare`` now; ``targets``/``solve`` later)."""
from .prepare import (CalibrationConfig, CloudConfig, CorrespondenceConfig, PrepareConfig, SdfConfig)

__all__ = ["CalibrationConfig", "CloudConfig", "CorrespondenceConfig", "PrepareConfig", "SdfConfig"]
