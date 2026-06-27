"""Config VALUES for HoloV2 — named presets (ready-made configurations), one module per step.

The schemas they instantiate live in ``config_types``. The orchestrators (runner / app /
viz) and tests pick a preset here; the ``src`` builders only depend on the TYPES."""
from .prepare import prepare_config

__all__ = ["prepare_config"]
