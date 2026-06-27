"""Config VALUES for HoloV2 — the named entry points (ready-made configurations), one module per step.

The schemas they instantiate live in ``config_types``. The orchestrators (runner / app /
viz) and tests call an entry point here; the ``src`` builders only depend on the TYPES."""
from .prepare import default_prepare_config

__all__ = ["default_prepare_config"]
