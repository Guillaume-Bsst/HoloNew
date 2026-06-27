"""Config VALUES for the ``prepare`` step — the named entry point that hands back a ready
``PrepareConfig``. The schema lives in ``config_types``; this is where future presets (higher
density, per-robot tuning) or a CLI front-end attach. Start from the default and override fields
with ``dataclasses.replace`` for a one-off."""
from __future__ import annotations

from config_types import PrepareConfig


def default_prepare_config() -> PrepareConfig:
    """The default prepare config (the single named entry point; presets/CLI attach here later)."""
    return PrepareConfig()
