"""Named PRESET configs for the ``prepare`` step (the VALUES). The schemas live in
``config_types``. Pick a preset by name, or start from one and override fields with
``dataclasses.replace``. New presets (e.g. higher density, per-robot tuning) are added to
``_PRESETS`` — the single place a public user looks for ready-made configurations."""
from __future__ import annotations

from config_types import PrepareConfig

_PRESETS: dict[str, PrepareConfig] = {
    "default": PrepareConfig(),
}


def prepare_config(name: str = "default") -> PrepareConfig:
    """Return the named ``prepare`` config preset (raises ValueError on an unknown name)."""
    try:
        return _PRESETS[name]
    except KeyError:
        known = ", ".join(sorted(_PRESETS))
        raise ValueError(f"unknown prepare preset {name!r}; known: {known}") from None
