"""Run manifest: which metric families were exported, how many channels, what failed.

The CLI builds every family inside a try/except so one missing prerequisite never
crashes the export — but that means a family can silently vanish from the CSV. The
manifest makes that explicit, so a comparison pipeline can assert the families it needs
are actually present instead of silently comparing incomplete runs.
"""
from __future__ import annotations

import json
from pathlib import Path

# Channel-name prefix -> family label. Order matters only for readability.
_FAMILIES = [
    ("tracking", "tracking/"),
    ("contacts", "contacts/"),
    ("smoothness", "smoothness/"),
    ("effort", "effort/"),
    ("dynamics", "dynamics/"),
    ("roots", "roots/"),
    ("solver", "solver/"),
    ("diagnostics", "diag/"),
]


def build_manifest(channels: dict, errors: dict[str, str] | None = None) -> dict:
    """{family: {present, n_channels, error}} + totals, from the final channel set."""
    errors = errors or {}
    families = {}
    for name, prefix in _FAMILIES:
        n = sum(1 for k in channels if k.startswith(prefix))
        families[name] = {"present": n > 0, "n_channels": n,
                          "error": errors.get(name)}
    # Surface any error whose family produced no channels but was attempted.
    for fam, msg in errors.items():
        families.setdefault(fam, {"present": False, "n_channels": 0, "error": msg})
        families[fam]["error"] = msg
    return {"families": families,
            "n_channels_total": len(channels),
            "missing": [f for f, m in families.items() if not m["present"]]}


def write_manifest(path, channels: dict, errors: dict[str, str] | None = None) -> dict:
    manifest = build_manifest(channels, errors)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return manifest
