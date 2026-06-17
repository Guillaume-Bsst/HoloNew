"""Reduce each per-frame channel to global scalars and write a JSON summary.

v1 uses generic reductions (mean / rms / min / max). Later increments replace these
with the canonical 7-family scoreboard as the *_series metrics land.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def reduce_channel(a: np.ndarray) -> dict[str, float]:
    a = np.asarray(a, dtype=float).ravel()
    if a.size == 0:
        return {"mean": 0.0, "rms": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(a)),
        "rms": float(np.sqrt(np.mean(np.square(a)))),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def write_summary(path, channels: dict[str, np.ndarray],
                  scoreboard: dict | None = None) -> dict:
    """Write {"scoreboard": canonical 7-family scalars, "channels": per-channel stats}.

    ``scoreboard`` is the canonical headline (see export.scoreboard); ``channels`` keeps
    the generic per-channel mean/rms/min/max for quick per-signal inspection.
    """
    summary = {"scoreboard": scoreboard or {},
               "channels": {name: reduce_channel(arr) for name, arr in channels.items()}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
    return summary
