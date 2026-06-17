"""Write a (header, table) pair as a CSV PlotJuggler reads natively (numpy only)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def write_csv(path, header: list[str], table: np.ndarray) -> Path:
    """Write CSV: first line = comma-joined header, then one row per frame."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(table, dtype=float), delimiter=",",
               header=",".join(header), comments="", fmt="%.9g")
    return path
