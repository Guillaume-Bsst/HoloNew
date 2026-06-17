"""Turn a channels mapping into a wide table: time column first, one column per channel."""
from __future__ import annotations

import numpy as np


def to_columns(time: np.ndarray, channels: dict[str, np.ndarray]):
    """(time, channels) -> (header, table).

    header: ["time", *channel names]. table: (T, 1 + len(channels)) float array.
    Every channel must be 1-D of length T == len(time); raises ValueError otherwise.
    """
    time = np.asarray(time, dtype=float).ravel()
    T = time.shape[0]
    names = list(channels.keys())
    cols = [time]
    for n in names:
        a = np.asarray(channels[n], dtype=float)
        if a.ndim != 1 or a.shape[0] != T:
            raise ValueError(f"channel {n!r} must be 1-D length {T}, got shape {a.shape}")
        cols.append(a)
    header = ["time"] + names
    table = np.column_stack(cols)
    return header, table
