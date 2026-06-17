"""RunSignals: turn a RetargetResult into a frame-aligned (time, channels) bundle.

The only component that knows RetargetResult. Runs the producer registry, enforces
that every emitted channel is frame-aligned to qpos (length T), and builds the time
axis from fps.
"""
from __future__ import annotations

import numpy as np

from .producers import run_all


class RunSignals:
    def __init__(self, result, fps: float = 30.0):
        self.T = int(np.asarray(result.qpos).shape[0])
        self.time = np.arange(self.T, dtype=float) / float(fps)
        self.channels: dict[str, np.ndarray] = {}
        for name, arr in run_all(result).items():
            arr = np.asarray(arr)
            if arr.shape[0] < self.T:
                raise ValueError(
                    f"channel {name!r}: leading axis {arr.shape[0]} < T={self.T}")
            # Reference channels (e.g. *_ref) carry the full source trajectory; when
            # the solve is truncated (max_frames) they run longer than T but stay
            # frame-aligned from frame 0, so prefix-align them to T.
            self.channels[name] = arr[:self.T]
