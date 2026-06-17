"""SignalContext: extra inputs producers need beyond the RetargetResult.

Some per-frame signals (e.g. smoothness) need the frame timestep and the actuated
joint count / names, which live on the retargeter, not the result. The CLI fills these
in; producers that don't need them ignore the context. ``dof`` is intentionally None by
default so qpos-derived producers stay off unless a caller explicitly provides the
actuated-joint count (avoids misreading trailing object DOFs as joints).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SignalContext:
    dt: float = 1.0 / 30.0
    dof: int | None = None
    joint_names: list[str] | None = None
    # Limited-joint ranges for the effort family: columns into the dof joint block plus
    # their bounds and names. None (default) -> the effort producer stays off.
    joint_limit_cols: np.ndarray | None = None
    joint_limit_lower: np.ndarray | None = None
    joint_limit_upper: np.ndarray | None = None
    joint_limit_names: list[str] | None = None
    # Pre-computed (T,) channels injected by the CLI for families that need the
    # retargeter (FK / GMR-grounded reference): tracking, style, contacts. Keeps those
    # heavy, rt-dependent computations out of the pure producers.
    extra_channels: dict[str, np.ndarray] | None = None
    # Per-probe body-segment label (N,) + the segment-name list (indexed by label), so
    # the human SDF distances aggregate per body part instead of one column per probe.
    # None -> the distance producer falls back to positional per-probe channels.
    probe_segments: np.ndarray | None = None
    probe_segment_names: list[str] | None = None
