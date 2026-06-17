"""SignalContext: extra inputs producers need beyond the RetargetResult.

Some per-frame signals (e.g. smoothness) need the frame timestep and the actuated
joint count / names, which live on the retargeter, not the result. The CLI fills these
in; producers that don't need them ignore the context. ``dof`` is intentionally None by
default so qpos-derived producers stay off unless a caller explicitly provides the
actuated-joint count (avoids misreading trailing object DOFs as joints).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SignalContext:
    dt: float = 1.0 / 30.0
    dof: int | None = None
    joint_names: list[str] | None = None
