"""Structured output of retarget_motion: qpos trajectory + per-frame stage data.

Keys in `stages` match StageSpec.key in src/stages.py. 'socp' is the final robot
qpos sequence; skeleton stages ('mapped', 'in_object') hold (T, J, 3) points.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RetargetResult:
    qpos: np.ndarray                                            # (T, 7+dof[+7])
    stages: dict[str, np.ndarray] = field(default_factory=dict)
    cost: float = 0.0
