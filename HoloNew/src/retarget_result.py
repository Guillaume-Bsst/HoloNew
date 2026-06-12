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
    # Interaction data (TEST-SOCP only; None elsewhere). T-length, frame-aligned.
    human_probe_pts: np.ndarray | None = None      # (T, N, 3) SMPL-X probe world points (Grounded)
    human_obj_dist: np.ndarray | None = None       # (T, N)    signed dist to object (SDF)
    human_flr_dist: np.ndarray | None = None       # (T, N)    signed dist to floor (analytic)
    human_witness: np.ndarray | None = None        # (T, N, 3) object-local witness for the object channel
    g1_transport_pts: np.ndarray | None = None     # (T, M, 3) correspondence points on the solved robot
    human_idx: np.ndarray | None = None            # (M,)      human point driving each G1 point
