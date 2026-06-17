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
    human_flr_witness: np.ndarray | None = None    # (T, N, 3) WORLD-frame floor witness (probe projected to z=0)
    g1_transport_pts: np.ndarray | None = None     # (T, M, 3) correspondence points on the solved robot
    human_idx: np.ndarray | None = None            # (M,)      human point driving each G1 point
    object_surface_local: np.ndarray | None = None # (M, 3)    object-local surface samples (object<->floor carrier)
    # Diagnostics for analysing the TEST-SOCP solve in the viewer (None elsewhere).
    solved_object_poses: np.ndarray | None = None  # (T, 7)  TEST's SOLVED object pose [qw,qx,qy,qz,x,y,z]
                                                   #         (movable/inertia) vs the reference object pose
    com: np.ndarray | None = None                  # (T, 3)  robot CoM (centroidal diagnostic)
    com_ref: np.ndarray | None = None              # (T, 3)  W^c_pos target CoM (grounded reference)
    angular_momentum: np.ndarray | None = None     # (T, 3)  centroidal angular momentum L (W^L diagnostic)
    angular_momentum_ref: np.ndarray | None = None # (T, 3)  W^L reference angular momentum (grounded target)
    foot_slip: np.ndarray | None = None            # (T,)    mean tangential foot slip at floor contacts (no-slip)
    per_frame_cost: np.ndarray | None = None       # (T,)    SQP objective value at each solved frame (solver health)
