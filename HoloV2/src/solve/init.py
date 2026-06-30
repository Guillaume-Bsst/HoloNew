"""init — seed for decision variables. PURE, pinocchio/torch-free (consumes a reference
``FrameTargets`` + the ``RobotModel``, never the Evaluator).

``compute_q_init`` (frame 0, Holosoma idiom): floating base = style pelvis target (position +
orientation), joints **neutral**, objects at their observed pose — much better seed than base at
origin. For G1 the URDF root link = ``pelvis`` so base ≡ direct pelvis target; a root↔pelvis offset
would compose HERE (single place) via ``base_link``. ``warm_start``: carry from f-1."""
from __future__ import annotations

import numpy as np

from .retract import mat_to_quat_wxyz, quat_wxyz_to_xyzw


def compute_q_init(frame_targets_0, robot, base_link: str = "pelvis") -> tuple[np.ndarray, np.ndarray]:
    """Seed f=0: ``q = [base_pos = pelvis target, base_quat = pelvis orient (xyzw), joints = 0]`` +
    objects ``(N,7)`` at their observed pose (rot -> quat wxyz). ``base_link`` = root link (G1: pelvis)."""
    style = frame_targets_0.style
    q = np.array(robot.neutral(), np.float64, copy=True)          # base identity (xyzw) + joints 0
    try:
        idx = tuple(style.link_names).index(base_link)
    except ValueError:
        raise ValueError(
            f"base link {base_link!r} absent de StyleTargets.link_names {tuple(style.link_names)!r}")
    q[0:3] = np.asarray(style.position[idx], np.float64)          # base pos = pelvis target
    if style.orientation is not None:
        q[3:7] = quat_wxyz_to_xyzw(np.asarray(style.orientation[idx], np.float64))  # wxyz -> xyzw

    rot = np.asarray(frame_targets_0.object_rot, np.float64)      # (N, 3, 3)
    pos = np.asarray(frame_targets_0.object_pos, np.float64)      # (N, 3)
    n = rot.shape[0]
    object_poses = np.zeros((n, 7), np.float64)
    for i in range(n):
        object_poses[i, :3] = pos[i]
        object_poses[i, 3:7] = mat_to_quat_wxyz(rot[i])           # object pose = quat wxyz
    return q, object_poses


def warm_start(prev_q: np.ndarray, prev_poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Carry from f-1 to f>0: defensive copies of previous state."""
    return (np.array(prev_q, np.float64, copy=True), np.array(prev_poses, np.float64, copy=True))
