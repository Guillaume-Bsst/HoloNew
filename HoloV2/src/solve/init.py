"""init — le seed des variables de décision. PUR, pinocchio/torch-free (consomme une référence
``FrameTargets`` + le ``RobotModel``, jamais l'Evaluator).

``compute_q_init`` (frame 0, idiome Holosoma) : base flottante = cible pelvis du style (position +
orientation), joints **neutres**, objets à leur pose observée — bien meilleur seed qu'une base à
l'origine. Pour G1 le lien root URDF = ``pelvis`` donc base ≡ cible pelvis directe ; un offset
root↔pelvis se composerait ICI (un seul endroit) via ``base_link``. ``warm_start`` : carry de f-1."""
from __future__ import annotations

import numpy as np

from .retract import mat_to_quat_wxyz, quat_wxyz_to_xyzw


def compute_q_init(frame_targets_0, robot, base_link: str = "pelvis") -> tuple[np.ndarray, np.ndarray]:
    """Seed f=0 : ``q = [base_pos = cible pelvis, base_quat = orient pelvis (xyzw), joints = 0]`` +
    objets ``(N,7)`` à leur pose observée (rot -> quat wxyz). ``base_link`` = lien root (G1 : pelvis)."""
    style = frame_targets_0.style
    q = np.array(robot.neutral(), np.float64, copy=True)          # base identité (xyzw) + joints 0
    try:
        idx = tuple(style.link_names).index(base_link)
    except ValueError:
        raise ValueError(
            f"base link {base_link!r} absent de StyleTargets.link_names {tuple(style.link_names)!r}")
    q[0:3] = np.asarray(style.position[idx], np.float64)          # base pos = cible pelvis
    if style.orientation is not None:
        q[3:7] = quat_wxyz_to_xyzw(np.asarray(style.orientation[idx], np.float64))  # wxyz -> xyzw

    rot = np.asarray(frame_targets_0.object_rot, np.float64)      # (N, 3, 3)
    pos = np.asarray(frame_targets_0.object_pos, np.float64)      # (N, 3)
    n = rot.shape[0]
    object_poses = np.zeros((n, 7), np.float64)
    for i in range(n):
        object_poses[i, :3] = pos[i]
        object_poses[i, 3:7] = mat_to_quat_wxyz(rot[i])           # pose objet = quat wxyz
    return q, object_poses


def warm_start(prev_q: np.ndarray, prev_poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Carry de f-1 vers f>0 : copies défensives de l'état précédent."""
    return (np.array(prev_q, np.float64, copy=True), np.array(prev_poses, np.float64, copy=True))
