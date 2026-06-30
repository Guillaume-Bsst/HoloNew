"""retract — applique le pas du solveur aux variables de décision. PUR, numpy-only,
pinocchio/torch-free : ``q ⊕ dv`` est délégué à ``RobotModel.integrate`` (seul détenteur des cinématiques
du free-flyer), ``object_pose ⊕ dξ`` est une exp SE(3) en pur numpy.

Convention de tangente d'objet (cohérente avec ``ContactEval.probe_jac_obj``, alignée-au-monde ``(δt,δθ)``,
LOCAL_WORLD_ALIGNED) : ``new_pos = pos + δt`` (translation monde), ``new_R = exp(δθ) · R`` (incrément de
rotation appliqué À GAUCHE, en frame monde). Quaternions wxyz ; pose d'objet ``[x,y,z,qw,qx,qy,qz]``."""
from __future__ import annotations

import numpy as np

from .contracts import Step


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Exponentielle SO(3) (Rodrigues) d'un vecteur de rotation ``w (3,)`` -> ``R (3,3)``. Utilise
    les petits angles (série Taylor) pour rester stable et différentiable près de 0."""
    w = np.asarray(w, np.float64)
    th = float(np.linalg.norm(w))
    K = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    if th < 1e-8:
        return np.eye(3) + K + 0.5 * (K @ K)
    return np.eye(3) + (np.sin(th) / th) * K + ((1.0 - np.cos(th)) / (th * th)) * (K @ K)


def quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    """Quaternion wxyz (supposé unitaire) -> matrice de rotation ``(3,3)``."""
    qw, qx, qy, qz = (float(v) for v in q)
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz),     2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy),     2 * (qy * qz + qw * qx),     1 - 2 * (qx * qx + qy * qy)],
    ])


def mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Matrice de rotation ``(3,3)`` -> quaternion wxyz unitaire (méthode Shepperd, stable)."""
    R = np.asarray(R, np.float64)
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] >= R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """Réordonne wxyz -> xyzw (convention pinocchio pour le free-flyer ``q``)."""
    q = np.asarray(q, np.float64)
    return np.array([q[1], q[2], q[3], q[0]])


def retract(q: np.ndarray, object_poses: np.ndarray, step: Step, robot) -> tuple[np.ndarray, np.ndarray]:
    """``q ⊕ dv`` via ``robot.integrate`` (free-flyer, pinocchio-free côté solve) + ``pose ⊕ dξ`` via
    exp SE(3) numpy par objet. Ne mute pas les inputs."""
    q_new = robot.integrate(np.asarray(q, np.float64), np.asarray(step.dv, np.float64))
    poses = np.array(object_poses, np.float64, copy=True)
    if step.dxi is not None and poses.shape[0] > 0:
        dxi = np.asarray(step.dxi, np.float64)
        for i in range(poses.shape[0]):
            dt, dth = dxi[i, :3], dxi[i, 3:6]
            R = so3_exp(dth) @ quat_wxyz_to_mat(poses[i, 3:7])    # exp LEFT (world frame)
            poses[i, :3] = poses[i, :3] + dt
            poses[i, 3:7] = mat_to_quat_wxyz(R)
    return q_new, poses
