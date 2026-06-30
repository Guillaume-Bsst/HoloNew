"""Conventions de frame pour la couche load — UNE SEULE définition, partagée par le modèle de corps
et les chargeurs d'objets. Les corps SMPL et les captures HODome/HOI-M3 sont natifs Y-up ; le monde
canonique est Z-up. ``YUP_TO_ZUP`` est la seule rotation qui cartographie entre eux, et ``object_pose_zup``
est la conversion de pose d'objet rigide qui l'utilise (la convention vit donc en un seul lieu, pas dans
chaque chargeur).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

# Y-up -> Z-up comme rotation propre Rx(+90deg) : (x,y,z) -> (x,-z,y). Un simple échange d'axes y<->z
# est une réflexion (det -1) qui miroir le corps et inverse l'orientation des faces ; la rotation la préserve.
YUP_TO_ZUP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def object_pose_zup(R_seq: np.ndarray, T_seq: np.ndarray) -> np.ndarray:
    """Pose d'objet rigide par frame (``R (T,3,3)``, ``T (T,3)``) dans la capture Y-up -> ``(T,7)``
    pose monde ``[x,y,z,qw,qx,qy,qz]`` en Z-up. Q multiplie à gauche les deux (``Q T``, ``Q R``) ;
    le mesh reste dans son frame local. Partagé par les chargeurs d'objets Y-up (HODome, HOI-M3)."""
    Tz = np.asarray(T_seq, np.float64) @ YUP_TO_ZUP.T
    Rz = YUP_TO_ZUP @ np.asarray(R_seq, np.float64)            # Q R (object_R utilisé directement)
    quat_wxyz = R.from_matrix(Rz).as_quat()[:, [3, 0, 1, 2]]
    return np.concatenate([Tz, quat_wxyz], axis=1).astype(np.float32)
