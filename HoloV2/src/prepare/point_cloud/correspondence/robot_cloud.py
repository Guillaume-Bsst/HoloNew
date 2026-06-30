"""Le côté robot de la correspondance comme un ``PointCloud`` posable — crée un pont entre la
``CorrespondenceTable`` statique (point du robot = lien + décalage local) et l'opération ``pose_cloud``
partagée pour que ``solve`` puisse poser les points de contrôle du robot par FK, exactement comme
les nuages humain/objet (homogénéité).
"""
from __future__ import annotations

import numpy as np

from ...contracts import CorrespondenceTable, PointCloud


def robot_point_cloud(corr: CorrespondenceTable,
                      robot_link_names: tuple[str, ...]) -> PointCloud:
    """Les M points du robot de correspondance comme un ``PointCloud`` K=1, ``parts`` indexant dans
    ``robot_link_names`` (l'ordre FK de ``RobotModel.link_transforms``).

    ``corr.link_idx`` indexe dans ``corr.link_names`` ; on remappes par NOM à ``robot_link_names``
    pour que ``pose_cloud(cloud, *robot.link_transforms(q))`` rassemble le bon lien par point. Lève
    une exception si un lien de correspondance est absent de l'ensemble de liens du robot.
    """
    name_to_fk = {n: i for i, n in enumerate(robot_link_names)}
    try:
        corr_to_fk = np.array([name_to_fk[n] for n in corr.link_names], np.int64)   # (L_corr,)
    except KeyError as e:
        raise ValueError(
            f"correspondence link {e.args[0]!r} absent from robot link_names") from None
    parts = corr_to_fk[np.asarray(corr.link_idx)][:, None]                # (M, 1) vers l'ordre FK
    weights = np.ones((corr.n_points, 1), np.float32)                     # K=1 rigid
    offsets = np.asarray(corr.offset_local, np.float32)[:, None, :]       # (M, 1, 3) link-local
    return PointCloud(parts=parts, weights=weights, offsets=offsets)
