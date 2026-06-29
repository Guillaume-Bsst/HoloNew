"""The robot side of the correspondence as a posable PointCloud — bridges the static
``CorrespondenceTable`` (robot point = link + local offset) to the shared ``pose_cloud`` op so
``solve`` can pose the robot control points by FK, exactly like the human/object clouds (homogeneity).
"""
from __future__ import annotations

import numpy as np

from ...contracts import CorrespondenceTable, PointCloud


def robot_point_cloud(corr: CorrespondenceTable,
                      robot_link_names: tuple[str, ...]) -> PointCloud:
    """The M correspondence robot points as a K=1 ``PointCloud``, ``parts`` indexing into
    ``robot_link_names`` (the FK order of ``RobotModel.link_transforms``).

    ``corr.link_idx`` indexes into ``corr.link_names``; we remap by NAME to ``robot_link_names`` so
    ``pose_cloud(cloud, *robot.link_transforms(q))`` gathers the right link per point. Raises if a
    correspondence link is absent from the robot's link set.
    """
    name_to_fk = {n: i for i, n in enumerate(robot_link_names)}
    try:
        corr_to_fk = np.array([name_to_fk[n] for n in corr.link_names], np.int64)   # (L_corr,)
    except KeyError as e:
        raise ValueError(
            f"correspondence link {e.args[0]!r} absent from robot link_names") from None
    parts = corr_to_fk[np.asarray(corr.link_idx)][:, None]                # (M, 1) into FK order
    weights = np.ones((corr.n_points, 1), np.float32)                     # K=1 rigid
    offsets = np.asarray(corr.offset_local, np.float32)[:, None, :]       # (M, 1, 3) link-local
    return PointCloud(parts=parts, weights=weights, offsets=offsets)
