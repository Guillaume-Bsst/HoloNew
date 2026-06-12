"""Place the fixed human->G1 correspondence onto a posed robot.

The ~0-compute online path: given the robot's link world transforms for a frame
and the correspondence (link index + link-local offset per G1 point), gather every
point onto the robot with one batched rigid transform. No optimal transport, no
contact recompute. Mirrors test_pipe's proximity transport.
"""
from __future__ import annotations

import numpy as np


def link_world_transforms(urdf, qpos: np.ndarray, link_names) -> dict[str, np.ndarray]:
    """World 4x4 transform per link name: base pose (qpos[:7]) composed with URDF FK.

    qpos: full robot config — [0:3] base pos, [3:7] base quat wxyz, [7:7+ndof] joints,
    in the same order the URDF's actuated joints expect (as the viewer's update_cfg).
    """
    qpos = np.asarray(qpos, dtype=np.float64)
    ndof = len(urdf.actuated_joints)
    urdf.update_cfg(qpos[7:7 + ndof])

    w, x, y, z = qpos[3:7]
    n = float(np.linalg.norm(qpos[3:7])) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    T_base = np.eye(4)
    T_base[:3, :3] = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    T_base[:3, 3] = qpos[0:3]
    return {ln: T_base @ np.asarray(urdf.get_transform(ln)) for ln in set(link_names)}


def transported_points(link_world_transforms: dict[str, np.ndarray],
                       link_idx: np.ndarray, offset_local: np.ndarray,
                       link_names) -> np.ndarray:
    """(N,3) world positions of the correspondence points placed on the posed robot."""
    link_idx = np.asarray(link_idx)
    offset_local = np.asarray(offset_local, dtype=np.float64)
    out = np.empty((len(link_idx), 3), dtype=np.float64)
    for li in np.unique(link_idx):
        T = link_world_transforms[link_names[int(li)]]
        sel = link_idx == li
        out[sel] = offset_local[sel] @ T[:3, :3].T + T[:3, 3]
    return out.astype(np.float32)
