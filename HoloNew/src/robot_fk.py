"""Forward kinematics to read robot link world poses from a qpos trajectory.

Mirrors how the GMR solve reads ``data.xpos`` / ``data.xquat``: load the robot
MJCF, set qpos, run ``mj_kinematics``, and read the world position (and
orientation) of each requested body. Used by the stage viewer to draw the solved
robot as a skeleton (joints + bones, plus per-link orientation frames) underneath
the URDF mesh, the same way test_pipe renders the G1 skeleton.
"""
from __future__ import annotations

import mujoco
import numpy as np


def robot_link_poses(mjcf_path: str, body_names: list[str], qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World positions and orientations of ``body_names`` over a qpos trajectory.

    Args:
        mjcf_path: MuJoCo model whose qpos layout matches ``qpos`` (the g1 .xml,
            free base + actuated joints).
        body_names: MuJoCo body names to read, in the desired output order.
        qpos: ``(T, nq+)`` trajectory; only the first ``model.nq`` columns are
            used (any trailing object pose is ignored).

    Returns:
        ``(pos (T, K, 3), quat (T, K, 4))`` float32; quat is wxyz (MuJoCo order).
    """
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    bids = []
    for name in body_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid == -1:
            raise ValueError(f"body '{name}' not found in {mjcf_path}")
        bids.append(bid)

    nq = model.nq
    qpos = np.asarray(qpos)
    T = qpos.shape[0]
    pos = np.empty((T, len(bids), 3), np.float32)
    quat = np.empty((T, len(bids), 4), np.float32)
    for t in range(T):
        data.qpos[:] = qpos[t, :nq]
        mujoco.mj_kinematics(model, data)
        for i, bid in enumerate(bids):
            pos[t, i] = data.xpos[bid]
            quat[t, i] = data.xquat[bid]   # MuJoCo xquat is wxyz
    return pos, quat


def robot_link_positions(mjcf_path: str, body_names: list[str], qpos: np.ndarray) -> np.ndarray:
    """World positions only; thin wrapper over :func:`robot_link_poses`."""
    return robot_link_poses(mjcf_path, body_names, qpos)[0]
