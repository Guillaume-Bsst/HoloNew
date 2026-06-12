"""Forward kinematics to read robot link world positions from a qpos trajectory.

Mirrors how the GMR solve reads ``data.xpos``: load the robot MJCF, set qpos,
run ``mj_kinematics``, and read the world position of each requested body. Used
by the stage viewer to draw the solved robot as a skeleton (joints + bones)
underneath the URDF mesh, the same way test_pipe renders the G1 skeleton.
"""
from __future__ import annotations

import mujoco
import numpy as np


def robot_link_positions(mjcf_path: str, body_names: list[str], qpos: np.ndarray) -> np.ndarray:
    """World positions of ``body_names`` over a qpos trajectory.

    Args:
        mjcf_path: MuJoCo model whose qpos layout matches ``qpos`` (the g1 .xml,
            free base + actuated joints).
        body_names: MuJoCo body names to read, in the desired output order.
        qpos: ``(T, nq+)`` trajectory; only the first ``model.nq`` columns are
            used (any trailing object pose is ignored).

    Returns:
        ``(T, len(body_names), 3)`` float32 world positions.
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
    out = np.empty((qpos.shape[0], len(bids), 3), np.float32)
    for t in range(qpos.shape[0]):
        data.qpos[:] = qpos[t, :nq]
        mujoco.mj_kinematics(model, data)
        for i, bid in enumerate(bids):
            out[t, i] = data.xpos[bid]
    return out
