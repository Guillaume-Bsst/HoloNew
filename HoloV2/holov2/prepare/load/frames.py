"""Frame conventions for the load layer — ONE definition, shared by the body model and the object
loaders. SMPL bodies and the HODome/HOI-M3 captures are native Y-up; the canonical world is Z-up.
``YUP_TO_ZUP`` is the single rotation that maps between them, and ``object_pose_zup`` is the rigid
object-pose conversion that uses it (so the convention lives in one place, not in every loader).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

# Y-up -> Z-up as a proper rotation Rx(+90deg): (x,y,z) -> (x,-z,y). A bare y<->z axis swap is a
# reflection (det -1) that mirrors the body and flips face winding; the rotation preserves it.
YUP_TO_ZUP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def object_pose_zup(R_seq: np.ndarray, T_seq: np.ndarray) -> np.ndarray:
    """Per-frame rigid object pose (``R (T,3,3)``, ``T (T,3)``) in the Y-up capture -> ``(T,7)``
    world pose ``[x,y,z,qw,qx,qy,qz]`` in Z-up. Q left-multiplies both (``Q T``, ``Q R``); the
    mesh stays in its local frame. Shared by the Y-up object loaders (HODome, HOI-M3)."""
    Tz = np.asarray(T_seq, np.float64) @ YUP_TO_ZUP.T
    Rz = YUP_TO_ZUP @ np.asarray(R_seq, np.float64)            # Q R (object_R used directly)
    quat_wxyz = R.from_matrix(Rz).as_quat()[:, [3, 0, 1, 2]]
    return np.concatenate([Tz, quat_wxyz], axis=1).astype(np.float32)
