"""pinocchio g1 model + MuJoCo<->pinocchio seam and kinematics.

Single rigid-body backend for TEST-SOCP: FK, frame Jacobians (tangent space),
point Jacobians, CoM and CoM Jacobian. MuJoCo/coal remain only for
collision/SDF. See docs/specs/2026-06-13-brick0-mujoco-to-pinocchio-design.md.
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin


class PinModel:
    def __init__(self, urdf_path: str):
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data = self.model.createData()
        self.joint_names = [n for n in self.model.names]

    def neutral(self) -> np.ndarray:
        return pin.neutral(self.model)
