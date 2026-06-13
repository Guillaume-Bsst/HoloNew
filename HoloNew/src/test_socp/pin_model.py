"""pinocchio g1 model + MuJoCo<->pinocchio seam and kinematics.

Single rigid-body backend for TEST-SOCP: FK, frame Jacobians (tangent space),
point Jacobians, CoM and CoM Jacobian. MuJoCo/coal remain only for
collision/SDF. See docs/specs/2026-06-13-brick0-mujoco-to-pinocchio-design.md.
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin
import mujoco


class PinModel:
    def __init__(self, urdf_path: str):
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data = self.model.createData()
        self.joint_names = [n for n in self.model.names]

    def neutral(self) -> np.ndarray:
        return pin.neutral(self.model)

    def bind_mujoco_order(self, mj_model) -> None:
        """Map each actuated joint between MuJoCo qpos order and pinocchio q order."""
        self._pin_joint_qadr = {}
        for jid in range(2, self.model.njoints):
            name = self.model.names[jid]
            self._pin_joint_qadr[name] = self.model.joints[jid].idx_q
        self._mj_joint_qadr = {}
        for j in range(mj_model.njnt):
            if mj_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)
            self._mj_joint_qadr[name] = int(mj_model.jnt_qposadr[j])
        assert set(self._pin_joint_qadr) == set(self._mj_joint_qadr), \
            "MuJoCo and pinocchio joint name sets differ"

    def qpos_mj_to_q_pin(self, q_mj: np.ndarray) -> np.ndarray:
        """Convert MuJoCo qpos [pos(3), wxyz(4), joints(29)] to pinocchio q [pos(3), xyzw(4), joints(29)]."""
        q = np.zeros(self.model.nq)
        q[0:3] = q_mj[0:3]
        q[3:7] = q_mj[[4, 5, 6, 3]]           # wxyz -> xyzw
        for name, pin_adr in self._pin_joint_qadr.items():
            q[pin_adr] = q_mj[self._mj_joint_qadr[name]]
        return q

    def q_pin_to_qpos_mj(self, q_pin: np.ndarray) -> np.ndarray:
        """Convert pinocchio q [pos(3), xyzw(4), joints(29)] to MuJoCo qpos [pos(3), wxyz(4), joints(29)]."""
        q = np.zeros(7 + len(self._mj_joint_qadr))
        q[0:3] = q_pin[0:3]
        q[3:7] = q_pin[[6, 3, 4, 5]]           # xyzw -> wxyz
        for name, pin_adr in self._pin_joint_qadr.items():
            q[self._mj_joint_qadr[name]] = q_pin[pin_adr]
        return q
