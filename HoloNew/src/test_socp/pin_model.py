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

    # ------------------------------------------------------------------
    # FK helpers
    # ------------------------------------------------------------------

    def _fk(self, q_pin: np.ndarray) -> None:
        """Run forward kinematics and update all frame placements.

        The input is normalized first so FK and frame_translational_jacobian
        operate on exactly the same configuration at non-unit quaternions
        (computeJointJacobians normalizes internally, forwardKinematics does not).
        """
        pin.forwardKinematics(self.model, self.data, pin.normalize(self.model, q_pin))
        pin.updateFramePlacements(self.model, self.data)

    def _frame_id(self, body_name: str) -> int:
        """Return the pinocchio frame id for a link name; raise if not found."""
        fid = self.model.getFrameId(body_name)
        if fid >= self.model.nframes:
            raise ValueError(f"Frame '{body_name}' not found in pinocchio model")
        return fid

    def body_position(self, q_pin: np.ndarray, body_name: str) -> np.ndarray:
        """World position of body_name at pinocchio configuration q_pin.

        Args:
            q_pin: Pinocchio configuration vector (length nq).
            body_name: Link name matching a URDF link / pinocchio frame.

        Returns:
            Position array of shape (3,).
        """
        self._fk(q_pin)
        return np.array(self.data.oMf[self._frame_id(body_name)].translation)

    def body_rotation(self, q_pin: np.ndarray, body_name: str) -> np.ndarray:
        """World rotation matrix of body_name at pinocchio configuration q_pin.

        Args:
            q_pin: Pinocchio configuration vector (length nq).
            body_name: Link name matching a URDF link / pinocchio frame.

        Returns:
            Rotation matrix of shape (3, 3).
        """
        self._fk(q_pin)
        return np.array(self.data.oMf[self._frame_id(body_name)].rotation)

    # ------------------------------------------------------------------
    # Jacobians
    # ------------------------------------------------------------------

    def frame_translational_jacobian(self, q_pin: np.ndarray, body_name: str) -> np.ndarray:
        """World-aligned translational Jacobian of a frame, in pinocchio tangent space.

        Computes J such that dp = J @ v, where v is the pinocchio tangent vector
        (nv-dimensional) and dp is the world-frame velocity of body_name's origin.
        Uses LOCAL_WORLD_ALIGNED reference frame: world-axis-aligned axes located
        at the frame origin, which matches the finite difference of world position
        via pin.integrate.

        The input is normalized so that computeJointJacobians and forwardKinematics
        operate on exactly the same configuration (computeJointJacobians normalizes
        unit quaternions internally, while forwardKinematics does not).

        Args:
            q_pin: Pinocchio configuration vector (length nq).
            body_name: Link name matching a URDF link / pinocchio frame.

        Returns:
            Translational Jacobian of shape (3, nv) in pinocchio v order.
        """
        q = pin.normalize(self.model, q_pin)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        fid = self._frame_id(body_name)
        J6 = pin.getFrameJacobian(self.model, self.data, fid, pin.LOCAL_WORLD_ALIGNED)
        return np.asarray(J6[0:3, :])

    def point_translational_jacobian(
        self,
        q_pin: np.ndarray,
        body_name: str,
        offset_local: np.ndarray,
    ) -> np.ndarray:
        """World-aligned translational Jacobian of a point fixed on a link, pinocchio v order.

        The point is defined in the link's local frame as ``offset_local``.  Its
        world position is ``p_frame + R @ offset_local`` and its velocity is

            J_point = J_trans - skew(R @ offset_local) @ J_ang

        where J_trans (rows 0:3) and J_ang (rows 3:6) are taken from the
        LOCAL_WORLD_ALIGNED 6xnv frame Jacobian.  LOCAL_WORLD_ALIGNED expresses
        both the translational and angular parts in world-aligned axes, so the
        skew-product correction is directly in the world frame.

        Args:
            q_pin: Pinocchio configuration vector (length nq).
            body_name: Link name matching a URDF link / pinocchio frame.
            offset_local: Offset from the frame origin in the local body frame, shape (3,).

        Returns:
            Translational Jacobian of shape (3, nv) in pinocchio v order.
        """
        q = pin.normalize(self.model, q_pin)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        fid = self._frame_id(body_name)
        J6 = pin.getFrameJacobian(self.model, self.data, fid, pin.LOCAL_WORLD_ALIGNED)
        # Rotate offset to world frame.
        R = np.asarray(self.data.oMf[fid].rotation)
        rp = R @ np.asarray(offset_local, dtype=float)
        # skew-symmetric matrix of rp so that skew(rp) @ w = rp x w.
        skew = np.array([
            [0.0,   -rp[2],  rp[1]],
            [rp[2],  0.0,   -rp[0]],
            [-rp[1], rp[0],  0.0],
        ])
        return np.asarray(J6[0:3, :]) - skew @ np.asarray(J6[3:6, :])

    # ------------------------------------------------------------------
    # Center of mass
    # ------------------------------------------------------------------

    def com(self, q_pin: np.ndarray) -> np.ndarray:
        """Whole-body center of mass in the world frame.

        Args:
            q_pin: Pinocchio configuration vector (length nq).

        Returns:
            CoM position of shape (3,).
        """
        return np.array(
            pin.centerOfMass(self.model, self.data, pin.normalize(self.model, q_pin))
        )

    def com_jacobian(self, q_pin: np.ndarray) -> np.ndarray:
        """Jacobian of the whole-body CoM with respect to the pinocchio tangent vector.

        Returns J such that d(CoM) = J @ v, where v is the pinocchio tangent
        vector (nv-dimensional).  The Jacobian is computed internally by
        ``pin.jacobianCenterOfMass`` which runs its own FK pass.

        Args:
            q_pin: Pinocchio configuration vector (length nq).

        Returns:
            CoM Jacobian of shape (3, nv) in pinocchio v order.
        """
        return np.asarray(
            pin.jacobianCenterOfMass(
                self.model, self.data, pin.normalize(self.model, q_pin)
            )
        )
