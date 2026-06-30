"""Robot kinematics from a URDF — robot-agnostic ``RobotModel`` (pinocchio free-flyer FK).

Generic across humanoids: the robot identity (URDF, link names, dof) comes from the ``RobotSpec``.
The only robot-SPECIFIC data lives here as a name-keyed table — ``CORRESPONDENCE_REST_POSE`` — so
adding a new robot is a data entry, never a change to the generic surface/OT/transport code.
"""
from __future__ import annotations

import numpy as np

from ..contracts import RobotSpec

# Rest pose used when sampling a robot's surface for the correspondence build: a T-pose-like config
# (limbs spread) that matches the SMPL-X rest and keeps the per-segment limb clouds separated for the
# OT. Joint angles (rad) by URDF joint name; unset joints default to 0. Keyed by RobotSpec.name —
# only G1 is defined for now; a new robot adds its own entry, nothing else changes.
CORRESPONDENCE_REST_POSE: dict[str, dict[str, float]] = {
    "g1": {
        "left_shoulder_roll_joint": 1.5708, "right_shoulder_roll_joint": -1.5708,
        "left_elbow_joint": 1.55, "right_elbow_joint": 1.55,
    },
}


def correspondence_rest_angles(robot_name: str) -> dict[str, float]:
    """Correspondence-build rest-pose joint angles for ``robot_name`` (raises if undefined)."""
    try:
        return CORRESPONDENCE_REST_POSE[robot_name]
    except KeyError:
        raise ValueError(f"no correspondence rest pose for robot {robot_name!r} — add an entry to "
                         f"CORRESPONDENCE_REST_POSE") from None


class PinRobot:
    """``RobotModel`` backed by pinocchio (free-flyer). World link transforms + analytic frame
    Jacobians (LOCAL_WORLD_ALIGNED). Config ``q = [pelvis(7: pos + quat xyzw), joints]`` (pinocchio
    order); tangent ``v`` dim ``nv = 6 + n_joints``. Ported from HoloNew ``test_socp/pin_model.py``."""

    def __init__(self, spec: RobotSpec) -> None:
        import pinocchio as pin
        self._pin = pin
        self.model = pin.buildModelFromUrdf(str(spec.urdf_path), pin.JointModelFreeFlyer())
        self.data = self.model.createData()
        self.nq: int = int(self.model.nq)
        self.nv: int = int(self.model.nv)
        self.dof: int = self.nv - 6
        # BODY frames = URDF links; keep their names + ids (transport/remap index by NAME).
        self.link_names: tuple[str, ...] = tuple(
            f.name for f in self.model.frames if f.type == pin.FrameType.BODY)
        self._fids = {name: self.model.getFrameId(name) for name in self.link_names}
        # actuated joint name -> idx_q / idx_v (joints 2..njoints; joint 1 is the free-flyer)
        self._joint_qadr = {self.model.names[j]: self.model.joints[j].idx_q
                            for j in range(2, self.model.njoints)}

    def neutral(self) -> np.ndarray:
        return np.asarray(self._pin.neutral(self.model), np.float64)

    def integrate(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.asarray(self._pin.integrate(self.model, np.asarray(q, np.float64),
                                              np.asarray(v, np.float64)), np.float64)

    def config_from_angles(self, angles: dict) -> np.ndarray:
        """Neutral base + named actuated joint angles -> q (nq,). Joints absent default to 0."""
        q = self.neutral()
        for name, a in angles.items():
            if name in self._joint_qadr:
                q[self._joint_qadr[name]] = float(a)
        return q

    def _fk(self, q: np.ndarray) -> None:
        pin = self._pin
        pin.forwardKinematics(self.model, self.data, pin.normalize(self.model, np.asarray(q, np.float64)))
        pin.updateFramePlacements(self.model, self.data)

    def link_transforms(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """WORLD link transforms for config ``q`` (nq,). Returns ``(rot (L,3,3), pos (L,3))`` aligned to ``link_names``."""
        self._fk(q)
        n = len(self.link_names)
        rot = np.empty((n, 3, 3)); pos = np.empty((n, 3))
        for i, name in enumerate(self.link_names):
            oMf = self.data.oMf[self._fids[name]]
            rot[i] = np.asarray(oMf.rotation); pos[i] = np.asarray(oMf.translation)
        return rot, pos

    def rest_transforms(self) -> tuple[np.ndarray, np.ndarray]:
        """Link transforms at the neutral free-flyer configuration (identity base)."""
        return self.link_transforms(self.neutral())

    def link_jacobians(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """World transforms + LOCAL_WORLD_ALIGNED translational/angular frame Jacobians per link.
        ``dp_world = jac_lin @ v``, ``omega_world = jac_ang @ v`` (v in pinocchio tangent order).
        Returns ``(rot (L,3,3), pos (L,3), jac_lin (L,3,nv), jac_ang (L,3,nv))``."""
        pin = self._pin
        qn = pin.normalize(self.model, np.asarray(q, np.float64))
        pin.computeJointJacobians(self.model, self.data, qn)
        pin.updateFramePlacements(self.model, self.data)
        n = len(self.link_names); nv = self.nv
        rot = np.empty((n, 3, 3)); pos = np.empty((n, 3))
        jac_lin = np.empty((n, 3, nv)); jac_ang = np.empty((n, 3, nv))
        for i, name in enumerate(self.link_names):
            fid = self._fids[name]
            oMf = self.data.oMf[fid]
            rot[i] = np.asarray(oMf.rotation); pos[i] = np.asarray(oMf.translation)
            J6 = np.asarray(pin.getFrameJacobian(self.model, self.data, fid, pin.LOCAL_WORLD_ALIGNED))
            jac_lin[i] = J6[0:3, :]; jac_ang[i] = J6[3:6, :]
        return rot, pos, jac_lin, jac_ang

    def joint_pos_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Actuated joint position limits from the URDF (rad), the joint slice of the free-flyer
        config (q[:7] is the base): ``(lower (dof,), upper (dof,))``."""
        lo = np.asarray(self.model.lowerPositionLimit, np.float64)[7:7 + self.dof]
        hi = np.asarray(self.model.upperPositionLimit, np.float64)[7:7 + self.dof]
        return lo, hi


def build_robot_model(spec: RobotSpec) -> PinRobot:
    """Build the pinocchio ``RobotModel`` for ``spec`` (FK + Jacobians, no meshes)."""
    return PinRobot(spec)
