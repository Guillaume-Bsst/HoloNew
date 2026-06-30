"""Cinématique robot à partir d'un URDF — ``RobotModel`` agnostique du robot (FK free-flyer pinocchio).

Générique pour les humanoides : l'identité du robot (URDF, noms de liens, dof) provient du ``RobotSpec``.
Les seules données spécifiques au ROBOT vivent ici dans une table indexée par nom —
``CORRESPONDENCE_REST_POSE`` — donc ajouter un robot est une saisie de données, jamais un changement
au code générique surface/OT/transport.
"""
from __future__ import annotations

import numpy as np

from ..contracts import RobotSpec

# Pose de repos utilisée lors de l'échantillonnage de la surface d'un robot pour la construction de
# correspondance : une configuration ressemblant à une T-pose (membres écartés) qui correspond au
# repos SMPL-X et garde les nuages de membres par segment séparés pour l'OT. Angles de joints (rad)
# par nom de joint URDF ; les joints non définis par défaut à 0. Indexé par RobotSpec.name —
# seul G1 est défini pour l'instant ; un nouveau robot ajoute sa propre entrée, rien d'autre ne change.
CORRESPONDENCE_REST_POSE: dict[str, dict[str, float]] = {
    "g1": {
        "left_shoulder_roll_joint": 1.5708, "right_shoulder_roll_joint": -1.5708,
        "left_elbow_joint": 1.55, "right_elbow_joint": 1.55,
    },
}


def correspondence_rest_angles(robot_name: str) -> dict[str, float]:
    """Angles de joints pose-de-repos pour la construction de correspondance de ``robot_name`` (lève si non défini)."""
    try:
        return CORRESPONDENCE_REST_POSE[robot_name]
    except KeyError:
        raise ValueError(f"no correspondence rest pose for robot {robot_name!r} — add an entry to "
                         f"CORRESPONDENCE_REST_POSE") from None


class PinRobot:
    """``RobotModel`` appuyé par pinocchio (free-flyer). Transformations de liens du monde + Jacobiens
    de frame analytiques (LOCAL_WORLD_ALIGNED). Config ``q = [pelvis(7: pos + quat xyzw), joints]``
    (ordre pinocchio); tangent ``v`` de dimension ``nv = 6 + n_joints``. Porté de HoloNew
    ``test_socp/pin_model.py``."""

    def __init__(self, spec: RobotSpec) -> None:
        import pinocchio as pin
        self._pin = pin
        self.model = pin.buildModelFromUrdf(str(spec.urdf_path), pin.JointModelFreeFlyer())
        self.data = self.model.createData()
        self.nq: int = int(self.model.nq)
        self.nv: int = int(self.model.nv)
        self.dof: int = self.nv - 6
        # Frames BODY = liens URDF ; garder leurs noms + ids (transport/remap index par NAME).
        self.link_names: tuple[str, ...] = tuple(
            f.name for f in self.model.frames if f.type == pin.FrameType.BODY)
        self._fids = {name: self.model.getFrameId(name) for name in self.link_names}
        # nom joint actualisé -> idx_q / idx_v (joints 2..njoints; joint 1 est le free-flyer)
        self._joint_qadr = {self.model.names[j]: self.model.joints[j].idx_q
                            for j in range(2, self.model.njoints)}

    def neutral(self) -> np.ndarray:
        return np.asarray(self._pin.neutral(self.model), np.float64)

    def integrate(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.asarray(self._pin.integrate(self.model, np.asarray(q, np.float64),
                                              np.asarray(v, np.float64)), np.float64)

    def config_from_angles(self, angles: dict) -> np.ndarray:
        """Base neutre + angles de joints actualisés nommés -> q (nq,). Les joints absents par défaut à 0."""
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
        """Transformations de liens du MONDE pour la config ``q`` (nq,). Retourne ``(rot (L,3,3), pos (L,3))`` alignés à ``link_names``."""
        self._fk(q)
        n = len(self.link_names)
        rot = np.empty((n, 3, 3)); pos = np.empty((n, 3))
        for i, name in enumerate(self.link_names):
            oMf = self.data.oMf[self._fids[name]]
            rot[i] = np.asarray(oMf.rotation); pos[i] = np.asarray(oMf.translation)
        return rot, pos

    def rest_transforms(self) -> tuple[np.ndarray, np.ndarray]:
        """Transformations de liens à la configuration free-flyer neutre (base identité)."""
        return self.link_transforms(self.neutral())

    def link_jacobians(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Transformations du monde + Jacobiens de frame translationnels/angulaires LOCAL_WORLD_ALIGNED par lien.
        ``dp_world = jac_lin @ v``, ``omega_world = jac_ang @ v`` (v dans l'ordre tangent pinocchio).
        Retourne ``(rot (L,3,3), pos (L,3), jac_lin (L,3,nv), jac_ang (L,3,nv))``."""
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
        """Limites de position de joints actualisés de l'URDF (rad), la tranche de joints de la config
        free-flyer (q[:7] est la base) : ``(lower (dof,), upper (dof,))``."""
        lo = np.asarray(self.model.lowerPositionLimit, np.float64)[7:7 + self.dof]
        hi = np.asarray(self.model.upperPositionLimit, np.float64)[7:7 + self.dof]
        return lo, hi


def build_robot_model(spec: RobotSpec) -> PinRobot:
    """Construire le ``RobotModel`` pinocchio pour ``spec`` (FK + Jacobiens, pas de meshes)."""
    return PinRobot(spec)
