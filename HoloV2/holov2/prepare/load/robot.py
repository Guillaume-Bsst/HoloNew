"""Robot kinematics from a URDF — robot-agnostic ``RobotModel`` (yourdfpy FK).

Generic across humanoids: the robot identity (URDF, link names, dof) comes from the ``RobotSpec``.
The only robot-SPECIFIC data lives here as a name-keyed table — ``CORRESPONDENCE_REST_POSE`` — so
adding a new robot is a data entry, never a change to the generic surface/OT/transport code.
"""
from __future__ import annotations

import numpy as np

from ...contracts import RobotSpec

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


class UrdfRobot:
    """``RobotModel`` backed by a URDF (yourdfpy). Fixed-base link world transforms for a given
    actuated-joint configuration; ``link_names`` are every URDF link (transport indexes by name)."""

    def __init__(self, spec: RobotSpec) -> None:
        import yourdfpy
        self._urdf = yourdfpy.URDF.load(str(spec.urdf_path), load_meshes=False, build_scene_graph=True)
        self.link_names: tuple[str, ...] = tuple(self._urdf.link_map.keys())
        self.dof: int = len(self._urdf.actuated_joint_names)

    def link_transforms(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(L,3,3) rotations, (L,3) positions of every link for the actuated config ``qpos`` (dof,),
        fixed base. Aligned with ``link_names``."""
        self._urdf.update_cfg(np.asarray(qpos, np.float64))
        rot = np.empty((len(self.link_names), 3, 3))
        pos = np.empty((len(self.link_names), 3))
        for i, name in enumerate(self.link_names):
            t = np.asarray(self._urdf.get_transform(name))
            rot[i], pos[i] = t[:3, :3], t[:3, 3]
        return rot, pos

    def rest_transforms(self) -> tuple[np.ndarray, np.ndarray]:
        """Link transforms at the zero (rest) configuration."""
        return self.link_transforms(np.zeros(self.dof))


def build_robot_model(spec: RobotSpec) -> UrdfRobot:
    """Build the ``RobotModel`` for ``spec`` (loads the URDF, no meshes — FK only)."""
    return UrdfRobot(spec)
