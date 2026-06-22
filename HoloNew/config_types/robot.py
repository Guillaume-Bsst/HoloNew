"""Configuration types for robot retargeting."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping, TypedDict

import numpy as np


def parse_robot_name(name: str) -> tuple[str, int | None]:
    """Split a robot name into (robot_type, robot_dof). A bare type ('g1', 't1') yields
    dof None (use the type default); a '_<N>dof' suffix ('g1_27dof') overrides the dof.
    Case-insensitive, so '--robot g1_27dof' / 'G1_27dof' both switch to 27-DOF."""
    m = re.fullmatch(r"([a-z]+\d*)(?:_(\d+)dof)?", name.strip().lower())
    if not m:
        return name, None
    return m.group(1), (int(m.group(2)) if m.group(2) else None)


def _g1_remap_27dof(overrides: dict[str, float], waist_pair: tuple[int, int]) -> dict[str, float]:
    """Remap g1 joint-override indices from 29-DOF to 27-DOF. The 27-DOF model drops
    waist_roll and waist_pitch, so their two indices are removed and every higher index
    shifts down by 2. ``waist_pair`` is those two indices in this table's index space
    (qpos for MANUAL_LB/UB, actuated-joint for MANUAL_COST)."""
    hi = max(waist_pair)
    out: dict[str, float] = {}
    for k, v in overrides.items():
        i = int(k)
        if i in waist_pair:
            continue
        out[str(i - 2 if i > hi else i)] = v
    return out


# Default values per robot type
class RobotDefaults(TypedDict):
    robot_dof: int
    robot_height: float
    object_name: str


_ROBOT_DEFAULTS: dict[str, RobotDefaults] = {
    "g1": {"robot_dof": 29, "robot_height": 1.32, "object_name": "ground"},
    "t1": {"robot_dof": 23, "robot_height": 1.2, "object_name": "ground"},
}


def _default_robot_defaults() -> dict[str, RobotDefaults]:
    """Copy robot defaults so each config instance can be customized safely."""
    return {name: defaults.copy() for name, defaults in _ROBOT_DEFAULTS.items()}


def _validate_robot_type(robot_type: str, robot_defaults: Mapping[str, RobotDefaults] | None = None) -> None:
    """Validate that robot_type exists in robot defaults."""
    if robot_defaults is None:
        robot_defaults = _ROBOT_DEFAULTS

    if robot_type not in robot_defaults:
        available = ", ".join(sorted(robot_defaults.keys()))
        raise ValueError(
            f"Invalid robot_type: '{robot_type}'. "
            f"Available robot types: {available}. "
            f"Add your robot to RobotConfig.robot_defaults "
            f"(default defined by _ROBOT_DEFAULTS in config_types/robot.py)"
        )


@dataclass(frozen=True)
class RobotConfig:
    """Unified configuration for all robot constants (G1, T1) using tyro.

    Example usage:
        # From CLI:
        config = tyro.cli(RobotConfig)  # --robot-type g1 --robot-dof 30

        # With defaults:
        config = RobotConfig(robot_type="g1")

        # Access values:
        robot_dof = config.ROBOT_DOF
        robot_height = config.ROBOT_HEIGHT
    """

    # Robot type selector - determines which defaults to use
    # Use str instead of Literal to allow dynamic robot types via _ROBOT_DEFAULTS
    robot_type: str = "g1"
    robot_defaults: dict[str, RobotDefaults] = field(default_factory=_default_robot_defaults)

    def __post_init__(self) -> None:
        """Validate robot_type; auto-split a dof-suffixed name ('g1_27dof' -> type 'g1',
        dof 27) so '--robot g1_27dof' switches DOF at every construction site (frozen
        dataclass -> object.__setattr__)."""
        rtype, rdof = parse_robot_name(self.robot_type)
        if rtype != self.robot_type:
            object.__setattr__(self, "robot_type", rtype)
            if rdof is not None and self.robot_dof is None:
                object.__setattr__(self, "robot_dof", rdof)
        _validate_robot_type(self.robot_type, self.robot_defaults)

    # Robot configuration (optional overrides)
    robot_dof: int | None = None
    robot_height: float | None = None
    robot_name: str | None = None
    robot_urdf_file: str | None = None

    # Joint definitions (optional overrides)
    foot_sticking_links: list[str] | None = None

    # Manual joint limits
    manual_lb: dict[str, float] | None = None
    manual_ub: dict[str, float] | None = None
    manual_cost: dict[str, float] | None = None

    # Nominal tracking indices
    nominal_tracking_indices: np.ndarray | None = None

    # Basic robot properties
    def _robot_dof(self) -> int:
        """Get robot DOF - use override if provided, else use robot_type default."""
        if self.robot_dof is not None:
            return self.robot_dof
        return self.robot_defaults[self.robot_type]["robot_dof"]

    ROBOT_DOF = property(
        _robot_dof,
        doc="Get robot DOF - use override if provided, else use robot_type default.",
    )

    def _robot_height(self) -> float:
        """Get robot height - use override if provided, else use robot_type default."""
        if self.robot_height is not None:
            return self.robot_height
        return self.robot_defaults[self.robot_type]["robot_height"]

    ROBOT_HEIGHT = property(
        _robot_height,
        doc="Get robot height - use override if provided, else use robot_type default.",
    )

    def _robot_name(self) -> str:
        """Get robot name - use override if provided, else compute from robot_type and DOF."""
        if self.robot_name is not None:
            return self.robot_name
        return f"{self.robot_type}_{self.ROBOT_DOF}dof"

    ROBOT_NAME = property(
        _robot_name,
        doc="Get robot name - use override if provided, else compute from robot_type and DOF.",
    )

    def _robot_urdf_file(self) -> str:
        """Get robot URDF file path."""
        if self.robot_urdf_file is not None:
            return self.robot_urdf_file
        return f"models/{self.robot_type}/{self.robot_type}_{self.ROBOT_DOF}dof.urdf"

    ROBOT_URDF_FILE = property(_robot_urdf_file, doc="Get robot URDF file path.")

    def _foot_sticking_links(self) -> list[str]:
        """Get foot sticking links - use override if provided, else use robot_type default."""
        if self.foot_sticking_links is not None:
            return self.foot_sticking_links

        if self.robot_type == "g1":
            return [
                "left_ankle_roll_sphere_1_link",
                "right_ankle_roll_sphere_1_link",
                "left_ankle_roll_sphere_2_link",
                "right_ankle_roll_sphere_2_link",
                "left_ankle_roll_sphere_3_link",
                "right_ankle_roll_sphere_3_link",
                "left_ankle_roll_sphere_4_link",
                "right_ankle_roll_sphere_4_link",
            ]
        if self.robot_type == "t1":
            return [
                "left_foot_sphere_1_link",
                "right_foot_sphere_1_link",
                "left_foot_sphere_2_link",
                "right_foot_sphere_2_link",
                "left_foot_sphere_3_link",
                "right_foot_sphere_3_link",
                "left_foot_sphere_4_link",
                "right_foot_sphere_4_link",
                "left_foot_sphere_5_link",
                "right_foot_sphere_5_link",
            ]
        raise ValueError(f"Invalid robot type: {self.robot_type}")

    FOOT_STICKING_LINKS = property(
        _foot_sticking_links,
        doc="Get foot sticking links - use override if provided, else use robot_type default.",
    )

    def _manual_lb(self) -> dict[str, float]:
        """Get manual lower bounds."""
        if self.manual_lb is not None:
            return self.manual_lb

        base: dict[str, float] = {"3": -1.0, "4": -1.0, "5": -1.0, "6": -1.0}  # quaternion bounds

        if self.robot_type == "g1":
            # qpos-space indices (29-DOF): waist_roll 20, waist_pitch 21,
            # left wrist 26-28, right wrist 33-35.
            g1 = {
                "20": -0.3,  # waist roll
                "21": -0.1,  # waist pitch
                "26": -0.1,  # left wrist roll/pitch/yaw
                "27": -0.1,
                "28": -0.05,
                "33": -0.1,  # right wrist roll/pitch/yaw
                "34": -0.1,
                "35": -0.05,
            }
            if self.ROBOT_DOF == 27:
                g1 = _g1_remap_27dof(g1, (20, 21))  # waist removed -> drop 20/21, shift >21 by -2
            base.update(g1)

        return base

    MANUAL_LB = property(_manual_lb, doc="Get manual lower bounds.")

    def _manual_ub(self) -> dict[str, float]:
        """Get manual upper bounds."""
        if self.manual_ub is not None:
            return self.manual_ub

        base: dict[str, float] = {"3": 1.0, "4": 1.0, "5": 1.0, "6": 1.0}  # quaternion bounds

        if self.robot_type == "g1":
            # qpos-space indices (29-DOF): waist_roll 20, left elbow 25, left wrist 26-28,
            # right elbow 32, right wrist 33-35.
            g1 = {
                "20": 0.3,  # waist roll
                "25": 1.4,  # left elbow
                "26": 0.2,  # left wrist
                "27": 0.3,
                "28": 0.05,
                "32": 1.4,  # right elbow
                "33": 0.2,  # right wrist
                "34": 0.3,
                "35": 0.05,
            }
            if self.ROBOT_DOF == 27:
                g1 = _g1_remap_27dof(g1, (20, 21))  # waist removed -> drop 20, shift >21 by -2
            base.update(g1)

        return base

    MANUAL_UB = property(_manual_ub, doc="Get manual upper bounds.")

    def _manual_cost(self) -> dict[str, float]:
        """Get manual cost weights."""
        if self.manual_cost is not None:
            return self.manual_cost

        if self.robot_type == "g1":
            # Actuated-joint space (29-DOF).
            cost = {"19": 0.2, "20": 0.2}
            if self.ROBOT_DOF == 27:
                # waist_roll/pitch are joints 13/14 here -> drop them, shift >14 by -2.
                cost = _g1_remap_27dof(cost, (13, 14))
            return cost
        return {}

    MANUAL_COST = property(_manual_cost, doc="Get manual cost weights.")

    def _nominal_tracking_indices(self) -> np.ndarray:
        """Get nominal tracking indices."""
        if self.nominal_tracking_indices is not None:
            return self.nominal_tracking_indices

        if self.robot_type == "g1":
            return np.arange(19)
        if self.robot_type == "t1":
            return np.concatenate([np.arange(7), np.arange(11, 23)])
        # Default: return empty array if robot type not defined (nominal tracking not used)
        return np.array([], dtype=int)

    NOMINAL_TRACKING_INDICES = property(
        _nominal_tracking_indices,
        doc="Get nominal tracking indices.",
    )
