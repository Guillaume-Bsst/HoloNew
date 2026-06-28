"""Interaction targets (online, per frame): pose the clouds, evaluate the channels, transport the
human field onto the robot, assemble. Pure ops in the submodules; the flow is one-way
(pose -> eval -> transport -> assemble). ``pose_cloud`` is shared by every cloud kind."""
from .pointclouds import pose_cloud
from .eval import eval_fields
from .transport import transport
from .targets import environment_interaction_targets, robot_interaction_targets

__all__ = ["pose_cloud", "eval_fields", "transport",
           "robot_interaction_targets", "environment_interaction_targets"]
