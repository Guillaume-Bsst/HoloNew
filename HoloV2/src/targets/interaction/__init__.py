"""Interaction targets (online, per frame): pose the clouds, evaluate the channels, transport the
human field onto the robot, assemble. Pure ops in the submodules; the flow is one-way
(pose -> eval -> transport -> assemble). ``pose_cloud`` is shared by every cloud kind."""
from .pointclouds import pose_cloud
from .fields import eval_fields
from .transport import transport
from .refs import environment_interaction_targets, robot_interaction_targets
from .geodesic import geo_value_grad, nearest_index

__all__ = ["pose_cloud", "eval_fields", "transport",
           "robot_interaction_targets", "environment_interaction_targets",
           "geo_value_grad", "nearest_index"]
