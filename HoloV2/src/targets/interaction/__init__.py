"""Interaction targets (online, per frame): pose the clouds, evaluate the channels, transport the
human field onto the robot, assemble (references); evaluate the robot @ q (``contact_eval``). Pure ops
in the submodules; the flow is one-way (pose -> eval -> transport -> assemble)."""
from .pointclouds import pose_cloud
from .fields import eval_fields
from .eval import contact_eval
from .transport import transport
from .refs import environment_interaction_targets, robot_interaction_targets
from .geodesic import geo_value_grad, nearest_index

__all__ = ["pose_cloud", "eval_fields", "contact_eval", "transport",
           "robot_interaction_targets", "environment_interaction_targets",
           "geo_value_grad", "nearest_index"]
