"""Cibles d'interaction (online, par frame) : positionner les nuages, évaluer les canaux, transporter
le champ humain sur le robot, assembler (références) ; évaluer le robot @ q (``contact_eval``). Ops purs
dans les sous-modules ; le flux est unidirectionnel (pose → eval → transport → assemble)."""
from .pointclouds import pose_cloud
from .fields import eval_fields
from .eval import contact_eval
from .transport import transport
from .refs import environment_interaction_targets, robot_interaction_targets
from .geodesic import geo_value_grad, nearest_index

__all__ = ["pose_cloud", "eval_fields", "contact_eval", "transport",
           "robot_interaction_targets", "environment_interaction_targets",
           "geo_value_grad", "nearest_index"]
