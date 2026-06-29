"""Assemble the interaction targets from the evaluated fields — explicit, trivial wrappers (the
contract shapes, made first-class). The robot side is the transported human field; the environment
side bundles the per-object-cloud fields (object-ground / object-object), not transported.
"""
from __future__ import annotations

from ..contracts import (EnvironmentInteractionTargets, MultiChannelField,
                         RobotInteractionTargets)


def robot_interaction_targets(robot_field: MultiChannelField) -> RobotInteractionTargets:
    """The transported human field on the M robot points. The point<->link binding lives in the
    context's correspondence (static), so the per-frame target is the field ALONE."""
    return RobotInteractionTargets(field=robot_field)


def environment_interaction_targets(
        object_fields: tuple[MultiChannelField, ...]) -> EnvironmentInteractionTargets:
    """Bundle the per-object-cloud fields (scene-side contact; viz / a later solve constraint)."""
    return EnvironmentInteractionTargets(per_object=tuple(object_fields))
