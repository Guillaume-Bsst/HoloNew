"""Assembler les cibles d'interaction à partir des champs évalués — des wrappers explicites et triviaux
(les formes de contrat, rendues de première classe). Le côté robot est le champ humain transporté ;
le côté environnement agrège les champs par nuage-objet (objet-sol / objet-objet), non transportés.
"""
from __future__ import annotations

from ..contracts import (EnvironmentInteractionTargets, MultiChannelField,
                         RobotInteractionTargets)


def robot_interaction_targets(robot_field: MultiChannelField) -> RobotInteractionTargets:
    """Le champ humain transporté sur les M points robot. La liaison point↔lien vit dans la
    correspondance du contexte (statique), donc la cible par frame est le champ SEUL."""
    return RobotInteractionTargets(field=robot_field)


def environment_interaction_targets(
        object_fields: tuple[MultiChannelField, ...]) -> EnvironmentInteractionTargets:
    """Agréger les champs par nuage-objet (contact côté scène ; viz / une contrainte de résolution ultérieure)."""
    return EnvironmentInteractionTargets(per_object=tuple(object_fields))
