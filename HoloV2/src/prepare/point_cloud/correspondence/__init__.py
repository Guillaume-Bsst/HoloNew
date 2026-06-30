"""Correspondance de surface humain<->robot, construite hors ligne par transport optimal par segment
(et relue du cache). La logique réside dans les sous-modules ; ``robot_surface``/``segments``/
``ot_couple`` sont les étapes de construction génériques, ``build`` l'orchestrateur + ``AssetBuilder``,
``cache`` le format (save+load)."""
from .build import CorrespondenceBuilder, build_correspondence, regenerate
from .cache import load_correspondence, save_correspondence
from .robot_cloud import robot_point_cloud

__all__ = ["load_correspondence", "save_correspondence", "build_correspondence",
           "CorrespondenceBuilder", "regenerate", "robot_point_cloud"]
