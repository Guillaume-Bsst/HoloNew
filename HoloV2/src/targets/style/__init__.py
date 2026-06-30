"""Traitement du style : référence (``build`` : articulations démo → StyleTargets) + évaluation
(``style_eval`` : FK robot @ q → StyleEval). Voir ``build.py`` / ``eval.py``."""
from .build import build
from .eval import style_eval

__all__ = ["build", "style_eval"]
