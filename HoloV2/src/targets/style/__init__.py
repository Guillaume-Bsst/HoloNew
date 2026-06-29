"""Style treatment: reference (``build``: demo joints -> StyleTargets) + evaluation
(``style_eval``: robot FK @ q -> StyleEval). See ``build.py`` / ``eval.py``."""
from .build import build
from .eval import style_eval

__all__ = ["build", "style_eval"]
