"""viz — consommateur pur de visualisation (règle d'or 6). Entrée publique : ``run_app`` / ``main``
(le viewer prod unifié sur le framework Source -> VizFrame -> Layers). Voir docs/VIZ.md."""
from .app import main, run_app

__all__ = ["run_app", "main"]
