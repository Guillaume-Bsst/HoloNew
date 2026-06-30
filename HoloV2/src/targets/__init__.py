"""étage ``targets`` — online : style par frame + RÉFÉRENCES d'interaction (q-indépendantes) ET
l'ÉVALUATEUR q-dépendant (le seam ``targets -> solve``).

Surface publique (ce que les étages aval importent) : ``targets.contracts`` (les types de données qu'il
produit — références ``FrameTargets``/``FrameTrace`` + eval ``StyleEval``/``ContactEval``/``ContactEnvEval``),
``targets.config`` (ses knobs — ``TargetsConfig`` / ``StyleConfig``), le noyau d'interaction pur
réutilisé par ``solve`` (``pose_cloud`` / ``eval_fields`` -> ``MultiChannelField``, plus les lecteurs
géodésiques), ET l'``Evaluator`` (construit une fois à partir de l'``InteractionContext``, évalue
``(q, object_poses)`` -> géométrie courante + Jacobiennes). Importer les noyaux purs, l'``Evaluator`` et
les types de données depuis le PACKAGE (``from ..targets import ...``), plutôt que depuis le sous-module
interne ``targets.interaction``. ``targets.pipeline`` (``process_frame`` / ``trace_frame`` /
``run_sequence``) est le point d'entrée de l'orchestrateur par frame — une cible consommatrice légitime
pour ``viz`` (qui importe ``trace_frame`` directement) et la boucle de run de haut niveau. Il consomme
les contrats amont ``prepare`` ; ``solve`` et ``viz`` importent leurs entrées depuis ``targets.contracts``.
"""
from .interaction import eval_fields, pose_cloud, geo_value_grad, nearest_index
from .contracts import MultiChannelField, StyleEval, ContactEval, ContactEnvEval
from .evaluator import Evaluator

__all__ = ["pose_cloud", "eval_fields", "MultiChannelField", "geo_value_grad", "nearest_index",
           "Evaluator", "StyleEval", "ContactEval", "ContactEnvEval"]
