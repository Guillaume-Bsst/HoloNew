"""``targets`` stage — online: per-frame style + interaction REFERENCES (q-independent) AND the
q-dependent EVALUATOR (the ``targets -> solve`` seam).

Public surface (what downstream stages import): ``targets.contracts`` (the data types it produces —
references ``FrameTargets``/``FrameTrace`` + eval ``StyleEval``/``ContactEval``/``ContactEnvEval``),
``targets.config`` (its knobs — ``TargetsConfig`` / ``StyleConfig``), the pure interaction kernel
reused by ``solve`` (``pose_cloud`` / ``eval_fields`` -> ``MultiChannelField``, plus the geodesic
readers), AND the ``Evaluator`` (built once from ``InteractionContext``, evaluates ``(q, object_poses)``
-> current geometry + Jacobians). Import the pure kernels, ``Evaluator``, and the data types from the
PACKAGE (``from ..targets import ...``), rather than from the internal ``targets.interaction``
submodule. ``targets.pipeline`` (``process_frame`` / ``trace_frame`` / ``run_sequence``) is the
per-frame orchestrator entry point — a legitimate consumer target for ``viz`` (which imports
``trace_frame`` directly) and the top-level run loop. It consumes the upstream ``prepare`` contracts;
``solve`` and ``viz`` import their inputs from ``targets.contracts``.
"""
from .interaction import eval_fields, pose_cloud, geo_value_grad, nearest_index
from .contracts import MultiChannelField, StyleEval, ContactEval, ContactEnvEval
from .evaluator import Evaluator

__all__ = ["pose_cloud", "eval_fields", "MultiChannelField", "geo_value_grad", "nearest_index",
           "Evaluator", "StyleEval", "ContactEval", "ContactEnvEval"]
