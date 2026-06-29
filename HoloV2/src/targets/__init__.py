"""``targets`` stage — online, q-independent: per-frame style + interaction targets.

Public surface (what downstream stages import): ``targets.contracts`` (the data types it produces,
e.g. ``FrameTargets``/``FrameTrace``), ``targets.config`` (its knobs — ``TargetsConfig`` /
``StyleConfig``), AND the pure interaction kernel reused by ``solve`` — ``pose_cloud`` /
``eval_fields`` (-> ``MultiChannelField``).
Import these from the PACKAGE (``from ..targets import ...``), never from the internal
``targets.interaction`` submodule. It consumes the upstream ``prepare`` contracts; ``solve`` and
``viz`` import their inputs from ``targets.contracts``.
"""
from .interaction import eval_fields, pose_cloud, geo_value_grad, nearest_index
from .contracts import MultiChannelField

__all__ = ["pose_cloud", "eval_fields", "MultiChannelField", "geo_value_grad", "nearest_index"]
