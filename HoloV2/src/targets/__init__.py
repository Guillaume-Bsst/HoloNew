"""``targets`` stage — online, q-independent: per-frame style + interaction targets.

Public surface (what downstream stages import): ``targets.contracts`` (the data types it produces,
e.g. ``FrameTargets``/``FrameTrace``) and ``targets.config`` (its knobs, when added). It consumes the
upstream ``prepare`` contracts; ``solve`` and ``viz`` import their inputs from ``targets.contracts``.
"""
