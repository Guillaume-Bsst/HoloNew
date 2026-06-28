"""style.build — demo joints -> ``StyleTargets`` (posture/style tracking, G1-ready via a joint
mapping). The object-agnostic "how the body should move" channel.

PROVISIONAL placeholder — the style objective (WHAT to track and the SMPL-demo -> G1 joint mapping)
is still being designed (see docs/TARGETS.md). It reads the demo joints straight from the
``GroundedScene`` (J_demo, NOT J_bones), so it needs no ``FramePose``. Port target: HoloNew
``gmr_socp`` + the ``data_type`` mappings.
"""
from __future__ import annotations

from ..contracts import StyleTargets
from ...prepare.contracts import GroundedScene


def build(grounded: GroundedScene, f: int) -> StyleTargets:
    """One frame of demo joints -> ``StyleTargets``. NOT YET IMPLEMENTED — the style objective is TBD."""
    raise NotImplementedError
