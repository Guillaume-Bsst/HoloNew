"""transport — copy the human field onto the robot's M correspondence points (gather by ``smpl_idx``).

Human-only: the human is the field SOURCE (it deforms, so it is probed, never a field). The static
point<->link binding stays in ``InteractionContext.correspondence`` — not duplicated per frame.
Ported from HoloNew ``correspondence/transport``.
"""
from __future__ import annotations

from ..contracts import MultiChannelField
from ...prepare.contracts import CorrespondenceTable


def transport(human_field: MultiChannelField, correspondence: CorrespondenceTable) -> MultiChannelField:
    """Gather the human ``MultiChannelField`` ``(C, P_human)`` onto the M robot points via
    ``correspondence.smpl_idx`` -> ``(C, M)``.

    TODO(scale): the human->robot metric scale (= ``robot_height / body.stature``) is applied HERE
    when distances/witnesses are mapped to robot size. Surface it as ``InteractionContext.scale`` — a
    single (human, robot) scalar computed once in the runner — and thread it in at that point (kept
    out of the contracts until this op needs it, per YAGNI)."""
    raise NotImplementedError
