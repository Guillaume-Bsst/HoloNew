"""transport — copy the human field onto the robot's M correspondence points (gather by ``smpl_idx``).

Human-only: the human is the field SOURCE (it deforms, so it is probed, never a field). The static
point<->link binding stays in ``InteractionContext.correspondence`` — not duplicated per frame. A pure
gather: ``out.<field>[c, m] = human_field.<field>[c, smpl_idx[m]]``. Ported from HoloNew
``correspondence/transport`` (the field gather is HoloNew ``interaction.frame_references``:
``pf.field.<x>[human_idx]``).

The human->robot metric ``scale`` is NOT applied here: ``transport`` stays a frame-agnostic gather.
L'échelle de scène (placement) est appliquée en ÉTAPE FINALE sur les RÉFÉRENCES par le ``pipeline``
(``targets.config.SceneScaleConfig`` + ``targets.scale``), APRÈS l'évaluation sur la scène réelle —
jamais avant (sinon l'assignation des contacts est corrompue). Le witness objet reste en frame local
(l'objet garde sa taille réelle) et suit la pose objet scalée.
"""
from __future__ import annotations

import numpy as np

from ..contracts import MultiChannelField
from ...prepare.contracts import CorrespondenceTable


def transport(human_field: MultiChannelField, correspondence: CorrespondenceTable) -> MultiChannelField:
    """Gather the human ``MultiChannelField`` ``(C, P_human)`` onto the M robot points via
    ``correspondence.smpl_idx`` -> ``(C, M)``. One fancy-index over the points axis per field
    (vectorised, channel-first, no per-point loop); channel names carry over. ``transport`` only
    gathers and never changes a per-channel frame (see ``MultiChannelField``), so every field —
    ``distance``, ``direction`` (unit vector), ``witness``, ``active`` (mask) — is gathered as-is."""
    smpl_idx = np.asarray(correspondence.smpl_idx)                    # (M,) into the human points axis
    distance = np.asarray(human_field.distance)[:, smpl_idx]         # (C, M)
    direction = np.asarray(human_field.direction)[:, smpl_idx]       # (C, M, 3) unit
    witness = np.asarray(human_field.witness)[:, smpl_idx]           # (C, M, 3)
    active = np.asarray(human_field.active, dtype=bool)[:, smpl_idx]  # (C, M)

    return MultiChannelField(distance=distance, direction=direction, witness=witness,
                             active=active, channels=human_field.channels)
