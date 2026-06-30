"""transport — copier le champ humain sur les M points de correspondance du robot (cueillir par ``smpl_idx``).

Humain uniquement : l'humain est la SOURCE du champ (il se déforme, donc il est sondé, jamais un champ).
La liaison point↔lien statique reste dans ``InteractionContext.correspondence`` — pas dupliquée par frame.
Une pure cueillette : ``out.<field>[c, m] = human_field.<field>[c, smpl_idx[m]]``. Porté de HoloNew
``correspondence/transport`` (la cueillette de champ est HoloNew ``interaction.frame_references`` :
``pf.field.<x>[human_idx]``).

L'échelle métrique humain→robot ``scale`` n'est PAS appliquée ici : ``transport`` reste une cueillette
frame-agnostique. L'échelle de scène (placement) est appliquée en ÉTAPE FINALE sur les RÉFÉRENCES par
le ``pipeline`` (``targets.config.SceneScaleConfig`` + ``targets.scale``), APRÈS l'évaluation sur la
scène réelle — jamais avant (sinon l'assignation des contacts est corrompue). Le témoin objet reste en
frame local (l'objet garde sa taille réelle) et suit la pose objet scalée.
"""
from __future__ import annotations

import numpy as np

from ..contracts import MultiChannelField
from ...prepare.contracts import CorrespondenceTable


def transport(human_field: MultiChannelField, correspondence: CorrespondenceTable) -> MultiChannelField:
    """Cueillir le ``MultiChannelField`` humain ``(C, P_human)`` sur les M points robot via
    ``correspondence.smpl_idx`` → ``(C, M)``. Un index sophistiqué sur l'axe des points par champ
    (vectorisé, premier canal, pas de boucle par point) ; les noms de canaux se conservent. ``transport``
    ne cueille que et ne change jamais un frame par canal (voir ``MultiChannelField``), donc chaque champ —
    ``distance``, ``direction`` (vecteur unitaire), ``témoin``, ``actif`` (masque) — est cueilli tel quel."""
    smpl_idx = np.asarray(correspondence.smpl_idx)                    # (M,) vers l'axe des points humains
    distance = np.asarray(human_field.distance)[:, smpl_idx]         # (C, M)
    direction = np.asarray(human_field.direction)[:, smpl_idx]       # (C, M, 3) unitaire
    witness = np.asarray(human_field.witness)[:, smpl_idx]           # (C, M, 3)
    active = np.asarray(human_field.active, dtype=bool)[:, smpl_idx]  # (C, M)

    return MultiChannelField(distance=distance, direction=direction, witness=witness,
                             active=active, channels=human_field.channels)
