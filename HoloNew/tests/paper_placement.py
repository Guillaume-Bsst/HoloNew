"""Explicit reconstruction of the removed ``inertia_mode`` preset (config flatten,
2026-06-15).

``inertia_mode=True`` used to be a single flag that silently rewrote ~8 other fields in
the builder. The config is now flat and explicit, so this test helper spells out the
fields that preset resolved to: paper-faithful body placement — the body and object are
placed by CONTACTS (feet on the floor entity + object<->floor contact, weak centroidal
filling the flight phase) with no positional pelvis / CoM / object anchor, and the Style
objective in its pelvis-relative frame.

Use ``paper_placement_config(**overrides)`` in tests instead of the old flag.
"""
from HoloNew.src.test_socp.config import TestSocpRetargeterConfig

# Explicit switches of the old inertia_mode bundle + the bricks it relied on. The cost
# weights themselves default to their tuned values (config §3), so only the switches and
# the pelvis scaffold weight are set here. The position anchors (W^c_pos, W^o_pos) stay
# OFF: the body and object are placed by contacts.
PAPER_PLACEMENT = dict(
    # Style objective, pelvis-relative frame (the paper-placement frame).
    activate_ws=True, style_pelvis_relative=True, pelvis_anchor_weight=0.0,
    # Contact terms that place the body/feet + temporal regularization.
    activate_wd=True, activate_wx=True, activate_persistence=True, activate_wr=True,
    # Movable object, placed by contacts (no position anchor).
    activate_movable=True, activate_wo=True,
    # Weak centroidal W^c / W^L fills the residual / flight.
    activate_wc=True, activate_wl=True,
    # Floor as a contact entity + object<->floor contact, with the required
    # non-penetration constraint (enforced in from_config).
    floor_as_entity=True, activate_wo_floor=True, activate_obj_non_penetration=True,
)


def paper_placement_config(**overrides) -> TestSocpRetargeterConfig:
    """TestSocpRetargeterConfig for the paper-faithful placement (old inertia_mode)."""
    return TestSocpRetargeterConfig(**{**PAPER_PLACEMENT, **overrides})
