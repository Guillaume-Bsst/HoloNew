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

# Effective fields of the old inertia_mode bundle + the bricks it relied on.
PAPER_PLACEMENT = dict(
    # Style objective, pelvis-relative frame (the paper-placement frame).
    activate_style=True, style_pelvis_relative=True, pelvis_anchor_weight=0.0,
    # Contact terms that actually place the body/feet.
    lambda_D=20.0, lambda_X=20.0, activate_persistence=True,
    lambda_r=0.2,
    # Movable object, placed by contacts (no position anchor).
    activate_movable=True, lambda_o=1.0, lambda_omega=1.0, lambda_o_pos=0.0,
    # Weak centroidal W^c / W^L fills the residual / flight.
    activate_centroidal=True, lambda_c=1e-5, lambda_c_pos=0.0, lambda_L=1e-4,
    # Floor as a contact entity + object<->floor contact, with the required
    # non-penetration constraint (now enforced explicitly in from_config).
    floor_as_entity=True, lambda_object_floor=5.0, activate_obj_non_penetration=True,
)


def paper_placement_config(**overrides) -> TestSocpRetargeterConfig:
    """TestSocpRetargeterConfig for the paper-faithful placement (old inertia_mode)."""
    return TestSocpRetargeterConfig(**{**PAPER_PLACEMENT, **overrides})
