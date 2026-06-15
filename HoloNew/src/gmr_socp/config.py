"""GMR-SOCP-specific retargeter config.

Subclasses holosoma's RetargeterConfig but defaults the holosoma-style optional
constraints OFF. These constraints are opt-in for GMR-SOCP: pass this config with
the relevant flag set to True (e.g. GmrSocpRetargeterConfig(activate_obj_non_penetration=True))
to enable them. With the defaults below the solve is identical to the plain GMR-SOCP solve.
"""
from __future__ import annotations

from dataclasses import dataclass

from HoloNew.config_types.retargeter import RetargeterConfig


@dataclass(frozen=True)
class GmrSocpRetargeterConfig(RetargeterConfig):
    """RetargeterConfig with holosoma-style constraints defaulting OFF for GMR-SOCP.

    The activate_* flags are gates: setting one to True is necessary but not sufficient.
    ``activate_self_collision=True`` only takes effect when paired with
    ``self_collision=SelfCollisionConfig(enable=True, pairs=[...])``.  Likewise,
    ``activate_foot_sticking`` requires populated foot-sticking sequences and
    ``foot_lock=FootLockConfig(enable=True, ...)`` to do anything.
    """

    activate_obj_non_penetration: bool = False
    activate_foot_sticking: bool = False
    activate_self_collision: bool = False

    # World placement applied inside the preprocess scale stage, independently for the
    # robot root and the object, in XY and Z. Each is a multiplier on the RAW grounded
    # axis (1.0 = raw). None keeps GMR-SOCP's native behaviour, resolved per axis in
    # from_config: XY -> the per-clip holosoma scale factor (ROBOT_HEIGHT/human_height,
    # ~0.68, pulling the root toward the world centre, matching holosoma); robot Z ->
    # the native morphological scaling; object Z -> raw (1.0, GMR never scaled it). Pass
    # a float to override any axis (e.g. scale_xy_robot=1.0 to keep the raw root XY and
    # match mink-GMR). Body proportions (pelvis-local) are unaffected by all four.
    scale_xy_robot: float | None = None
    scale_z_robot: float | None = None
    scale_xy_object: float | None = None
    scale_z_object: float | None = None
