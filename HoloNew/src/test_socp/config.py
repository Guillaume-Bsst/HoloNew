"""TEST-SOCP-specific retargeter config.

Subclasses holosoma's RetargeterConfig but defaults the holosoma-style optional
constraints OFF. These constraints are opt-in for TEST-SOCP: pass this config with
the relevant flag set to True (e.g. TestSocpRetargeterConfig(activate_obj_non_penetration=True))
to enable them. With the defaults below the solve is identical to the plain TEST-SOCP solve.
"""
from __future__ import annotations

from dataclasses import dataclass

from HoloNew.config_types.retargeter import RetargeterConfig


@dataclass(frozen=True)
class TestSocpRetargeterConfig(RetargeterConfig):
    """RetargeterConfig with holosoma-style constraints defaulting OFF for TEST-SOCP.

    The activate_* flags are gates: setting one to True is necessary but not sufficient.
    ``activate_self_collision=True`` only takes effect when paired with
    ``self_collision=SelfCollisionConfig(enable=True, pairs=[...])``.  Likewise,
    ``activate_foot_sticking`` requires populated foot-sticking sequences and
    ``foot_lock=FootLockConfig(enable=True, ...)`` to do anything.
    """

    activate_obj_non_penetration: bool = False
    activate_foot_sticking: bool = False
    activate_self_collision: bool = False

    # Interaction D/X/P cost weights (default 0.0 = off; solve is unchanged).
    lambda_D: float = 0.0
    lambda_X: float = 0.0
    lambda_P: float = 0.0
    sigma_v: float = 0.05
