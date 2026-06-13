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

    # Interaction cost weights. D (normal proximity) + X (tangential placement)
    # are enabled by default on object tasks; robot_only (no object SDF) is
    # unchanged. The interaction costs REQUIRE the non-penetration constraint to
    # be stable — from_config auto-enables ground non-penetration when interaction
    # is active on an object task (without it the D term marches the floating base
    # through the floor). Validated stable; reduces the contact gap (~0.012 vs
    # ~0.028 off on sub3_largebox_003).
    #
    # P (contact persistence): implemented, math-validated, and renormalized by the
    # field range L^2 (same scale as X — see interaction.build_p_terms; a deliberate
    # divergence from the paper's (sigma_v*dt)^2 which is ~3600x larger and wrecks
    # conditioning). With the L^2 scale P no longer explodes, but its hundreds of
    # near-parallel per-point rows still make CLARABEL fail intermittently mid-clip,
    # so the solve falls back to SCS on those iterations (see solve_single_iteration)
    # — which is robust but ~3x slower over a clip. P is therefore OFF by default to
    # keep the solve fast; enabling it is a one-liner (lambda_P>0). Making it fast
    # needs a per-carrier aggregation of the persistence residual (few well-
    # conditioned rows instead of hundreds). sigma_v is kept for API compatibility.
    lambda_D: float = 1.0
    lambda_X: float = 1.0
    lambda_P: float = 0.0
    sigma_v: float = 0.05

    # Temporal regularization (W^r): penalizes tangent-space acceleration across
    # consecutive frames. sigma_qddot / sigma_Vdot set the per-DOF noise scale
    # for joints and base, respectively (same units as temporal.py).
    #
    # Defaults tuned 2026-06-13 (Brick 2) on robot_only sub3_largebox_003 (K=30):
    #   off (lambda_r=0): mean joint jerk=0.003640, pelvis track=0.040 m
    #   on  (0.2/20/20):  mean joint jerk=0.002990 (−17.9 %), track=0.049 m (+0.009 m)
    # Jerk is reduced without meaningful tracking degradation (< 0.01 m slack).
    lambda_r: float = 0.2
    sigma_qddot: float = 20.0
    sigma_Vdot: float = 20.0

    # Brick 3 — Pelvis-relative Style objective.
    # Enabled by default after validation (2026-06-13): 30-frame robot_only solve
    # shows pelvis-relative fidelity 0.60 rad vs 0.82 rad for world tracking
    # (~27 % improvement) and pelvis z in [0.562, 0.800] m — clean, no collapse.
    # When on, joint orientation targets are re-based by the current pelvis,
    # the pelvis orientation term becomes a roll/pitch-only tilt, joint position
    # terms are dropped, and a weak pelvis position scaffold is kept.
    # pelvis_anchor_weight scales the scaffold (1.0 = unchanged relative to w_p).
    activate_style: bool = True
    pelvis_anchor_weight: float = 1.0

    # Brick 4 — Centroidal W^c (CoM acceleration) + W^L (angular momentum).
    # Default OFF (activate_centroidal=False, lambda_c/lambda_L=0): solve is
    # bit-exact with Brick 3 baseline. Enable only if validated stable (Task 4).
    # When on, two quadratic terms are added per frame (frame_idx >= 2):
    #   W^c = lambda_c * ||c_ddot - c_ddot_ref||^2  (CoM acceleration tracking)
    #   W^L = lambda_L * ||L||^2                     (centroidal angular momentum -> 0)
    activate_centroidal: bool = False
    lambda_c: float = 0.0
    lambda_L: float = 0.0
