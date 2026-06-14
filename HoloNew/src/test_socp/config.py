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
    # through the floor).
    #
    # Weights jointly re-tuned 2026-06-14 from 1.0 to 5.0. In the full pipeline
    # (Style scaffold + W^r + persistence + movable all on) lambda=1.0 was
    # DOMINATED by the other objective terms: D/X at 1.0 made BOTH contact gaps
    # WORSE than D/X off (the cost was present but too weak to win). Sweep on
    # sub3_largebox_003 (K=8, all bricks on), object/floor mean contact gap:
    #   lambda=1:   object=0.0301 floor=0.0363   (worse than off: object=0.0200)
    #   lambda=5:   object=0.0096 floor=0.0314   <-- chosen (object channel best)
    #   lambda=20:  object=0.0306 floor=0.0171
    #   lambda=100: object=0.0372 floor=0.0118
    # lambda=5 halves the OBJECT (manipulation) contact gap vs D/X off — that is
    # D/X's primary job. The FLOOR channel is now owned by the persistence no-slip
    # hard constraint (D/X off + persistence on gives floor=0.0107, which D/X
    # cannot beat at any weight), so the acceptance metric tracks the object gap.
    #
    # P (contact persistence / no-slip): the SOFT cost form (lambda_P>0) is kept for
    # reference but defaults OFF. It renormalizes the persistence residual by the field
    # range L^2 (same scale as X — see interaction.build_p_terms; a deliberate divergence
    # from the paper's (sigma_v*dt)^2 which is ~3600x larger and wrecks conditioning).
    # Even at the L^2 scale its hundreds of near-parallel per-point rows make CLARABEL
    # fail intermittently mid-clip, forcing a ~3x-slower SCS fallback. The no-slip
    # behaviour is instead delivered by the HARD tangential band constraint
    # (activate_persistence, below), which is both fast and tight — prefer it. sigma_v
    # is kept for API compatibility and is unused by the hard constraint.
    lambda_D: float = 5.0
    lambda_X: float = 5.0
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
    # terms are dropped, and a position scaffold keeps the base on the reference path.
    # pelvis_anchor_weight scales the scaffold relative to w_p.  Style frees the
    # pelvis ORIENTATION (yaw), not its position; the scaffold must be strong enough
    # to prevent the base from drifting off the reference trajectory.  Sweep on
    # sub3_largebox_003 (30 frames): paw=1 → mean xy-drift 0.243 m (too much),
    # paw=10 → mean xy-drift ~0.09 m, well within tolerance.
    activate_style: bool = True
    pelvis_anchor_weight: float = 10.0

    # Brick 4 — Centroidal W^c (CoM acceleration) + W^c_pos (CoM position) + W^L.
    # Default OFF (activate_centroidal=False, lambdas=0): solve is bit-exact with
    # Brick 3 baseline. Enable only if validated stable (Task 4).
    # When on, up to three quadratic terms are added per frame (frame_idx >= 2 for
    # W^c and W^L; W^c_pos fires from frame 0 as it needs no prior CoM history):
    #   W^c     = lambda_c     * ||c_ddot - c_ddot_ref||^2  (CoM accel tracking)
    #   W^c_pos = lambda_c_pos * ||c - c_ref||^2             (CoM position anchor)
    #   W^L     = lambda_L     * ||L||^2                     (angular momentum -> 0)
    # W^c_pos anchors the absolute CoM to the reference pelvis trajectory, curing
    # the constant-velocity drift that W^c (second difference only) cannot prevent.
    activate_centroidal: bool = False
    lambda_c: float = 0.0
    lambda_c_pos: float = 0.0
    lambda_L: float = 0.0

    # Inertia mode (paper-faithful body placement). When True, from_config applies
    # a bundle: floor_as_entity=True, pelvis_anchor_weight=0, lambda_c_pos=0,
    # activate_centroidal=True with weak lambda_c/lambda_L. The body is placed by
    # contacts (feet pinned to the permanent floor entity) and a weak W^c filling
    # the residual/flight, with NO positional pelvis/CoM target. Default off so the
    # parity/golden tests stay bit-exact. See docs/specs/2026-06-14-inertia-mode-design.md.
    inertia_mode: bool = False
    # floor_as_entity: load the floor interaction channel (correspondence + ground
    # probe + floor field) for ANY task, not only object tasks. Turned on by
    # inertia_mode; kept separable for testing.
    floor_as_entity: bool = False

    # Brick 1 — Contact persistence as a hard tangential band constraint.
    # Enabled by default after validation (2026-06-14): 30-frame object_interaction
    # sub3_largebox_003 with persistence_tol=0.005 (5 mm) shows:
    #   mean tangential slip: 5.09 mm (vs 74.25 mm without constraint; 14.6x tighter)
    #   runtime: ~1.55x baseline (vs ~3x for soft lambda_P cost with SCS fallback)
    #   SCS fallbacks: 0 (CLARABEL handles all frames at tol=5 mm)
    #   all 30 frames finite; no infeasible solves
    # The hard band Aproj_k @ dqa in [bproj_k - tol, bproj_k + tol] (one per
    # carrier/link) replaces hundreds of near-parallel per-point soft rows that
    # wrecked CLARABEL's conditioning. robot_only (no object_sdf) has the flag
    # forced off by from_config, so its solve is bit-exact with the parity baseline.
    # persistence_tol: band half-width in metres (5 mm validated; raise to 1 cm if
    # a task geometry makes the 5 mm band infeasible).
    activate_persistence: bool = True
    persistence_tol: float = 0.005

    # Brick 5 — Movable entities W^o (object motion regularization).
    # Enabled by default after validation (2026-06-14): 30-frame object_interaction
    # sub3_largebox_003 with lambda_o=1.0/lambda_omega=1.0 shows:
    #   mean solved-object position error vs reference: 0.0004 m (< 0.15 m limit)
    #   mean contact gap |d_robot - d_ref|: 0.07183 m (vs 0.07202 m off; slightly better)
    #   pelvis z in [0.360, 0.812] m; all qpos finite; runtime ~1.1 s/frame.
    # The object stays tightly near its reference trajectory (W^o regularization)
    # while the bilateral D/X coupling keeps contact tracking at least as good.
    # robot_only has no object (obj_pose is None) so its solve is structurally
    # unaffected; the parity test (test_test_socp_parity.py) remains bit-exact.
    # Only active on object tasks (obj_pose is not None) from frame_idx >= 2.
    #   W^o = lambda_o     * ||vdot_obj - vdot_ref||^2  (linear acceleration tracking)
    #         + lambda_omega * ||omega_obj - omega_ref||^2  (angular velocity tracking)
    activate_movable: bool = True
    lambda_o: float = 1.0
    lambda_omega: float = 1.0
    # W^o position anchor. W^o (lambda_o/lambda_omega) regularizes only the
    # object's acceleration/velocity, which is invariant to a constant position
    # offset, so with the bilateral D/X coupling free to move the object the
    # solved object pose drifts in absolute position while still matching the
    # reference acceleration (the same position-blindness as centroidal W^c).
    # This anchor pins the absolute object position to the reference path. It is
    # the object analogue of lambda_c_pos. Tuned 2026-06-14 by sweep on
    # sub3_largebox_003 (30 frames, all bricks on incl. persistence):
    #   lambda_o_pos=0:   obj err mean=267.6 mm, contact gap=75.59 mm
    #   lambda_o_pos=1:   obj err mean=  0.8 mm, contact gap=54.37 mm
    #   lambda_o_pos=10:  obj err mean=  0.5 mm, contact gap=54.34 mm  <-- chosen
    #   lambda_o_pos>=50: saturated (no further change).
    # The anchor not only removes the drift but IMPROVES contact tracking (the gap
    # was previously measured against a drifted object pose). 10.0 sits safely in
    # the saturated regime. The persistence hard constraint pins the robot's
    # contact points, so without this anchor the bilateral D/X coupling offsets
    # the (position-blind) object instead; this term resolves that coupling.
    lambda_o_pos: float = 10.0
