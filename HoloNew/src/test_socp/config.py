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

    # World placement applied inside the preprocess scale stage, independently for the
    # robot root and the object, in XY and Z. Each is a multiplier on the RAW grounded
    # axis (1.0 = raw; <1 pulls toward the origin / floor like holosoma). None means the
    # native morphological scaling (HUMAN_SCALE_TABLE[root]*height_ratio) for that axis.
    # TEST defaults keep the RAW grounded pelvis XY (1.0) so the GMR targets and the
    # SmplxGroundProbe contact field agree (see the lambda_D/X note below); the robot Z
    # keeps its native morphological scaling (None); the object is left raw (1.0).
    # Body proportions (pelvis-local) are unaffected by all four.
    scale_xy_robot: float = 1.0
    scale_z_robot: float | None = None
    scale_xy_object: float = 1.0
    scale_z_object: float = 1.0

    # Interaction cost weights. D (normal proximity) + X (tangential placement)
    # are enabled by default on object tasks; robot_only (no object SDF) is
    # unchanged. The interaction costs REQUIRE the non-penetration constraint to
    # be stable — from_config auto-enables ground non-penetration when interaction
    # is active on an object task (without it the D term marches the floating base
    # through the floor).
    #
    # Weights jointly re-tuned. The interaction must outweigh the other objective
    # terms (Style + W^r + persistence + movable) to actually reduce the contact
    # gap. Re-tuned again 2026-06-14 from 5.0 to 20.0 after removing the holosoma
    # root-XY scale (scale_xy_robot=1.0): aligning the GMR targets to
    # the raw pelvis XY (so targets and interaction fields share one world frame)
    # shifted the contact geometry, and the old lambda=5 no longer beat D/X off.
    # Sweep on sub3_largebox_003 (K=30, aligned frame), object mean contact gap:
    #   lambda=0  (off): 0.0323
    #   lambda=5:        0.0344   (worse than off)
    #   lambda=10:       0.0344
    #   lambda=20:       0.0276   <-- chosen (clear win, moderate weight)
    #   lambda=50:       0.0235   (best, but aggressive)
    # lambda=20 cuts the OBJECT (manipulation) contact gap ~15% below D/X off — D/X's
    # primary job. The FLOOR channel is owned by the persistence no-slip hard
    # constraint, so the acceptance metric tracks the object gap.
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
    lambda_D: float = 20.0
    lambda_X: float = 20.0
    lambda_P: float = 0.0
    sigma_v: float = 0.05

    # Object-surface non-penetration (paper's d_{i,j} >= 0 for the object entity).
    # The D cost only discourages penetration softly; this adds the HARD inequality
    # so robot control points cannot pass through the object surface. Implemented and
    # validated (deepest penetration on sub3_largebox_003 cut -32.5 -> -10.3 mm), but
    # it defaults OFF because it is SLOW: the many near-surface inequalities make
    # CLARABEL fail and fall back to SCS (~13-26 s/frame vs ~1.1 baseline). Same
    # speed/conditioning tradeoff as the soft P cost — kept as an explicit opt-in,
    # not on by default. tol is the allowed signed-distance floor.
    activate_obj_surface_nonpen: bool = False
    obj_surface_nonpen_tol: float = 0.005

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

    # W^L reference tracking (opt-in). The default W^L (lambda_L) drives the angular
    # momentum toward 0; the paper tracks the reference momentum L_ref. When
    # track_L_ref is True, a lumped orbital angular momentum (robot link masses at
    # the 14 mapped bodies) is built for BOTH the reference (from the GMR target
    # trajectory) and the current config (linearized in dqa), and W^L tracks
    # ||L_lumped - L_ref||^2 with weight lambda_L_track. This matters mainly in
    # flight (in stance L is dominated by contacts); validated on aerial SFU clips.
    # Default off so the grounded pipeline is unchanged. See centroidal.py.
    track_L_ref: bool = False
    # Tuned on SFU 0007_Cartwheel001 (60 frames): solved-vs-reference angular-momentum
    # correlation off=-0.03 (Style ignores the aerial spin) -> w=1: 0.76, w=5: 0.97,
    # w=20: 1.00. w=5 reproduces the reference spin cleanly without over-constraining.
    lambda_L_track: float = 5.0

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
    # Object<->floor contact (the paper's object-environment pair). Places the
    # object by its floor contact instead of a positional target: object surface
    # points carried by T_obj query the floor field, resisting any motion that
    # breaks the near-floor contact, and vanishing when the object is lifted (then
    # the object is free, placed by object<->robot contact + ballistic W^o). Used
    # by inertia_mode, which sets lambda_o_pos=0 (drop the anchor) and this > 0 so
    # the object, like the body, is placed by contacts. Default off (parity).
    lambda_object_floor: float = 0.0
