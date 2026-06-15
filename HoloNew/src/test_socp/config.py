"""TEST-SOCP retargeter config.

Flat and explicit: every field maps 1:1 to one solver effect. The builder passes them
through unchanged (no hidden presets/rewrites); illegal combinations raise a ValueError
in from_config instead of being silently fixed.

Organised by family, following the article's formulation:
    §0 Preprocess               world placement (scale stage), upstream of the solve
    §1 Variables                what the solve optimises: q_a, T_B, T_m
    §2 Weights                  GMR tracking → Holosoma (TODO) → our article terms
    §3 Constraints              Holosoma → GMR → our customs
    §4 Solver                   SQP mechanics

Defaults are the GMR BASELINE: the bare GMR-SOCP objective (per-point position +
orientation tracking, §2) + joint limits, nothing else. Every brick below defaults OFF;
add them back one at a time. Re-enable hints carry the tuned value.
"""
from __future__ import annotations

from dataclasses import dataclass

from HoloNew.config_types.retargeter import RetargeterConfig


@dataclass(frozen=True)
class TestSocpRetargeterConfig(RetargeterConfig):
    # =====================================================================
    # §0 — PREPROCESS (world placement, applied in the scale stage upstream of the solve)
    # Robot-root and object placement, per axis (same convention for both):
    #   None  -> AUTO: from_config computes ROBOT_HEIGHT / human_height (per clip), the
    #            physically-correct scale that puts the body/object at robot scale.
    #            Nothing hardcoded — the height comes from the clip + the robot model.
    #   float -> explicit multiplier on the raw grounded axis (1.0 = raw).
    # Defaults: robot base Z auto (None); robot XY raw (1.0) so the GMR targets and the
    # contact field share one world frame. The object is kept raw (1.0 / 1.0) for our
    # case so its trajectory is unchanged; set None on an axis to scale it like the robot.
    # =====================================================================
    scale_xy_robot: float | None = 1.0
    scale_z_robot: float | None = None
    scale_xy_object: float | None = 1.0
    scale_z_object: float | None = 1.0

    # =====================================================================
    # §1 — VARIABLES (the decision variables of the solve, as in the article)
    #   q_a : actuated joints       — activate_qa (False freezes the joints: dqa_joints = 0)
    #   T_B : pelvis / floating base — activate_tb (False freezes the base: dqa_base = 0)
    #   T_m : movable object pose    — activate_tm (True makes the object a variable; its
    #         W^o weights live in §2). Object-as-interaction-entity is handled separately.
    # =====================================================================
    activate_qa: bool = True
    activate_tb: bool = True
    activate_tm: bool = False

    # =====================================================================
    # §2 — WEIGHTS (GMR tracking → Holosoma → our article terms)
    # =====================================================================
    # --- GMR: per-point position (w_p) / orientation (w_r) tracking. Toggle each channel
    # on/off; the weight VALUES stay in IK_MATCH_TABLE1/2. On by default (core objective).
    activate_pos_tracking: bool = True
    activate_rot_tracking: bool = True

    # --- Holosoma: TODO — not yet computed by TEST-SOCP. Port the Laplacian interaction-
    # mesh deformation, nominal tracking, Q_diag joint regularization, and 1st-order
    # smoothness as optional bricks, then expose their weights here.

    # --- Our article terms (LaTeX order). Each has its own activate_w<sym> switch and a
    # lowercase lambda_<sym> weight; the switch alone decides (weight = tuned value when
    # on). All default OFF (GMR baseline).
    #
    # W^s — Style. activate_ws swaps plain world-frame tracking for the Style objective
    # (pelvis orientation = roll/pitch tilt only; joint positions dropped).
    # style_pelvis_relative (only when activate_ws): True = joint orientations re-based by
    # the current pelvis + weak pelvis scaffold (pelvis_anchor_weight*w_p); False = world
    # frame + full world pelvis position (GMR-like, pelvis_anchor_weight unused).
    activate_ws: bool = False
    style_pelvis_relative: bool = False
    pelvis_anchor_weight: float = 10.0

    # W^D / W^X / W^P — Interaction. D = normal proximity, X = tangential placement,
    # P = soft persistence (prefer the hard constraint in §3). Need an interaction entity
    # + non-penetration. sigma_v scales P. (lambda_p untuned placeholder.)
    activate_wd: bool = False
    lambda_d: float = 20.0
    activate_wx: bool = False
    lambda_x: float = 20.0
    activate_wp: bool = False
    lambda_p: float = 20.0
    sigma_v: float = 0.05

    # W^c / W^L — Centroidal (frame >= 2 for W^c/W^L; W^c_pos from frame 0):
    #   W^c     = lambda_c     * ||c_ddot - c_ddot_ref||^2   (CoM accel)
    #   W^c_pos = lambda_c_pos * ||c - c_ref||^2             (CoM position anchor)
    #   W^L     = lambda_l     * ||L||^2                      (angular momentum -> 0)
    # activate_wl_track: track the reference momentum L_ref instead of driving L->0;
    # lambda_l_track weights ||L_lumped - L_ref||^2. (lambda_c_pos untuned placeholder.)
    activate_wc: bool = False
    lambda_c: float = 1e-5
    activate_wc_pos: bool = False
    lambda_c_pos: float = 1.0
    activate_wl: bool = False
    lambda_l: float = 1e-4
    activate_wl_track: bool = False
    lambda_l_track: float = 5.0

    # W^o — Movable object (needs activate_tm in §1):
    #   W^o = lambda_o * ||vdot_obj - vdot_ref||^2 + lambda_omega * ||omega_obj - omega_ref||^2
    # activate_wo_pos / lambda_o_pos: absolute object-position anchor (analogue of W^c_pos).
    # activate_wo_floor / lambda_o_floor: place the object by its floor contact
    # (paper's object<->env pair).
    activate_wo: bool = False
    lambda_o: float = 1.0
    lambda_omega: float = 1.0
    activate_wo_pos: bool = False
    lambda_o_pos: float = 10.0
    activate_wo_floor: bool = False
    lambda_o_floor: float = 5.0

    # W^r — Temporal regularization (tangent-space acceleration). sigma_* = per-DOF noise
    # scale for joints / base.
    activate_wr: bool = False
    lambda_r: float = 0.2
    sigma_qddot: float = 20.0
    sigma_Vdot: float = 20.0

    # =====================================================================
    # §3 — CONSTRAINTS (Holosoma → GMR → our customs)
    # =====================================================================
    # Holosoma-style (need their companion config to act): self-collision needs
    # self_collision=SelfCollisionConfig(...); foot-sticking needs foot_lock + sequences
    # (self_collision, foot_lock, foot_sticking_tolerance are inherited).
    activate_self_collision: bool = False
    activate_foot_sticking: bool = False
    # GMR: actuated joint limits q_a^- <= q_a <= q_a^+.
    activate_joint_limits: bool = True
    # Our customs: non-penetration d_ij >= 0 (required by the interaction costs;
    # penetration_tolerance inherited). load_object_scene: with non-pen + a real object,
    # True = object<->robot non-pen (full geometry), False = ground non-pen only.
    # activate_obj_surface_nonpen: hard d_ij>=0 on the object surface (SLOW).
    # activate_persistence: hard tangential no-slip band (the article's P as a constraint);
    # persistence_tol = band half-width (m). Needs an interaction entity.
    activate_obj_non_penetration: bool = False
    load_object_scene: bool = True
    activate_obj_surface_nonpen: bool = False
    obj_surface_nonpen_tol: float = 0.005
    activate_persistence: bool = False
    persistence_tol: float = 0.005
    # floor_as_entity: add the floor to the interaction entities O (so D/X and the
    # object<->floor contact can act on the floor). Object-as-interaction-entity is TBD;
    # parked here for now. Requires activate_obj_non_penetration.
    floor_as_entity: bool = False

    # =====================================================================
    # §4 — SOLVER (SQP mechanics)
    # fps -> frame timestep dt = 1/fps (every temporal term). n_iter_first /
    # n_iter_per_frame: inner SQP iterations on the first frame vs the rest (per pass).
    # iterate_step_tol: early-stop when the actuated step norm drops below it (0 disables).
    # (step_size is inherited from RetargeterConfig.)
    # =====================================================================
    fps: float = 30.0
    n_iter_first: int = 50
    n_iter_per_frame: int = 10
    iterate_step_tol: float = 0.01
