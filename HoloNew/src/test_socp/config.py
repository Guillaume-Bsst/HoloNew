"""TEST-SOCP retargeter config.

Flat and explicit: every field maps 1:1 to one solver effect. The builder passes them
through unchanged (no hidden presets/rewrites); illegal combinations raise a ValueError
in from_config instead of being silently fixed.

Organised by family, following the article's formulation:
    §1 Variables & entities      what the solve optimises / interacts with
    §2 Tracking weights          GMR (+ Holosoma, TODO) — values from the IK tables
    §3 Our custom weights        the article terms W^s / W^D,X,P / W^c,L / W^o / W^r
    §4 Constraints               Holosoma → GMR → our customs
    §5 Solver                    SQP mechanics
    §6 World placement           preprocessing (scale stage)

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
    # §1 — VARIABLES & ENTITIES
    # Joints q_a and the pelvis pose T_B are always variables (q_a_init_idx, inherited,
    # selects which base DOFs are actuated). The object pose T_m becomes a variable only
    # when activate_movable (a movable entity); floor_as_entity adds the floor to the
    # interaction entities O.
    # =====================================================================
    activate_movable: bool = False
    floor_as_entity: bool = False

    # =====================================================================
    # §2 — TRACKING WEIGHTS (per-point; the weight VALUES live in IK_MATCH_TABLE1/2)
    # =====================================================================
    # GMR: per-point position (w_p) / orientation (w_r) cost. Toggle each channel on/off
    # globally; the table values still apply when on. On by default (core objective).
    activate_pos_tracking: bool = True
    activate_rot_tracking: bool = True
    # Holosoma: TODO — not yet computed by TEST-SOCP. Port the Laplacian interaction-mesh
    # deformation, nominal tracking, Q_diag joint regularization, and 1st-order smoothness
    # as optional bricks, then expose their weights here.

    # =====================================================================
    # §3 — OUR CUSTOM WEIGHTS (article, in the LaTeX order)
    # Every term has its own activate_* switch and its weight. The switch alone decides
    # (the weight defaults to its tuned value); a term applies iff its activate_* is True.
    # All default OFF (GMR baseline).
    # =====================================================================
    # W^s — Style. activate_style swaps plain world-frame tracking for the Style objective
    # (pelvis orientation = roll/pitch tilt only; joint positions dropped).
    # style_pelvis_relative (only when activate_style): True = joint orientations re-based
    # by the current pelvis + weak pelvis scaffold (pelvis_anchor_weight*w_p); False = world
    # frame + full world pelvis position (GMR-like, pelvis_anchor_weight unused).
    activate_style: bool = False
    style_pelvis_relative: bool = False
    pelvis_anchor_weight: float = 10.0

    # W^D / W^X / W^P — Interaction. D = normal proximity, X = tangential placement,
    # P = soft persistence (prefer the hard constraint in §4). Need an object/floor entity
    # + non-penetration. sigma_v scales P. (lambda_P untuned placeholder.)
    activate_d: bool = False
    lambda_D: float = 20.0
    activate_x: bool = False
    lambda_X: float = 20.0
    activate_p: bool = False
    lambda_P: float = 20.0
    sigma_v: float = 0.05

    # W^c / W^L — Centroidal (frame >= 2 for W^c/W^L; W^c_pos from frame 0):
    #   W^c     = lambda_c     * ||c_ddot - c_ddot_ref||^2   (CoM accel)
    #   W^c_pos = lambda_c_pos * ||c - c_ref||^2             (CoM position anchor)
    #   W^L     = lambda_L     * ||L||^2                      (angular momentum -> 0)
    # track_L_ref: track the reference momentum L_ref instead of driving L->0 (flight);
    # lambda_L_track weights ||L_lumped - L_ref||^2. (lambda_c_pos untuned placeholder.)
    activate_wc: bool = False
    lambda_c: float = 1e-5
    activate_wc_pos: bool = False
    lambda_c_pos: float = 1.0
    activate_wl: bool = False
    lambda_L: float = 1e-4
    track_L_ref: bool = False
    lambda_L_track: float = 5.0

    # W^o — Movable object (object tasks only; needs activate_movable in §1):
    #   W^o = lambda_o * ||vdot_obj - vdot_ref||^2 + lambda_omega * ||omega_obj - omega_ref||^2
    # activate_o_pos / lambda_o_pos: absolute object-position anchor (analogue of W^c_pos).
    # activate_object_floor / lambda_object_floor: place the object by its floor contact
    # (paper's object<->env pair). All three require activate_movable (§1).
    activate_wo: bool = False
    lambda_o: float = 1.0
    lambda_omega: float = 1.0
    activate_o_pos: bool = False
    lambda_o_pos: float = 10.0
    activate_object_floor: bool = False
    lambda_object_floor: float = 5.0

    # W^r — Temporal regularization (tangent-space acceleration). sigma_* = per-DOF noise
    # scale for joints / base.
    activate_wr: bool = False
    lambda_r: float = 0.2
    sigma_qddot: float = 20.0
    sigma_Vdot: float = 20.0

    # =====================================================================
    # §4 — CONSTRAINTS (Holosoma → GMR → our customs)
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
    # True = object<->robot non-pen (full geometry), False = ground non-pen only (what D/X
    # wants). activate_obj_surface_nonpen: hard d_ij>=0 on the object surface (SLOW).
    # activate_persistence: hard tangential no-slip band (the article's P delivered as a
    # constraint); persistence_tol = band half-width (m). Needs an object/floor entity.
    activate_obj_non_penetration: bool = False
    load_object_scene: bool = True
    activate_obj_surface_nonpen: bool = False
    obj_surface_nonpen_tol: float = 0.005
    activate_persistence: bool = False
    persistence_tol: float = 0.005

    # =====================================================================
    # §5 — SOLVER (SQP mechanics)
    # fps -> frame timestep dt = 1/fps (every temporal term). n_iter_first / n_iter_per_frame:
    # inner SQP iterations on the first frame vs the rest (per pass). iterate_step_tol:
    # early-stop when the actuated step norm drops below it (0 disables). (step_size is
    # inherited from RetargeterConfig.)
    # =====================================================================
    fps: float = 30.0
    n_iter_first: int = 50
    n_iter_per_frame: int = 10
    iterate_step_tol: float = 0.01

    # =====================================================================
    # §6 — WORLD PLACEMENT (preprocess scale stage), per axis, multiplier on the raw
    # grounded value. 1.0 = raw; None = native morphological scaling. Default: only robot
    # Z is scaled (None, like GMR); robot XY + object XY/Z stay raw, one shared world frame.
    # =====================================================================
    scale_xy_robot: float = 1.0
    scale_z_robot: float | None = None
    scale_xy_object: float = 1.0
    scale_z_object: float = 1.0
