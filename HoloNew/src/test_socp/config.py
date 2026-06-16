"""TEST-SOCP retargeter config.

Flat and explicit: every field maps 1:1 to one solver effect; the builder passes them
through unchanged. Defaults = GMR baseline (position + orientation tracking + joint
limits); every other brick defaults OFF, add them back one at a time.

PROVENANCE — each field is tagged by where it comes from:
    [GMR]   baseline tracking objective, inherited from GMR-SOCP
    [HOLO]  ported / inherited from Holosoma (machinery in holosoma_constraints.py)
    [TEST]  TEST-SOCP's own: the article terms + our customs

Layout follows the solve pipeline (§0 preprocess → §1 variables → §2 weights →
§3 constraints → §4 solver), so a section can mix tags; read the tag per field.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from HoloNew.config_types.retargeter import FootLockConfig, RetargeterConfig


@dataclass(frozen=True)
class TestSocpRetargeterConfig(RetargeterConfig):
    
    # === §0 PREPROCESS — world placement (scale stage, upstream of the solve) ===

    # [TEST] per axis, robot and object share the convention: None = AUTO
    # (ROBOT_HEIGHT/human_height from the clip), float = multiplier on the raw axis (1.0 = raw).
    scale_xy_robot: float | None = 1.0
    scale_z_robot: float | None = None
    scale_xy_object: float | None = 1.0
    scale_z_object: float | None = 1.0

    # === §1 VARIABLES — what the solve optimises ===

    # [TEST] article formulation: q_a actuated joints / T_B floating base / T_m movable
    # object pose. False freezes each; activate_tm makes the object a variable (W^o in §2).
    activate_qa: bool = True
    activate_tb: bool = True
    activate_tm: bool = False

    # === §2 WEIGHTS ===

    # [GMR] per-point position / orientation tracking (values in IK_MATCH_TABLE1/2). Core objective.
    activate_pos_tracking: bool = True
    lambda_pos: float = 1.0
    sigma_p: float = 1.0
    activate_rot_tracking: bool = True
    lambda_rot: float = 1.0
    sigma_rot: float = 1.0

    # [HOLO] General joint-space regularizers (flat ports of Holosoma's regularization
    # terms; useful in TEST on their own). NOTE: Holosoma's Laplacian deformation term
    # is intentionally NOT ported — it needs a different preprocess path; use the native
    # Holosoma solver as the comparison for that.
    # W^smooth: step-toward-previous-frame joint smoothness (Holosoma smooth_weight).
    activate_smooth: bool = False
    lambda_smooth: float = 1.0
    # W^qdiag: absolute joint-config regularizer, per-joint weights from MANUAL_COST.
    activate_qdiag: bool = False
    lambda_qdiag: float = 1.0
    # W^nominal: pull NOMINAL_TRACKING_INDICES joints toward a nominal pose, with an
    # exp-decaying weight over SQP iterations (Holosoma w_nominal_tracking_init / tau).
    activate_nominal: bool = False
    lambda_nominal: float = 1.0
    nominal_tau: float = 10.0
    
    # [TEST] σ characteristic scales — FLAT constants, none auto-computed.
    # Each residual is divided by its σ so λ is a pure fps-invariant priority.
    sigma_R: float = 0.2        # style: orientation error (rad)
    sigma_a: float = 9.81       # W^c: CoM accel (m/s²), = g
    sigma_L: float = 10.0       # W^L: angular momentum (kg·m²/s), hand-set
    sigma_ao: float = 9.81      # W^o linear: object accel (m/s²), = g
    sigma_omega: float = 6.283185307179586  # W^o angular: spin (rad/s), = 2π

    # [TEST] W^s Style (additive, flat): ADDS pelvis-relative joint-orientation matching
    # (S_k) + pelvis-tilt against gravity (S_B) ON TOP of the GMR world tracking — no mode
    # swap. Per-body weights are normalized internally (sum to 1) so lambda_ws is a pure
    # priority (lambda^s in the spec). Activate + weight, like the other custom weights.
    activate_ws: bool = False
    lambda_ws: float = 1.0      # seed (σ_R folded); validate via scoreboard
    # Intra-style distribution ω_k^s: None = the uniform STYLE_WEIGHT_TABLE (default,
    # == legacy). Pass a dict {robot_body -> weight, "__pelvis_tilt__" -> ω_B} to re-weight
    # (e.g. arms > legs). Normalized internally (Σω=1).
    style_weights: dict | None = None

    # [TEST] W^D/W^X/W^P Interaction: carrier control points query each target's signed field.
    # Targets = floor (always) + object; carriers = robot (always) + object. D = normal
    # proximity, X = tangential placement, P = soft persistence (prefer the §3 hard constraint).
    # Robot channel shares lambda_d/lambda_x; sigma_v scales P.
    activate_wd: bool = False
    lambda_d: float = 1.0
    activate_wx: bool = False
    lambda_x: float = 1.0
    activate_wp: bool = False
    lambda_p: float = 1.0       # seed; P now (σ_v·dt)²-normalized
    sigma_v: float = 0.05
    # [TEST] per-entity field range Lⱼ (activation distance AND positional scale).
    # None = AUTO: inherit the SDF probe margin (current shared value).
    L_floor: float | None = None
    L_object: float | None = None
    #object carrier → floor (object<->environment pair); separate weight, needs activate_tm.
    activate_obj_floor: bool = False
    lambda_obj_floor: float = 1.0

    # [TEST] W^c/W^L Centroidal (W^c/W^L from frame >= 2, W^c_pos from frame 0):
    # W^c = ||c_ddot - c_ddot_ref||^2, W^c_pos = ||c - c_ref||^2, W^L = ||L||^2.
    # activate_wl_track: track L_ref (||L_lumped - L_ref||^2) instead of driving L->0.
    activate_wc: bool = False
    lambda_c: float = 1.0       # seed; was 1e-5
    activate_wc_pos: bool = False
    lambda_c_pos: float = 1.0
    activate_wl: bool = False
    lambda_l: float = 1.0       # seed; was 1e-4
    activate_wl_track: bool = False
    lambda_l_track: float = 1.0

    # [TEST] W^o Movable-object motion reg (needs activate_tm):
    # lambda_o * ( ||(vdot - vdot_ref)/sigma_ao||^2 + ||(omega - omega_ref)/sigma_omega||^2 ).
    # Single lambda_o; sigma_ao/sigma_omega carry the linear/angular asymmetry (Task 1.6).
    # activate_wo_pos: absolute object-position anchor. (object<->floor contact = activate_obj_floor.)
    activate_wo: bool = False
    lambda_o: float = 1.0       # seed (collapsed; σ_ao/σ_omega folded)
    activate_wo_pos: bool = False
    lambda_o_pos: float = 1.0

    # [TEST] W^r Temporal reg (tangent-space accel); sigma_* = per-DOF noise scale (joints / base).
    # lambda_r re-tuned to 0.5 for the single-pass solve (the old two-pass value 0.2 is
    # counterproductive single-pass; 0.5 cuts jerk ~47% within the tracking slack).
    activate_wr: bool = False
    lambda_r: float = 1.0
    sigma_qddot: float = 20.0
    sigma_Vdot: float = 20.0

    # === §3 CONSTRAINTS ===

    # [HOLO] self-collision companion config (inherited SelfCollisionConfig for geom pairs;
    # self_collision_margin surfaces its tolerance flat).
    activate_self_collision: bool = False
    self_collision_margin: float = 0.02
    # [HOLO] foot-sticking: per-frame XY no-slip on planted feet (sequence auto-built from the
    # source). foot_sticking_tolerance = XY band half-width (m).
    activate_foot_sticking: bool = False
    foot_sticking_tolerance: float = 1e-3
    # [HOLO] foot-lock: pin a foot's Z to z_floor over configured frame windows. Re-surfaced
    # from the inherited RetargeterConfig so it is settable/visible here. Off until
    # foot_lock.enable=True (set windows / z_floor / tolerance on the FootLockConfig).
    foot_lock: FootLockConfig = field(default_factory=FootLockConfig)

    # [HOLO] non-penetration d_ij >= -tolerance: phi from MuJoCo collision (mj_collision +
    # mj_geomDistance) on the loaded geometry, with our object/ground pair filter. Inherited
    # from RetargeterConfig. load_object_scene is an addon selecting the geometry MuJoCo
    # measures against (xml swap): True = object<->robot (full object xml), False = ground only.
    activate_obj_non_penetration: bool = False
    load_object_scene: bool = False
    # penetration_tolerance: allowed signed-distance slack d_ij >= -tolerance (m). Re-surfaced
    # from the inherited RetargeterConfig.
    penetration_tolerance: float = 0.001

    # [GMR] actuated joint limits.
    activate_joint_limits: bool = True

    # [TEST] obj_surface_nonpen = hard d_ij>=0 on the object surface from the SDF/contact
    # field (not MuJoCo; SLOW). persistence = hard tangential no-slip band (article's P as a
    # constraint, tol = band half-width m, needs an interaction entity).
    activate_obj_surface_nonpen: bool = False
    obj_surface_nonpen_tol: float = 0.005
    activate_persistence: bool = False
    persistence_tol: float = 0.005
    # The floor is ALWAYS an interaction target (no floor_as_entity flag); every carrier contacts it.

    # === §4 SOLVER (SQP mechanics) ===

    # [TEST] fps -> dt = 1/fps. n_iter_first / n_iter_per_frame: inner SQP iterations on frame
    # 0 vs rest. iterate_step_tol: early-stop when the actuated step norm drops below it (0 off).
    fps: float = 30.0
    n_iter_first: int = 50
    n_iter_per_frame: int = 10
    iterate_step_tol: float = 0.01
    # step_size: SOC trust-region radius ||dqa|| <= step_size per SQP iteration. Re-surfaced
    # from the inherited RetargeterConfig.
    step_size: float = 0.2
