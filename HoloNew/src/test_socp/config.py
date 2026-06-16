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

from dataclasses import dataclass

from HoloNew.config_types.retargeter import RetargeterConfig


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
    activate_rot_tracking: bool = True

    # [HOLO] General joint-space regularizers (flat ports of Holosoma's regularization
    # terms; useful in TEST on their own). NOTE: Holosoma's Laplacian deformation term
    # is intentionally NOT ported — it needs a different preprocess path; use the native
    # Holosoma solver as the comparison for that.
    # W^smooth: step-toward-previous-frame joint smoothness (Holosoma smooth_weight).
    activate_smooth: bool = False
    lambda_smooth: float = 0.2
    # W^qdiag: absolute joint-config regularizer, per-joint weights from MANUAL_COST.
    activate_qdiag: bool = False
    lambda_qdiag: float = 1.0
    # W^nominal: pull NOMINAL_TRACKING_INDICES joints toward a nominal pose, with an
    # exp-decaying weight over SQP iterations (Holosoma w_nominal_tracking_init / tau).
    activate_nominal: bool = False
    lambda_nominal: float = 5.0
    nominal_tau: float = 10.0
    
    # [TEST] W^s Style: swaps world-frame tracking for pelvis tilt-only orientation (joint
    # positions dropped). style_pelvis_relative: re-base joint orientations on the current
    # pelvis + weak scaffold (pelvis_anchor_weight*w_p); False = world frame + full pelvis position.
    activate_ws: bool = False
    style_pelvis_relative: bool = False
    pelvis_anchor_weight: float = 10.0

    # [TEST] W^D/W^X/W^P Interaction: carrier control points query each target's signed field.
    # Targets = floor (always) + object; carriers = robot (always) + object. D = normal
    # proximity, X = tangential placement, P = soft persistence (prefer the §3 hard constraint).
    # Robot channel shares lambda_d/lambda_x; sigma_v scales P.
    activate_wd: bool = False
    lambda_d: float = 20.0
    activate_wx: bool = False
    lambda_x: float = 20.0
    activate_wp: bool = False
    lambda_p: float = 20.0
    sigma_v: float = 0.05
    #object carrier → floor (object<->environment pair); separate weight, needs activate_tm.
    activate_obj_floor: bool = False
    lambda_obj_floor: float = 5.0

    # [TEST] W^c/W^L Centroidal (W^c/W^L from frame >= 2, W^c_pos from frame 0):
    # W^c = ||c_ddot - c_ddot_ref||^2, W^c_pos = ||c - c_ref||^2, W^L = ||L||^2.
    # activate_wl_track: track L_ref (||L_lumped - L_ref||^2) instead of driving L->0.
    activate_wc: bool = False
    lambda_c: float = 1e-5
    activate_wc_pos: bool = False
    lambda_c_pos: float = 1.0
    activate_wl: bool = False
    lambda_l: float = 1e-4
    activate_wl_track: bool = False
    lambda_l_track: float = 5.0

    # [TEST] W^o Movable-object motion reg (needs activate_tm):
    # lambda_o*||vdot - vdot_ref||^2 + lambda_omega*||omega - omega_ref||^2.
    # activate_wo_pos: absolute object-position anchor. (object<->floor contact = activate_obj_floor.)
    activate_wo: bool = False
    lambda_o: float = 1.0
    lambda_omega: float = 1.0
    activate_wo_pos: bool = False
    lambda_o_pos: float = 10.0

    # [TEST] W^r Temporal reg (tangent-space accel); sigma_* = per-DOF noise scale (joints / base).
    # lambda_r re-tuned to 0.5 for the single-pass solve (the old two-pass value 0.2 is
    # counterproductive single-pass; 0.5 cuts jerk ~47% within the tracking slack).
    activate_wr: bool = False
    lambda_r: float = 0.5
    sigma_qddot: float = 20.0
    sigma_Vdot: float = 20.0

    # === §3 CONSTRAINTS ===

    # [HOLO] need their companion config: self_collision=SelfCollisionConfig(...),
    # foot-sticking needs foot_lock + sequences (both inherited).
    activate_self_collision: bool = False
    activate_foot_sticking: bool = False

    # [HOLO] non-penetration d_ij >= -tolerance: phi from MuJoCo collision (mj_collision +
    # mj_geomDistance) on the loaded geometry, with our object/ground pair filter. Inherited
    # from RetargeterConfig. load_object_scene is an addon selecting the geometry MuJoCo
    # measures against (xml swap): True = object<->robot (full object xml), False = ground only.
    activate_obj_non_penetration: bool = False
    load_object_scene: bool = False

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
    # (step_size is inherited from Holosoma's RetargeterConfig.)
    fps: float = 30.0
    n_iter_first: int = 50
    n_iter_per_frame: int = 10
    iterate_step_tol: float = 0.01
