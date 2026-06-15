"""TEST-SOCP retargeter config.

Flat and explicit: every field maps 1:1 to one solver effect. The builder passes them
through unchanged (no hidden presets/rewrites); illegal combinations raise a ValueError
in from_config instead of being silently fixed.

Defaults are the GMR BASELINE: the bare GMR-SOCP objective (world-frame position +
orientation tracking of every mapped body) + joint limits, nothing else. Add the TEST
bricks back one at a time via the switches below. Re-enable hints (tuned value):
    Style objective        activate_style=True (+ style_pelvis_relative)
    Interaction D / X      lambda_D / lambda_X  = 20.0
    Temporal W^r           lambda_r             = 0.2
    Persistence (no-slip)  activate_persistence=True
    Movable W^o            activate_movable=True
    W^o position anchor    lambda_o_pos         = 10.0
    Centroidal W^c / W^L   activate_centroidal=True
    Floor contact entity   floor_as_entity=True
    Object<->floor contact lambda_object_floor  = 5.0
The old `inertia_mode` preset is gone; for paper-faithful placement set those fields
explicitly (see tests/paper_placement.py).
"""
from __future__ import annotations

from dataclasses import dataclass

from HoloNew.config_types.retargeter import RetargeterConfig


@dataclass(frozen=True)
class TestSocpRetargeterConfig(RetargeterConfig):
    # --- Solve mechanics. fps -> frame timestep dt=1/fps (used by every temporal term).
    # n_iter_first / n_iter_per_frame: SQP inner iterations on the first frame vs the rest
    # (per pass). iterate_step_tol: early-stop when the actuated step norm drops below it
    # (0 disables). (step_size, penetration_tolerance, foot_sticking_tolerance are inherited
    # from RetargeterConfig.)
    fps: float = 30.0
    n_iter_first: int = 50
    n_iter_per_frame: int = 10
    iterate_step_tol: float = 0.01

    # --- Constraints (opt-in; a True flag needs its companion config to do anything) ---
    activate_obj_non_penetration: bool = False
    activate_foot_sticking: bool = False
    activate_self_collision: bool = False
    # With non-penetration on + a real object: True = object<->robot non-pen (full object
    # geometry); False = ground non-pen only (plain model, what D/X interaction wants).
    load_object_scene: bool = True

    # --- World placement (preprocess scale stage), per axis, multiplier on raw grounded.
    # 1.0 = raw; None = native morphological scaling. Default: only robot Z is scaled
    # (None, like GMR); robot XY + object XY/Z stay raw so all share one world frame.
    scale_xy_robot: float = 1.0
    scale_z_robot: float | None = None
    scale_xy_object: float = 1.0
    scale_z_object: float = 1.0

    # --- GMR base tracking: the per-point position (w_p) / orientation (w_r) cost, with
    # the weight VALUES read from IK_MATCH_TABLE1/2 (unchanged). These toggle each channel
    # on/off globally; the table values still apply when on. On by default (core objective).
    activate_pos_tracking: bool = True
    activate_rot_tracking: bool = True

    # --- Tracking objective. activate_style swaps plain world-frame tracking for the
    # Style objective (pelvis orientation = roll/pitch tilt only; joint positions dropped).
    # style_pelvis_relative (only when activate_style): True = joint orientations re-based
    # by the current pelvis + weak pelvis scaffold (pelvis_anchor_weight*w_p); False = joint
    # orientations in world frame + full world pelvis position (GMR-like).
    activate_style: bool = False
    style_pelvis_relative: bool = False
    pelvis_anchor_weight: float = 10.0

    # --- Temporal regularization W^r (tangent-space acceleration). sigma_* = per-DOF
    # noise scale for joints / base. Only active when lambda_r > 0.
    lambda_r: float = 0.0
    sigma_qddot: float = 20.0
    sigma_Vdot: float = 20.0

    # --- Interaction costs. D = normal proximity, X = tangential placement (require an
    # object/floor entity + non-penetration). lambda_P = soft persistence (prefer the hard
    # activate_persistence instead). sigma_v: unused by the hard constraint (API compat).
    lambda_D: float = 0.0
    lambda_X: float = 0.0
    lambda_P: float = 0.0
    sigma_v: float = 0.05

    # --- Object-surface non-penetration (hard d_ij >= 0 on the object). Off: SLOW (SCS
    # fallback ~13-26 s/frame). tol = allowed signed-distance floor.
    activate_obj_surface_nonpen: bool = False
    obj_surface_nonpen_tol: float = 0.005

    # --- Centroidal terms (frame >= 2 for W^c/W^L; W^c_pos from frame 0):
    #   W^c = lambda_c * ||c_ddot - c_ddot_ref||^2   (CoM accel)
    #   W^c_pos = lambda_c_pos * ||c - c_ref||^2      (CoM position anchor)
    #   W^L = lambda_L * ||L||^2                      (angular momentum -> 0)
    activate_centroidal: bool = False
    lambda_c: float = 0.0
    lambda_c_pos: float = 0.0
    lambda_L: float = 0.0

    # W^L reference tracking: track the reference momentum L_ref instead of driving L->0
    # (matters in flight). lambda_L_track weights ||L_lumped - L_ref||^2.
    track_L_ref: bool = False
    lambda_L_track: float = 5.0

    # --- Floor as a contact entity for ANY task (not just object tasks), so D/X and the
    # object<->floor contact can act on the floor. Requires activate_obj_non_penetration.
    floor_as_entity: bool = False

    # --- Contact persistence: hard tangential no-slip band. persistence_tol = band
    # half-width (m). Needs an object/floor entity.
    activate_persistence: bool = False
    persistence_tol: float = 0.005

    # --- Movable object W^o (object tasks only): regularize the object's motion.
    #   lambda_o * ||vdot_obj - vdot_ref||^2 + lambda_omega * ||omega_obj - omega_ref||^2
    # lambda_o_pos: absolute object-position anchor to the reference path (analogue of
    # lambda_c_pos). lambda_object_floor: place the object by its floor contact instead
    # of a positional anchor (paper's object-environment pair).
    activate_movable: bool = False
    lambda_o: float = 1.0
    lambda_omega: float = 1.0
    lambda_o_pos: float = 0.0
    lambda_object_floor: float = 0.0
