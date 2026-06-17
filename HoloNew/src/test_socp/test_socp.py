"""TEST-SOCP retargeter — the paper-formulation solver (G. Besset, "Kinematic
Retargeting Formulation for the G1 Robot").

TestSocpRetargeter is a per-frame, pinocchio-backed SQP: each frame solves a
linearised trust-region SOCP whose objective sums the paper's components — pelvis-
relative Style, interaction D/X/P, temporal W^r, centroidal W^c/W^L, and movable
W^o — over the tangent step ``dqa`` (and the object step ``dxi``). This file holds
the solver itself: __init__, the FK helpers, the objective assembly
(solve_single_iteration), the SQP loop (iterate), and the per-frame driver
(retarget).

Related modules: builder.py (from_config construction + asset/data loading),
holosoma_constraints.py (default-off foot-lock / self-collision mixin),
interaction.py / centroidal.py / movable.py / temporal.py / style.py (the
per-component term builders), pin_model.py (the pinocchio kinematics backend).
"""
from __future__ import annotations

from pathlib import Path
from types import ModuleType

import cvxpy as cp
import mujoco
import numpy as np
import pinocchio as pin

from HoloNew.config_types.retargeter import FootLockConfig, SelfCollisionConfig
from .holosoma_constraints import HolosomaConstraintsMixin
from .tables import IK_MATCH_TABLE1, STYLE_WEIGHT_TABLE

# Body name remapping: keys are IK table frame names; values are actual G1
# MuJoCo body names.  Only entries that differ from the table key are listed.
# GMR's smplx_to_g1.json uses "left_toe_link" / "right_toe_link" but the G1
# model (g1_29dof.xml) does not have those bodies — the most distal foot body
# is left_ankle_roll_link / right_ankle_roll_link.
_BODY_NAME_REMAP: dict[str, str] = {
    "left_toe_link": "left_ankle_roll_link",
    "right_toe_link": "right_ankle_roll_link",
}


class TestSocpRetargeter(HolosomaConstraintsMixin):
    """Position + orientation tracking SOCP retargeter (the TEST-SOCP experiment).

    Solves a two-pass linearised IK problem using a trust-region SOCP.
    The objective is a sum of weighted squared-error terms (one per robot
    frame), combining position and orientation residuals.  Frame targets
    are produced by build_frame_targets and have the form:

        {frame: (p_target (3,), R_target (3,3), w_p, w_r)}

    where ``w_p`` weights the translational term and ``w_r`` the rotational
    term.  Either weight may be zero to disable that term.
    """

    # Not a pytest test class despite the "Test" prefix (TEST-SOCP solver).
    __test__ = False

    def __init__(
        self,
        task_constants: ModuleType,
        object_urdf_path: str | None,
        q_a_init_idx: int = -7,
        activate_joint_limits: bool = True,
        step_size: float = 0.2,
        n_iter_first: int = 50,
        n_iter_per_frame: int = 10,
        iterate_step_tol: float = 0.01,
        activate_obj_non_penetration: bool = False,
        activate_self_collision: bool = False,
        activate_foot_sticking: bool = False,
        penetration_tolerance: float = 1e-3,
        foot_sticking_tolerance: float = 1e-3,
        foot_lock=None,            # FootLockConfig | None
        self_collision=None,       # SelfCollisionConfig | None
        lambda_d: float = 0.0,
        lambda_x: float = 0.0,
        lambda_p: float = 0.0,
        sigma_v: float = 0.05,
        lambda_r: float = 0.0,
        sigma_qddot: float = 1.0,
        sigma_Vdot: float = 1.0,
        lambda_smooth: float = 0.0,
        lambda_qdiag: float = 0.0,
        lambda_nominal: float = 0.0,
        nominal_tau: float = 10.0,
        activate_pos_tracking: bool = True,
        activate_rot_tracking: bool = True,
        lambda_pos: float = 1.0,
        sigma_p: float = 1.0,
        lambda_rot: float = 1.0,
        sigma_rot: float = 1.0,
        lambda_ws: float = 0.0,
        style_weights: dict | None = None,
        sigma_R: float = 0.2,
        sigma_a: float = 9.81,
        sigma_L: float = 10.0,
        sigma_ao: float = 9.81,
        sigma_omega: float = 6.283185307179586,
        activate_centroidal: bool = False,
        lambda_c: float = 0.0,
        lambda_c_pos: float = 0.0,
        lambda_l: float = 0.0,
        lambda_cv: float = 0.0,
        sigma_cv: float = 1.0,
        activate_wl_track: bool = False,
        lambda_l_track: float = 1.0,
        activate_qa: bool = True,
        activate_tb: bool = True,
        activate_tm: bool = False,
        lambda_o: float = 0.0,
        lambda_o_pos: float = 0.0,
        lambda_d_obj: float = 0.0,
        lambda_x_obj: float = 0.0,
        lambda_p_obj: float = 0.0,
        sigma_v_obj: float = 0.05,
        activate_obj_surface_nonpen: bool = False,
        obj_surface_nonpen_tol: float = 0.005,
        load_object_scene: bool = True,
        activate_persistence: bool = False,
        persistence_tol: float = 0.005,
        L_interaction: float = 0.10,
        L_floor: float | None = None,
        L_object: float | None = None,
        sdf_resolution: float = 0.01,
        **_ignored,
    ):
        """Initialise the retargeter from task constants.

        Args:
            task_constants: SimpleNamespace with at minimum ROBOT_URDF_FILE,
                ROBOT_DOF, MANUAL_LB, MANUAL_UB, MANUAL_COST, and
                NOMINAL_TRACKING_INDICES.
            object_urdf_path: Ignored for robot_only; accepted for API compat.
            q_a_init_idx: First actuated index relative to joint start (-7 =
                include floating base).
            activate_joint_limits: Whether to add joint-limit constraints.
            step_size: Trust-region radius for the SOCP.
            **_ignored: Any remaining kwargs are silently ignored so callers
                can pass a superset of kwargs without error.
        """
        self.task_constants = task_constants
        self.activate_joint_limits = activate_joint_limits
        self.step_size = step_size
        # SQP inner iterations per pass: more on the first frame (cold start), fewer after.
        self.n_iter_first = n_iter_first
        self.n_iter_per_frame = n_iter_per_frame
        # SQP inner-loop early-stop: break when the actuated step norm falls below
        # this (catches passes plateauing at the trust-region boundary). 0 disables.
        self._iterate_step_tol = iterate_step_tol
        self.visualize = False
        self.demo_joints = task_constants.DEMO_JOINTS

        # object_name must be known before the model is loaded so the gated xml
        # selection below can reference it.
        self.object_name = getattr(task_constants, "OBJECT_NAME", "ground")

        # Load MuJoCo model.  Default (flag off or ground): plain robot xml.
        # When activate_obj_non_penetration is on AND the task has a real object
        # AND load_object_scene is True, swap to the object-scene xml so the
        # object collision geometry is present (object hard non-penetration).
        # The interaction coupling uses load_object_scene=False: it wants only the
        # ground non-penetration (the plain g1 xml already has a ground plane) to
        # keep the solve stable, and lets the soft D term handle object contact —
        # the object hard constraint conflicts with the D pull and trips CLARABEL.
        robot_xml_path = task_constants.ROBOT_URDF_FILE.replace(".urdf", ".xml")
        if activate_obj_non_penetration and load_object_scene and self.object_name not in (None, "ground"):
            if self.object_name == "multi_boxes":
                robot_xml_path = task_constants.SCENE_XML_FILE
            else:
                robot_xml_path = task_constants.ROBOT_URDF_FILE.replace(
                    ".urdf", "_w_" + self.object_name + ".xml"
                )
        self.robot_model = mujoco.MjModel.from_xml_path(robot_xml_path)
        print(f"[TestSocp] Loading robot model from: {robot_xml_path}")
        self.robot_data = mujoco.MjData(self.robot_model)

        if self.robot_data.qpos.shape[0] > 7 + task_constants.ROBOT_DOF:
            self.has_dynamic_object = True
        else:
            self.has_dynamic_object = False

        self.nq = self.robot_model.nq
        self.q_a_init_idx = q_a_init_idx
        self.q_a_indices = np.arange(7 + q_a_init_idx, 7 + task_constants.ROBOT_DOF)
        self.nq_a = len(self.q_a_indices)

        # --- pinocchio backend ---
        from HoloNew.src.test_socp.pin_model import PinModel
        self.pin = PinModel(task_constants.ROBOT_URDF_FILE)
        self.pin.bind_mujoco_order(self.robot_model)
        if self.q_a_init_idx == -7:
            # Full floating-base + joint tangent: base 6 DOF + 29 joint DOFs = 35.
            self.v_a_indices = np.arange(0, 6 + task_constants.ROBOT_DOF)
        else:
            # Joints-from-index: base excluded; qpos joint j -> tangent j-1.
            self.v_a_indices = np.arange(6 + self.q_a_init_idx, 6 + task_constants.ROBOT_DOF)
        self.nv_a = len(self.v_a_indices)

        # Joint limits — kept in MuJoCo qpos space for joint-range extraction;
        # also build tangent-space bounds for the pinocchio joint limit constraint.
        n_floating_base = 7
        joint_names = [self.robot_model.joint(i).name for i in range(self.robot_model.njnt)]
        actuated_joints = [(i, name) for i, name in enumerate(joint_names) if name]
        large_number = 1e6
        complete_lower = np.concatenate(
            [-large_number * np.ones(n_floating_base),
             self.robot_model.jnt_range[[i for i, _ in actuated_joints], 0]]
        )
        complete_upper = np.concatenate(
            [large_number * np.ones(n_floating_base),
             self.robot_model.jnt_range[[i for i, _ in actuated_joints], 1]]
        )
        # Tangent-space joint bounds (nv_a,): used by the pinocchio joint limit constraint.
        # Base tangent DOFs (indices 0:6) are effectively unconstrained (trust region limits step).
        # Joints (qpos index k >= 7) map to tangent index k-1 for hinge/slide joints.
        # MANUAL overrides for quaternion components (qpos 3-6) are dropped here because
        # quaternion components have no direct tangent-space equivalent.
        _nv = self.pin.model.nv  # 35
        _jnt_lb_v = np.full(_nv, -large_number)  # base: free
        _jnt_ub_v = np.full(_nv, large_number)
        # Joint DOFs: qpos index k (k in 7..35) -> tangent index k-1 (6..34)
        for qpos_k in range(7, 7 + task_constants.ROBOT_DOF):
            v_k = qpos_k - 1
            _jnt_lb_v[v_k] = complete_lower[qpos_k]
            _jnt_ub_v[v_k] = complete_upper[qpos_k]
        # Apply MANUAL overrides for joints only (skip qpos 3-6 quaternion overrides).
        if task_constants.MANUAL_LB:
            for k_str, val in task_constants.MANUAL_LB.items():
                k = int(k_str)
                if k >= 7:
                    _jnt_lb_v[k - 1] = val
        if task_constants.MANUAL_UB:
            for k_str, val in task_constants.MANUAL_UB.items():
                k = int(k_str)
                if k >= 7:
                    _jnt_ub_v[k - 1] = val
        # Slice to the active tangent indices.
        self._v_a_lb = _jnt_lb_v[self.v_a_indices]
        self._v_a_ub = _jnt_ub_v[self.v_a_indices]

        # Correspondence table (loaded by from_config from the bundled artifact).
        # None until from_config populates it; not used in the solve yet.
        self.correspondence = None

        # Object SDF: built/loaded by from_config, keyed by the configured (L, resolution).
        # None until from_config populates it (and for object-less / floor-only tasks).
        self.object_sdf = None
        # Object surface control points (object-local), sampled from the object
        # mesh by from_config for the object<->floor inertia term. None until set.
        self.object_surface_local = None

        # Online SMPL-X -> SDF probe + its per-frame outputs (set by from_config).
        self.smplx_ground_probe = None
        self.smplx_sdf_fields: list = []
        # When True, retarget() fills CoM / angular-momentum / foot-slip diagnostics
        # post-hoc for the viewer. Off by default to keep the solve fast.
        self.collect_diagnostics = False

        # ===== Holosoma-style optional constraints (default OFF; copied verbatim
        # from src/holosoma/interaction_mesh_retargeter.py). When every flag is
        # off the solve is unchanged. =====
        self.activate_obj_non_penetration = activate_obj_non_penetration
        self.activate_self_collision = activate_self_collision
        self.activate_foot_sticking = activate_foot_sticking
        self.penetration_tolerance = penetration_tolerance
        self.foot_sticking_tolerance = foot_sticking_tolerance
        # self.object_name already set above (before model load)
        self.foot_links = dict(zip(task_constants.FOOT_STICKING_LINKS,
                                   task_constants.FOOT_STICKING_LINKS))
        self.collision_detection_threshold = 0.1
        self._geom_names = [self.robot_model.geom(g).name or "" for g in range(self.robot_model.ngeom)]
        self._init_foot_lock(foot_lock if foot_lock is not None else FootLockConfig())
        self._init_self_collision(self_collision if self_collision is not None else SelfCollisionConfig())
        # foot_sticking_sequences is filled by from_config; () = no sticking.
        self.foot_sticking_sequences: list = []

        # Interaction D/X/P cost weights (default 0.0 = off; solve is unchanged).
        self.lambda_d = lambda_d
        self.lambda_x = lambda_x
        self.lambda_p = lambda_p
        self.sigma_v = sigma_v

        # Native-Holosoma objective ports (default 0.0 = off; solve is unchanged).
        self.lambda_smooth = lambda_smooth
        self.lambda_qdiag = lambda_qdiag
        # Per-actuated-joint Q_diag weights from MANUAL_COST (0 elsewhere), mirroring
        # Holosoma's self.Q_diag. Indexed by actuated-joint position (0..ROBOT_DOF-1).
        self._q_diag_joints = np.zeros(task_constants.ROBOT_DOF)
        for _k, _v in getattr(task_constants, "MANUAL_COST", {}).items():
            self._q_diag_joints[int(_k)] = float(_v)
        # W^nominal: which actuated joints, and the nominal pose to pull them toward.
        # Default nominal = q_init's joint pose (static). For exact Holosoma parity the
        # per-frame nominal (data qpos) is threaded in at the parity step.
        self.lambda_nominal = lambda_nominal
        self.nominal_tau = nominal_tau
        self._nominal_idx = np.asarray(
            getattr(task_constants, "NOMINAL_TRACKING_INDICES", []), dtype=int)

        # Temporal regularization weights (default 0.0 = off; solve is unchanged).
        self.lambda_r = lambda_r
        self.sigma_qddot = sigma_qddot
        self.sigma_Vdot = sigma_Vdot

        # GMR base tracking: per-point position / orientation cost (weights from the
        # IK match tables). Toggle each channel on/off; the table values still apply.
        self.activate_pos_tracking = activate_pos_tracking
        self.activate_rot_tracking = activate_rot_tracking
        # GMR tracking: global priority λ + characteristic scale σ per channel.
        self.lambda_pos = lambda_pos
        self.sigma_p = sigma_p
        self.lambda_rot = lambda_rot
        self.sigma_rot = sigma_rot
        # W^s Style (additive pelvis-relative orientation matching; default 0 = off).
        self.lambda_ws = lambda_ws
        self.sigma_R = sigma_R
        # Brick 2 — per-body style weights ω_k^s. Default = uniform table (same
        # behavior as the legacy w_r path, since all w_r=10). Pass an explicit
        # dict to override; set to None to fall back to the legacy w_r path.
        self.style_weights = style_weights if style_weights is not None else STYLE_WEIGHT_TABLE
        # Brick 1 — σ characteristic scales (flat constants; plumbed from config).
        self.sigma_a = sigma_a
        self.sigma_L = sigma_L
        self.sigma_ao = sigma_ao
        self.sigma_omega = sigma_omega

        # Brick 4 — Centroidal W^c / W^c_pos / W^L (default off; parity preserved).
        self.activate_centroidal = activate_centroidal
        self.lambda_c = lambda_c
        self.lambda_c_pos = lambda_c_pos
        self.lambda_l = lambda_l
        self.lambda_cv = lambda_cv      # W^c_vel: CoM velocity tracking
        self.sigma_cv = sigma_cv

        # Brick 5 — Movable entities W^o (default off; parity preserved).
        # §1 Variables: q_a (joints) and T_B (base) are free by default; setting their
        # flag False freezes that block of the tangent step (dqa). T_m (object) becomes a
        # variable only when activate_tm (handled where dxi_obj is created).
        self.activate_qa = activate_qa
        self.activate_tb = activate_tb
        self.activate_tm = activate_tm
        self.lambda_o = lambda_o
        self.lambda_o_pos = lambda_o_pos
        # Object-as-carrier interaction (object ↔ environment): D/X/P, mirroring the robot.
        self.lambda_d_obj = lambda_d_obj
        self.lambda_x_obj = lambda_x_obj
        self.lambda_p_obj = lambda_p_obj
        self.sigma_v_obj = sigma_v_obj
        self.activate_obj_surface_nonpen = activate_obj_surface_nonpen
        self.obj_surface_nonpen_tol = obj_surface_nonpen_tol
        self.activate_wl_track = activate_wl_track
        self.lambda_l_track = lambda_l_track
        # Solved object pose history: updated by retarget() when movable is on.
        self._obj_solved_poses: list = []

        # Brick 1 — Contact persistence hard band constraint (default off).
        self.activate_persistence = activate_persistence
        self.persistence_tol = persistence_tol

        # Brick 3 — Interaction length. L_interaction is the master range; L_floor/L_object
        # are optional per-channel overrides (None = inherit the master, then the probe
        # margin, set by from_config after __init__). sdf_resolution sets the SDF voxel.
        self._L_interaction_cfg = L_interaction
        self._L_floor_cfg = L_floor
        self._L_object_cfg = L_object
        self.sdf_resolution = sdf_resolution

        # Build robot_link_names: map each IK table frame -> actual G1 body name,
        # applying the remap for the two missing toe bodies.
        available_bodies = {self.robot_model.body(i).name for i in range(self.robot_model.nbody)}
        self.robot_link_names: dict[str, str] = {}
        for frame in IK_MATCH_TABLE1:
            actual = _BODY_NAME_REMAP.get(frame, frame)
            bid = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, actual)
            if bid == -1:
                raise ValueError(
                    f"[TestSocp] Body '{actual}' (mapped from table key '{frame}') "
                    f"not found in model. Available: {sorted(available_bodies)}"
                )
            if actual != frame:
                print(f"[TestSocp] Remapped body: '{frame}' -> '{actual}'")
            self.robot_link_names[frame] = actual

    @property
    def L_floor(self) -> float:
        """Floor field range (activation distance + positional scale): the L_floor override
        if set, else the master L_interaction, else the probe margin (legacy AUTO)."""
        if self._L_floor_cfg is not None:
            return self._L_floor_cfg
        if self._L_interaction_cfg is not None:
            return self._L_interaction_cfg
        return self.smplx_ground_probe.margin

    @property
    def L_object(self) -> float:
        """Object field range (activation distance + positional scale): the L_object
        override if set, else the master L_interaction, else the probe margin (legacy)."""
        if self._L_object_cfg is not None:
            return self._L_object_cfg
        if self._L_interaction_cfg is not None:
            return self._L_interaction_cfg
        return self.smplx_ground_probe.margin

    def _body_jac(self, q: np.ndarray, body_name: str):
        """World-frame (Jp, Jr) for a body, reduced to active tangent columns.

        Uses pinocchio LOCAL_WORLD_ALIGNED Jacobians (translational rows 0:3,
        angular rows 3:6) sliced to v_a_indices.

        Args:
            q: Full configuration vector (length nq; robot part is q[:36]).
            body_name: Link name (pinocchio frame name = MuJoCo body name).

        Returns:
            Tuple (Jp, Jr) each of shape (3, nv_a).
        """
        q_pin = self.pin.qpos_mj_to_q_pin(q[:36])
        Jp = self.pin.frame_translational_jacobian(q_pin, body_name)  # (3, nv)
        Jr = self.pin.frame_angular_jacobian(q_pin, body_name)        # (3, nv)
        return Jp[:, self.v_a_indices], Jr[:, self.v_a_indices]

    def body_position(self, q: np.ndarray, body_name: str) -> np.ndarray:
        """World position of ``body_name`` at configuration ``q``.

        Delegates to pinocchio FK on the robot qpos slice q[:36].

        Args:
            q: Full configuration vector (length nq).
            body_name: Link name (pinocchio frame name).

        Returns:
            Position array of shape (3,).
        """
        return self.pin.body_position(self.pin.qpos_mj_to_q_pin(q[:36]), body_name)

    def body_rotation(self, q: np.ndarray, body_name: str) -> np.ndarray:
        """World rotation matrix of ``body_name`` at configuration ``q``.

        Delegates to pinocchio FK on the robot qpos slice q[:36].

        Args:
            q: Full configuration vector (length nq).
            body_name: Link name (pinocchio frame name).

        Returns:
            Rotation matrix of shape (3, 3).
        """
        return self.pin.body_rotation(self.pin.qpos_mj_to_q_pin(q[:36]), body_name)

    # ------------------------------------------------------------------
    # Core solve
    # ------------------------------------------------------------------

    def solve_single_iteration(
        self,
        q_locked: np.ndarray,
        q_a_n_last: np.ndarray,
        q_t_last: np.ndarray,
        frame_targets: dict,
        init_t: bool = False,
        frame_idx: int = 0,
        foot_sticking: tuple[bool, bool] | None = None,
        obj_pose=None,
        obj_pose_ref=None,
        obj_pose_ref_tm1=None,
        q_t_last2: np.ndarray | None = None,
        c_tm1: np.ndarray | None = None,
        c_tm2: np.ndarray | None = None,
        cddot_ref: np.ndarray | None = None,
        cdot_ref: np.ndarray | None = None,
        c_ref: np.ndarray | None = None,
        obj_pose_tm1=None,
        obj_pose_tm2=None,
        vdot_ref_obj: np.ndarray | None = None,
        omega_ref_obj: np.ndarray | None = None,
        sqp_iter: int = 0,
    ):
        """One linearised IK step with GMR position + orientation objective.

        Builds a SOCP of the form:

            min  sum_f  w_p * ||Jp_f dqa - (p_t - p_c)||^2
                      + w_r * ||Jr_body_f dqa - e_body||^2
            s.t. ||dqa|| <= step_size
                 q_lb <= q_a + dqa <= q_ub  (if activate_joint_limits)

        where Jr_body_f = R_c.T @ Jr_f is the body-frame angular Jacobian and
        e_body = log(R_c.T @ R_t) is the body-frame rotation error.

        Args:
            q_locked: Full configuration with the floating-base part locked.
            q_a_n_last: Actuated-DoF qpos seed at the current iterate (length
                nq_a); used only to reconstruct the full config via
                ``q[self.q_a_indices] = q_a_n_last``.  The SOCP decision
                variable ``dqa`` lives in pinocchio tangent space (length
                nv_a), which differs from nq_a for quaternion joints.
            q_t_last: Full configuration from the previous time-step (for API
                compatibility; unused in v2 which has no smoothness cost).
            frame_targets: {frame: (p_target(3,), R_target(3,3), w_p, w_r)}
                as returned by build_frame_targets.
            init_t: True on the very first frame (unused in v2, kept for compat).
            frame_idx: Index of the current frame; used for window filtering by
                the holosoma-style constraints (self-collision / foot) when enabled.
            foot_sticking: Per-foot sticking flags (left, right) for this frame;
                used by the foot-sticking constraint when enabled.

        Returns:
            (q_star, cost): updated full config and objective value.
        """
        q = np.copy(q_locked)
        q[self.q_a_indices] = q_a_n_last

        # q[:36] is fixed within a solve_single_iteration call, so convert to
        # the pinocchio configuration ONCE and reuse it across all the term
        # builders (D/X, persistence, centroidal, L_ref, ...) instead of ~9x.
        _q_pin = self.pin.qpos_mj_to_q_pin(q[:36])

        dqa = cp.Variable(self.nv_a, name="dqa")

        obj_terms = []
        # World-frame tracking (GMR): always runs; gated per body by activate_pos/rot.
        # Global priority λ + characteristic scale σ folded per channel (defaults 1.0
        # => effective weight == legacy w_p/w_r). See tracking.build_tracking_terms.
        from HoloNew.src.test_socp.tracking import build_tracking_terms
        obj_terms.extend(build_tracking_terms(
            self, frame_targets, dqa, q,
            lambda_pos=self.lambda_pos, sigma_p=self.sigma_p,
            lambda_rot=self.lambda_rot, sigma_rot=self.sigma_rot,
            activate_pos=self.activate_pos_tracking,
            activate_rot=self.activate_rot_tracking))

        # W^s Style: pelvis-relative joint-orientation matching (S_k) + pelvis
        # tilt against gravity (S_B), each residual divided by σ_R. See style.py.
        from HoloNew.src.test_socp.style import build_style_terms
        obj_terms.extend(build_style_terms(
            self, q, frame_targets, dqa,
            lambda_ws=self.lambda_ws, sigma_R=self.sigma_R,
            style_weights=getattr(self, "style_weights", None)))

        constraints = [cp.SOC(self.step_size, dqa)]
        # §1 Variables: freeze q_a (joints) or T_B (base) by pinning their tangent block
        # to zero. dqa is indexed by self.v_a_indices; tangent index < 6 is the floating
        # base, >= 6 are the actuated joints.
        if not self.activate_tb:
            _base = np.where(self.v_a_indices < 6)[0]
            if _base.size:
                constraints.append(dqa[_base] == 0)
        if not self.activate_qa:
            _joints = np.where(self.v_a_indices >= 6)[0]
            if _joints.size:
                constraints.append(dqa[_joints] == 0)
        if self.activate_joint_limits:
            # Tangent-space joint limit box: precomputed absolute bounds minus
            # current joint values.  For hinge/slide joints, the pinocchio tangent
            # increment equals (target - current) directly (same as subtraction).
            # Base DOFs (tangent 0:6) use large bounds and are effectively unconstrained.
            # Tangent index vi >= 6 maps to pinocchio q index vi + 1 (offset by 1
            # because the root FREE joint occupies 7 qpos DOFs but only 6 velocity DOFs).
            q_pin_cur = _q_pin
            lo = np.copy(self._v_a_lb)
            hi = np.copy(self._v_a_ub)
            joint_mask = self.v_a_indices >= 6
            vi_joints = self.v_a_indices[joint_mask]       # tangent indices for joints
            q_pin_vals = q_pin_cur[vi_joints + 1]          # q_pin at corresponding qpos idx
            lo[joint_mask] -= q_pin_vals
            hi[joint_mask] -= q_pin_vals
            constraints += [dqa >= lo, dqa <= hi]

        # Foot constraints (sticking + foot lock window Z pinning) — holosoma-style, default off
        apply_foot_sticking = (self.q_a_init_idx < 12) and self.activate_foot_sticking and foot_sticking is not None
        apply_foot_lock = (self.q_a_init_idx < 12) and self.foot_lock.enable
        if apply_foot_sticking or apply_foot_lock:
            J_WF_dict, p_WF_dict, _ = self._calc_manipulator_jacobians(q, links=self.foot_links, obj_frame=False)

            # Foot sticking: constrain XY to stay near previous frame position
            if apply_foot_sticking:
                _, p_WF_t_last_dict, _ = self._calc_manipulator_jacobians(
                    q_t_last, links=self.foot_links, obj_frame=False
                )
                left_key = right_key = None
                for key in foot_sticking:
                    if key.lower().startswith("l"):
                        left_key = key
                    elif key.lower().startswith("r"):
                        right_key = key
                if left_key is None or right_key is None:
                    raise ValueError("foot_sticking must include one left* and one right* key")

                for key, J_WF in J_WF_dict.items():
                    apply_left = ("left" in key) and foot_sticking[left_key]
                    apply_right = ("right" in key) and foot_sticking[right_key]
                    if apply_left or apply_right:
                        p_lb = p_WF_t_last_dict[key] - p_WF_dict[key] - self.foot_sticking_tolerance
                        p_ub = p_lb + 2 * self.foot_sticking_tolerance  # symmetric window

                        # J_WF is already (3, nv_a) from _calc_manipulator_jacobians.
                        Jxy = J_WF[:2, :]  # (2, nv_a)
                        constraints += [
                            Jxy @ dqa >= p_lb[:2],
                            Jxy @ dqa <= p_ub[:2],
                        ]

            # Foot lock windows: pin Z to floor within configured frame ranges
            if apply_foot_lock:
                for key, J_WF in J_WF_dict.items():
                    if not self._is_foot_locked_in_window(key, frame_idx):
                        continue

                    z_anchor = self.foot_lock.z_floor
                    z_delta = z_anchor - p_WF_dict[key][2]
                    # J_WF is already (3, nv_a) from _calc_manipulator_jacobians.
                    Jz = J_WF[2, :]  # (nv_a,)
                    constraints += [
                        Jz @ dqa >= z_delta - self.foot_lock.tolerance,
                        Jz @ dqa <= z_delta + self.foot_lock.tolerance,
                    ]

        # Self-collision constraints (holosoma-style, default off)
        if self.activate_self_collision and self._self_collision_enabled:
            self.robot_data.qpos[:len(q)] = q
            mujoco.mj_forward(self.robot_model, self.robot_data)
            Js_sc, phis_sc = self._compute_self_collision_constraints(frame_idx)
            for key, phi in phis_sc.items():
                Ja_n_full = Js_sc[key]  # (nv,) relative Jacobian
                Ja_n = Ja_n_full[self.v_a_indices]  # (nv_a,)
                # Enforce: new_distance >= tolerance  =>  phi + J @ dqa >= tol
                rhs = self._self_collision_tolerance - phi
                constraints += [Ja_n @ dqa >= rhs]

        # Non-penetration constraints (holosoma-style, default off)
        if self.activate_obj_non_penetration:
            Js, phis = self._update_jacobians_and_phis_from_q(q)
            for key, phi in phis.items():
                Ja_n_full = Js[key]  # (nv,) relative Jacobian
                Ja_n = Ja_n_full[self.v_a_indices]  # (nv_a,)
                # Enforce: phi + J @ dqa >= -tol  (keep signed distance above -tolerance).
                rhs = -phi - self.penetration_tolerance
                constraints += [Ja_n @ dqa >= rhs]

        # Brick 5 — Object tangent variable: created whenever movable is on and an
        # object pose is present, so both the W^o term and the bilateral D/X term
        # share the SAME dxi_obj variable.  The trust-region SOC is added here so
        # the step size is bounded regardless of which downstream terms use dxi_obj.
        dxi_obj = None
        if self.activate_tm and obj_pose is not None:
            dxi_obj = cp.Variable(6, name="dxi_obj")
            constraints.append(cp.SOC(self.step_size, dxi_obj))

        # D + X interaction terms (default off; only active when weights > 0). The floor
        # is always a target, so the only asset needed is the ground field + correspondence;
        # the object channel inside build_dx_terms is self-gated on obj_pose.
        if (self.lambda_d > 0 or self.lambda_x > 0) \
                and getattr(self, "correspondence", None) is not None \
                and getattr(self, "smplx_ground_probe", None) is not None:
            from HoloNew.src.test_socp.interaction import build_dx_terms
            q_pin = _q_pin
            # Pass dxi_obj for bilateral coupling when the object is a variable.
            # When dxi_obj is None (movable off), behaviour is unchanged.
            obj_terms += build_dx_terms(self, q_pin, dqa, frame_idx, obj_pose,
                                        self.lambda_d, self.lambda_x,
                                        dxi=dxi_obj)

        # P (contact persistence) terms (default off; requires cross-frame state
        # from the previous solved frame, so only active at frame >= 1).
        if self.lambda_p > 0 \
                and getattr(self, "correspondence", None) is not None \
                and getattr(self, "smplx_ground_probe", None) is not None \
                and frame_idx >= 1 \
                and getattr(self, "_p_state", None) is not None:
            from HoloNew.src.test_socp.interaction import build_p_terms
            q_pin = _q_pin
            obj_terms += build_p_terms(self, q_pin, dqa, frame_idx, obj_pose,
                                       self.lambda_p, self.sigma_v, self._dt)

        # Contact persistence hard tangential band constraint (default off).
        # Enforces no-slip per carrier instead of the soft P cost. Requires the
        # same cross-frame _p_state as the soft P term; only fires at frame >= 1.
        if self.activate_persistence \
                and getattr(self, "correspondence", None) is not None \
                and getattr(self, "smplx_ground_probe", None) is not None \
                and frame_idx >= 1 \
                and getattr(self, "_p_state", None) is not None:
            from HoloNew.src.test_socp.interaction import build_p_constraints
            q_pin = _q_pin
            constraints += build_p_constraints(self, q_pin, dqa, frame_idx, obj_pose,
                                               self.persistence_tol)

        # Object-surface non-penetration (paper's d_{i,j} >= 0 for the object).
        # Hard inequality so robot points cannot pass through the object surface;
        # the D cost only discourages it softly. Only fires when an object SDF and
        # pose are present.
        if self.activate_obj_surface_nonpen \
                and getattr(self, "object_sdf", None) is not None \
                and obj_pose is not None \
                and getattr(self, "correspondence", None) is not None:
            from HoloNew.src.test_socp.interaction import build_obj_surface_nonpen_constraints
            q_pin = _q_pin
            constraints += build_obj_surface_nonpen_constraints(
                self, q_pin, dqa, frame_idx, obj_pose, dxi=dxi_obj,
                tol=self.obj_surface_nonpen_tol)

        # Temporal regularization W^r (default off; only active when lambda_r > 0
        # and two previous frames are available).
        if self.lambda_r > 0 and q_t_last is not None and q_t_last2 is not None:
            from HoloNew.src.test_socp.temporal import build_temporal_term
            q_pin0 = _q_pin
            q_pin1 = self.pin.qpos_mj_to_q_pin(q_t_last[:36])
            q_pin2 = self.pin.qpos_mj_to_q_pin(q_t_last2[:36])
            obj_terms += [build_temporal_term(self, q_pin0, q_pin1, q_pin2, dqa,
                                              self.lambda_r, self.sigma_qddot,
                                              self.sigma_Vdot, self._dt)]

        # Centroidal W^c / W^c_pos / W^L terms (Brick 4, default off).
        # W^c and W^L require two prior solved CoMs (frame_idx >= 2).
        # W^c_pos only needs c_ref and fires at any frame when active.
        # Guard: activate_centroidal and at least one lambda > 0.
        if self.activate_centroidal \
                and (self.lambda_c > 0 or self.lambda_c_pos > 0
                     or self.lambda_l > 0 or self.lambda_cv > 0) \
                and c_ref is not None:
            # W^c / W^L require the two-step CoM history; only activate from frame 2.
            _lam_c_eff = self.lambda_c if (
                frame_idx >= 2
                and c_tm1 is not None and c_tm2 is not None
                and cddot_ref is not None and q_t_last is not None
            ) else 0.0
            _lam_L_eff = self.lambda_l if (
                frame_idx >= 2
                and c_tm1 is not None and c_tm2 is not None
                and q_t_last is not None
            ) else 0.0
            # W^c_vel: CoM velocity tracking needs only the robot velocity (q_t_last) and
            # the reference CoM velocity (first diff) -> available from frame 1.
            _lam_cv_eff = self.lambda_cv if (
                frame_idx >= 1 and q_t_last is not None and cdot_ref is not None
            ) else 0.0
            if self.lambda_c_pos > 0 or _lam_c_eff > 0 or _lam_L_eff > 0 or _lam_cv_eff > 0:
                from HoloNew.src.test_socp.centroidal import build_centroidal_terms
                q_t0 = _q_pin
                q_tm1 = (self.pin.qpos_mj_to_q_pin(q_t_last[:36])
                         if q_t_last is not None else q_t0)
                _cddot_ref_eff = cddot_ref if cddot_ref is not None else np.zeros(3)
                _c_tm1_eff = c_tm1 if c_tm1 is not None else self.pin.com(q_t0)
                _c_tm2_eff = c_tm2 if c_tm2 is not None else self.pin.com(q_t0)
                obj_terms += build_centroidal_terms(
                    self, q_t0, q_tm1, _c_tm1_eff, _c_tm2_eff,
                    _cddot_ref_eff, c_ref, dqa,
                    _lam_c_eff, self.lambda_c_pos, _lam_L_eff,
                    self._dt,
                    sigma_a=self.sigma_a, sigma_L=self.sigma_L,
                    lambda_cv=_lam_cv_eff, sigma_cv=self.sigma_cv, cdot_ref=cdot_ref,
                )

        # W^L reference tracking (opt-in): track the lumped reference angular
        # momentum instead of driving L to 0. Needs the previous solved config for
        # the velocity finite difference, so fires from frame_idx >= 1.
        if (self.activate_wl_track
                and getattr(self, "_L_ref_all", None) is not None
                and frame_idx >= 1 and q_t_last is not None):
            from HoloNew.src.test_socp.centroidal import build_lumped_L_term
            q_pin_cur = _q_pin
            q_pin_prev = self.pin.qpos_mj_to_q_pin(q_t_last[:36])
            obj_terms.append(build_lumped_L_term(
                self, q_pin_cur, q_pin_prev, dqa, self._lumped_frames,
                self._lumped_masses, self._L_ref_all[frame_idx],
                self.lambda_l_track, self._dt,
                sigma_L=self.sigma_L))

        # W^o object motion regularization (Brick 5, default off).
        # dxi_obj was already created above; the W^o cost is added when the two
        # prior object poses and at least one lambda are available (frame_idx >= 2).
        if (dxi_obj is not None
                and obj_pose_tm1 is not None
                and obj_pose_tm2 is not None
                and self.lambda_o > 0
                and frame_idx >= 2):
            from HoloNew.src.test_socp.movable import build_wo_term, pose_to_se3
            T_obj0 = pose_to_se3(obj_pose)
            T_obj_tm1 = pose_to_se3(obj_pose_tm1)
            T_obj_tm2 = pose_to_se3(obj_pose_tm2)
            _vdot_ref = vdot_ref_obj if vdot_ref_obj is not None else np.zeros(3)
            _omega_ref = omega_ref_obj if omega_ref_obj is not None else np.zeros(3)
            obj_terms.append(build_wo_term(
                T_obj0, T_obj_tm1, T_obj_tm2,
                _vdot_ref, _omega_ref,
                dxi_obj, self.lambda_o, self._dt,
                sigma_ao=self.sigma_ao, sigma_omega=self.sigma_omega,
            ))

        # W^o position anchor: pins the absolute object position to the reference
        # path. W^o regularizes only object acceleration/velocity (position-blind),
        # so the bilateral D/X coupling can offset the object in absolute position
        # while still matching the reference acceleration. This anchor cures that
        # drift. Needs no object history, so it fires whenever the object is a
        # variable (all frames), unlike the frame_idx>=2 W^o term above.
        if (dxi_obj is not None
                and obj_pose is not None
                and self.lambda_o_pos > 0):
            from HoloNew.src.test_socp.movable import (
                build_wo_position_anchor, pose_to_se3)
            # Anchor target is the REFERENCE position (obj_pose_ref), NOT the warm-start
            # obj_pose — otherwise the anchor would chase the drift instead of pinning it.
            _anchor_ref = obj_pose_ref if obj_pose_ref is not None else obj_pose
            obj_terms.append(build_wo_position_anchor(
                pose_to_se3(obj_pose), np.asarray(_anchor_ref[4:7], dtype=float),
                dxi_obj, self.lambda_o_pos,
            ))

        # Object<->floor contact (paper's object-environment pair; inertia mode).
        # Places the object by its floor contact instead of a positional anchor:
        # near-floor object surface points resist breaking contact, vanishing when
        # the object is lifted (then placed by object<->robot + ballistic W^o).
        if (dxi_obj is not None
                and obj_pose is not None
                and getattr(self, "object_surface_local", None) is not None
                and (self.lambda_d_obj > 0 or self.lambda_x_obj > 0 or self.lambda_p_obj > 0)):
            from HoloNew.src.test_socp.movable import (
                build_object_floor_terms, build_object_floor_persistence)
            _L_floor = (self.L_floor
                        if self.smplx_ground_probe is not None else 0.1)
            # D/X: object surface points vs the floor field.
            if self.lambda_d_obj > 0 or self.lambda_x_obj > 0:
                obj_terms += build_object_floor_terms(
                    self, dxi_obj, obj_pose, self.lambda_d_obj, self.lambda_x_obj, _L_floor,
                    obj_pose_ref=obj_pose_ref)
            # P: tangential no-slip of the object on the floor, tracking the reference
            # object's slide. Needs the previous SOLVED pose + reference[t] & [t-1] (frame>=1).
            if (self.lambda_p_obj > 0 and obj_pose_tm1 is not None
                    and obj_pose_ref is not None and obj_pose_ref_tm1 is not None):
                obj_terms += build_object_floor_persistence(
                    self, dxi_obj, obj_pose, obj_pose_tm1, obj_pose_ref, obj_pose_ref_tm1,
                    self.lambda_p_obj, self.sigma_v_obj, _L_floor, self._dt)

        # W^smooth (native Holosoma): penalize the actuated-joint step deviating from
        # the step toward the previous frame's pose. Matches Holosoma's
        #   smooth_weight * ||dqa - (q_t_last_a - q_a_n_last)||^2
        # restricted to the actuated-joint columns of dqa (v_a tangent index >= 6).
        if self.lambda_smooth > 0 and q_t_last is not None:
            joint_cols = np.where(self.v_a_indices >= 6)[0]      # joint tangent columns
            joint_qpos = self.v_a_indices[joint_cols] + 1        # tangent 6+j -> qpos 7+j
            dqa_smooth = q_t_last[joint_qpos] - q_a_n_last[joint_qpos]   # (n_joint,)
            obj_terms.append(
                self.lambda_smooth * cp.sum_squares(dqa[joint_cols] - dqa_smooth))

        # W^qdiag (native Holosoma): per-joint regularizer on the absolute new joint
        # config. Matches Holosoma's ||sqrt(Q_diag) (dqa + q_a_n_last)||^2 on the
        # actuated joints (Q_diag from MANUAL_COST, 0 elsewhere).
        if self.lambda_qdiag > 0:
            joint_cols = np.where(self.v_a_indices >= 6)[0]
            joint_qpos = self.v_a_indices[joint_cols] + 1
            sw = np.sqrt(self.lambda_qdiag * self._q_diag_joints[:joint_cols.size])
            new_joints = dqa[joint_cols] + q_a_n_last[joint_qpos]
            obj_terms.append(cp.sum_squares(cp.multiply(sw, new_joints)))

        # W^nominal (native Holosoma): pull selected joints toward a nominal pose with
        # an exp-decaying weight over SQP iterations. Matches Holosoma's
        #   w_init*exp(-i/tau) * ||dqa[idx] - (q_nominal[idx] - q_a_n_last[idx])||^2.
        # Nominal defaults to q_init's joints (the per-frame nominal is wired at parity).
        if self.lambda_nominal > 0 and self._nominal_idx.size > 0:
            joint_cols = np.where(self.v_a_indices >= 6)[0]
            joint_qpos = self.v_a_indices[joint_cols] + 1
            cols_sel = joint_cols[self._nominal_idx]
            qpos_sel = joint_qpos[self._nominal_idx]
            w_nom = self.lambda_nominal * float(np.exp(-sqp_iter / self.nominal_tau))
            z = dqa[cols_sel] - (self.q_init_full[qpos_sel] - q_a_n_last[qpos_sel])
            obj_terms.append(w_nom * cp.sum_squares(z))

        prob = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)
        try:
            prob.solve(solver=cp.CLARABEL)
            _ok = prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        except cp.error.SolverError:
            _ok = False
        if not _ok:
            # CLARABEL occasionally fails on ill-conditioned iterations (e.g. when
            # the interaction terms engage large relative motions). Fall back to
            # SCS, a first-order solver that is more robust to conditioning.
            prob.solve(solver=cp.SCS)
            if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                raise RuntimeError(f"TEST-SOCP solve failed: {prob.status}")

        v_full = np.zeros(self.pin.model.nv)
        v_full[self.v_a_indices] = dqa.value
        q_pin_new = self.pin.integrate(_q_pin, v_full)
        q_star = np.copy(q)
        q_star[:36] = self.pin.q_pin_to_qpos_mj(q_pin_new)
        # pin.integrate keeps the quaternion unit; no manual renormalisation needed.

        # Integrate the solved object tangent step if movable was active this frame.
        # T_obj_new = exp6(dxi) * T_obj0  (left-compose, world-frame, matches build_wo_term).
        solved_obj_pose = None
        if dxi_obj is not None and dxi_obj.value is not None:
            from HoloNew.src.test_socp.movable import pose_to_se3, se3_to_pose
            T_obj0 = pose_to_se3(obj_pose)
            T_obj_new = pin.exp6(dxi_obj.value) * T_obj0
            solved_obj_pose = se3_to_pose(T_obj_new)

        return q_star, float(prob.value), solved_obj_pose

    def iterate(
        self,
        q_locked: np.ndarray,
        q_n: np.ndarray,
        q_t_last: np.ndarray,
        frame_targets: dict,
        n_iter: int = 10,
        frame_idx: int = 0,
        foot_sticking: tuple[bool, bool] | None = None,
        obj_pose=None,
        obj_pose_ref=None,
        obj_pose_ref_tm1=None,
        q_t_last2: np.ndarray | None = None,
        c_tm1: np.ndarray | None = None,
        c_tm2: np.ndarray | None = None,
        cddot_ref: np.ndarray | None = None,
        cdot_ref: np.ndarray | None = None,
        c_ref: np.ndarray | None = None,
        obj_pose_tm1=None,
        obj_pose_tm2=None,
        vdot_ref_obj: np.ndarray | None = None,
        omega_ref_obj: np.ndarray | None = None,
    ):
        """Iterate solve_single_iteration until convergence or n_iter steps.

        Two stop conditions: the objective stops improving (np.isclose), or the
        actuated configuration stops moving (step norm < _iterate_step_tol). The
        second catches passes that plateau at a tiny oscillating step at the
        trust-region / active-set boundary — they never trip the cost test but
        their extra iterations only chatter and waste solves.

        The object is a first-class SQP variable: ``obj_pose`` is its linearization
        point, RE-LINEARIZED each inner iteration around the freshly solved pose so
        it converges within the frame just like the robot. ``obj_pose_ref`` stays
        the reference pose throughout (the W^o_pos anchor target) and is never
        re-linearized — only obj_pose advances.
        """
        if obj_pose_ref is None:
            obj_pose_ref = obj_pose
        last = np.inf
        cost = 0.0
        solved_obj_pose = None
        for _sqp_it in range(n_iter):
            q_a_prev = q_n[self.q_a_indices].copy()
            q_n, cost, solved_obj_pose = self.solve_single_iteration(
                q_locked, q_n[self.q_a_indices], q_t_last, frame_targets,
                frame_idx=frame_idx, foot_sticking=foot_sticking,
                obj_pose=obj_pose, obj_pose_ref=obj_pose_ref,
                obj_pose_ref_tm1=obj_pose_ref_tm1, q_t_last2=q_t_last2,
                c_tm1=c_tm1, c_tm2=c_tm2, cddot_ref=cddot_ref, cdot_ref=cdot_ref,
                c_ref=c_ref,
                obj_pose_tm1=obj_pose_tm1, obj_pose_tm2=obj_pose_tm2,
                vdot_ref_obj=vdot_ref_obj, omega_ref_obj=omega_ref_obj,
                sqp_iter=_sqp_it,
            )
            # Re-linearize the object around its solved pose for the next inner step
            # (the robot is already re-linearized via q_n). obj_pose_ref is untouched.
            if solved_obj_pose is not None:
                obj_pose = solved_obj_pose
            step = float(np.linalg.norm(q_n[self.q_a_indices] - q_a_prev))
            if np.isclose(cost, last) or step < self._iterate_step_tol:
                break
            last = cost
        return q_n, cost, solved_obj_pose

    def retarget(self, max_frames: int | None = None):
        """Run the full two-pass GMR solve over all frames.

        Requires from_config to have been called first (sets self.gmr_ground,
        self.q_init_full).

        Args:
            max_frames: If given, solve only the first ``max_frames`` frames (for
                fast tests / partial validation runs). None solves the whole clip.

        Returns:
            RetargetResult with qpos (T, 7+DOF) trajectory.
        """
        from tqdm import tqdm

        from HoloNew.src.retarget_result import RetargetResult
        from .tables import IK_MATCH_TABLE_SINGLE
        from .targets import ground_frame_targets

        gpos = self.gmr_ground["pos"]
        gquat = self.gmr_ground["quat"]
        T = gpos.shape[0]
        if max_frames is not None:
            T = min(T, int(max_frames))
        q = np.copy(self.q_init_full)
        out = []
        costs = []  # per-frame SQP objective value (solver-health diagnostic)
        # q_prev: previous-frame foot anchor for foot-sticking; both passes use the same anchor,
        # updated once per frame (after both passes). Frame 0 anchors to init config.
        q_prev = np.copy(self.q_init_full)
        # q_prev2: frame-before-previous config for temporal W^r acceleration penalty.
        # Initialized to init config so the first two frames produce ~zero acceleration.
        q_prev2 = np.copy(self.q_init_full)

        # Brick 4 — Centroidal: precompute reference CoM acceleration from the
        # reference pelvis trajectory (a dominant-mass CoM proxy).  Uses central
        # finite differences; frames 0-1 are set to zero (warm-up, guard inactive).
        # The reference pelvis positions live in gpos[:, pelvis_body_idx, :].
        # Ground targets index: 'pos' is (T, N_bodies, 3); index 0 is the pelvis.
        _g_pelvis = gpos[:, 0, :]  # (T, 3) reference pelvis positions
        _cddot_ref_all = np.zeros((T, 3), dtype=np.float64)
        for _t in range(2, T):
            _cddot_ref_all[_t] = (
                _g_pelvis[_t] - 2.0 * _g_pelvis[_t - 1] + _g_pelvis[_t - 2]
            ) / (self._dt ** 2)

        # Precompute reference CoM positions for W^c_pos.
        # The reference is the pelvis trajectory (CoM proxy), but the robot CoM sits
        # below and behind the pelvis (a constant structural offset).  To avoid
        # pulling the robot CoM to the pelvis position (which would push the pelvis
        # upward by ~7 cm), we compute the offset at init and apply it to all frames:
        #   c_ref[t] = g_pelvis[t] + (c_init - pelvis_init)
        # This anchors the CoM to its reference trajectory *in the robot's own frame
        # relative to the pelvis*, so W^c_pos corrects drift without biasing height.
        _q_init_pin = self.pin.qpos_mj_to_q_pin(self.q_init_full[:36])
        _com_init = self.pin.com(_q_init_pin)
        _pelvis_init = self.pin.body_position(_q_init_pin, "pelvis")
        _com_pelvis_offset = _com_init - _pelvis_init  # (3,) structural offset
        _c_ref_all = _g_pelvis + _com_pelvis_offset     # (T, 3) reference CoM positions
        # Keep the grounded CoM target around for the viewer diagnostics (the W^c_pos
        # anchor target), so the solved-vs-target CoM gap can be shown geometrically.
        self._c_ref_all = _c_ref_all

        # TRUE CoM reference for W^c (replaces the pelvis proxy above, which made W^c fight
        # every limb motion). Calibrate the per-part SMPL masses ONCE (morphology-only),
        # then compute the CoM incrementally per frame in the loop (causal / online). Falls
        # back to the pelvis _cddot_ref_all when the SMPL body is unavailable.
        _smpl_calib = None
        _hb = getattr(self.smplx_ground_probe, "human_body", None) if self.smplx_ground_probe else None
        if _hb is not None and getattr(_hb, "_betas", None) is not None:
            try:
                from HoloNew.src.test_socp.smpl_com import calibrate_smpl_com
                _smpl_calib = calibrate_smpl_com(_hb)
            except Exception:  # noqa: BLE001 - degrade gracefully to the pelvis proxy
                _smpl_calib = None
        _com_ref_prev = _com_ref_prev2 = None

        # Previous two solved CoMs for W^c finite-difference stencil.
        # Initialised to the init-config CoM so the warm-up frames are stable.
        if self.activate_centroidal:
            _c0_init = _com_init.copy()
            _c_prev = _c0_init.copy()   # CoM at frame t-1
            _c_prev2 = _c0_init.copy()  # CoM at frame t-2
        else:
            _c_prev = None
            _c_prev2 = None
        pelvis_grounded = self.gmr_grounded[:, 0]   # (T, 3) grounded SMPL-X pelvis per frame

        # Brick 5 — Movable entities: precompute reference object motion from
        # _obj_poses_raw using causal SE(3) finite differences, and initialise
        # the two prior solved object poses.
        # V_ref[t] = log6(M_{t-1}^{-1} M_t).vector / dt   (6-vector [v; omega])
        # vdot_ref[t] = (V_ref[t][:3] - V_ref[t-1][:3]) / dt
        # omega_ref[t] = V_ref[t][3:6]
        # Frames 0 and 1 get zero references (guard: W^o only fires at frame_idx>=2).
        _vdot_ref_obj_all: list = [None] * T
        _omega_ref_obj_all: list = [None] * T
        _obj_prev = None   # solved object pose at t-1
        _obj_prev2 = None  # solved object pose at t-2
        self._obj_solved_poses = []
        if self.activate_tm and getattr(self, "_obj_poses_raw", None) is not None:
            from HoloNew.src.test_socp.movable import pose_to_se3
            _V_ref_obj = [np.zeros(6)] * T
            for _t in range(1, T):
                _M_prev = pose_to_se3(self._obj_poses_raw[_t - 1])
                _M_cur = pose_to_se3(self._obj_poses_raw[_t])
                _V_ref_obj[_t] = pin.log6(_M_prev.inverse() * _M_cur).vector / self._dt
            for _t in range(2, T):
                _vdot_ref_obj_all[_t] = (_V_ref_obj[_t][:3] - _V_ref_obj[_t - 1][:3]) / self._dt
                _omega_ref_obj_all[_t] = _V_ref_obj[_t][3:6]
            # Initialise prior solved poses to the reference at frame 0.
            _obj_prev = self._obj_poses_raw[0].copy()
            _obj_prev2 = self._obj_poses_raw[0].copy()

        probe_pts, probe_obj, probe_flr, probe_wit, probe_flr_wit, g1_pts = [], [], [], [], [], []
        urdf = None
        if self.smplx_ground_probe is not None:
            from HoloNew.src.test_socp.contact.backends.floor import floor_field

        # Initialise cross-frame persistence state when P is active.
        # p_prev_world (M,3): previous solved robot control-point world positions.
        # obj_prev (7,): previous object pose [qw,qx,qy,qz,x,y,z].
        # d_prev_obj/flr (M,): previous SOLVED robot-side distances (α̂^{t-1}).
        # a_prev_obj/flr (M,): previous source activations α^{t-1}.
        # Initialised to "no previous contact": d_prev=+inf, a_prev=0.
        if (self.lambda_p > 0 or self.activate_persistence) \
                and getattr(self, "correspondence", None) is not None \
                and getattr(self, "smplx_ground_probe", None) is not None:
            _M = self.correspondence.link_idx.shape[0]
            self._p_state = {
                "p_prev_world": np.zeros((_M, 3), dtype=np.float64),
                "obj_prev": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                "d_prev_obj": np.full(_M, np.inf),
                "a_prev_obj": np.zeros(_M),
                "d_prev_flr": np.full(_M, np.inf),
                "a_prev_flr": np.zeros(_M),
            }
        else:
            self._p_state = None

        if self.smplx_ground_probe is not None and self.correspondence is not None:
            import yourdfpy

            from HoloNew.src.test_socp.correspondence.transport import (
                link_world_transforms,
                transported_points,
            )
            urdf = yourdfpy.URDF.load(self.task_constants.ROBOT_URDF_FILE,
                                      load_meshes=False, build_scene_graph=True)

        for t in tqdm(range(T), desc="TEST-SOCP"):
            # Drive object free-joint qpos per frame when non-penetration is on
            # and object poses were loaded.  Guard: has_dynamic_object ensures q
            # actually has the trailing 7 object DOFs (flag off → False → skipped).
            if self.has_dynamic_object and getattr(self, "_obj_poses_mj", None) is not None:
                q[-7:] = self._obj_poses_mj[t]
            # Online SMPL-X -> object-SDF query for this frame (causal: reads only t).
            # Available as smplx_field for the step; also recorded for inspection.
            if self.smplx_ground_probe is not None:
                # AMASS clips pose the probe body from the 22 SMPL-order joints; OMOMO
                # uses the full MuJoCo-order per-joint quats.
                _probe_q = (self._smplx_orientations[t]
                            if getattr(self, "_smplx_orientations", None) is not None
                            else self.human_quat[t])
                pf = self.smplx_ground_probe(t, _probe_q, pelvis_grounded[t])
                self.smplx_sdf_fields.append(pf.field)
                probe_pts.append(pf.points)
                probe_obj.append(pf.field.distance.copy())
                probe_wit.append(pf.field.witness.copy())
                _flr_field = floor_field(pf.points, self.smplx_ground_probe.margin)
                probe_flr.append(_flr_field.distance.copy())
                probe_flr_wit.append(_flr_field.witness.copy())  # world-frame (probe projected to z=0)
            # Single IK pass per frame (TEST is no longer two-pass): one merged table
            # that tracks every body (position + orientation) with strong pelvis/feet
            # placement. GMR-SOCP (two-pass) remains the separate comparison solver.
            tg = ground_frame_targets(gpos[t], gquat[t], IK_MATCH_TABLE_SINGLE)
            _fs = self.foot_sticking_sequences[t] if self.foot_sticking_sequences else None
            # Object: reference pose at t (the W^o_pos anchor target) and the FEED-FORWARD
            # warm-start linearization point = previous SOLVED pose advanced by the
            # reference's per-frame increment (accumulates the grasp correction across
            # frames AND pre-applies the reference motion, so the SQP only fixes the
            # contact residual — no lag on fast object motion). Frame 0 / movable-off
            # fall back to the raw reference.
            _obj_pose_ref = (self._obj_poses_raw[t]
                             if getattr(self, "_obj_poses_raw", None) is not None else None)
            if (self.activate_tm and _obj_pose_ref is not None
                    and t >= 1 and _obj_prev is not None):
                from HoloNew.src.test_socp.movable import feedforward_object_warmstart
                _obj_pose = feedforward_object_warmstart(
                    self._obj_poses_raw[t], self._obj_poses_raw[t - 1], _obj_prev)
            else:
                _obj_pose = _obj_pose_ref
            # CoM-acceleration reference for W^c. Prefer the TRUE SMPL CoM (per-part rigid
            # FK from this frame's body orientations + grounded pelvis), computed
            # incrementally with a causal 2nd difference (no look-ahead). Fall back to the
            # batch pelvis proxy when SMPL is unavailable.
            if _smpl_calib is not None:
                from HoloNew.src.test_socp.smpl_com import smpl_com_from_pose
                _probe_q_com = (self._smplx_orientations[t]
                                if getattr(self, "_smplx_orientations", None) is not None
                                else self.human_quat[t])
                _com_ref_t = smpl_com_from_pose(_smpl_calib, _probe_q_com, pelvis_grounded[t])
                if t >= 2 and _com_ref_prev2 is not None:
                    _cddot_ref_t = (_com_ref_t - 2.0 * _com_ref_prev + _com_ref_prev2) / (self._dt ** 2)
                else:
                    _cddot_ref_t = np.zeros(3)
                # CoM VELOCITY reference (W^c_vel): causal first difference, from frame 1.
                _cdot_ref_t = ((_com_ref_t - _com_ref_prev) / self._dt
                               if (t >= 1 and _com_ref_prev is not None) else np.zeros(3))
                _com_ref_prev2 = _com_ref_prev
                _com_ref_prev = _com_ref_t
            else:
                _cddot_ref_t = _cddot_ref_all[t]
                _cdot_ref_t = None   # velocity reference needs the SMPL CoM trajectory
            # c_ref: reference CoM = reference pelvis + structural com-pelvis offset.
            # Keeps the W^c_pos anchor on the correct trajectory without height bias.
            _c_ref_t = _c_ref_all[t]
            # Object history for W^o.
            _vdot_ref_obj_t = _vdot_ref_obj_all[t]
            _omega_ref_obj_t = _omega_ref_obj_all[t]
            _n_iter = self.n_iter_first if t == 0 else self.n_iter_per_frame
            # Reference object pose at t-1 (for the object↔floor persistence P slide target).
            _obj_pose_ref_tm1 = (self._obj_poses_raw[t - 1]
                                 if (getattr(self, "_obj_poses_raw", None) is not None and t >= 1)
                                 else None)
            q, _frame_cost, _frame_solved_obj = self.iterate(q, q, q_prev, tg, n_iter=_n_iter,
                                frame_idx=t, foot_sticking=_fs, obj_pose=_obj_pose,
                                obj_pose_ref=_obj_pose_ref,
                                obj_pose_ref_tm1=_obj_pose_ref_tm1,
                                q_t_last2=q_prev2,
                                c_tm1=_c_prev, c_tm2=_c_prev2, cddot_ref=_cddot_ref_t,
                                cdot_ref=_cdot_ref_t,
                                c_ref=_c_ref_t,
                                obj_pose_tm1=_obj_prev, obj_pose_tm2=_obj_prev2,
                                vdot_ref_obj=_vdot_ref_obj_t, omega_ref_obj=_omega_ref_obj_t)
            if _frame_solved_obj is None and _obj_pose_ref is not None:
                # movable off / not solved this frame: the object sits at its reference.
                _frame_solved_obj = _obj_pose_ref
            if _frame_solved_obj is not None:
                self._obj_solved_poses.append(_frame_solved_obj.copy())
            # Shift two-frame history: q_prev2 <- q_prev (frame t-2 slot) BEFORE
            # q_prev <- q (frame t-1 slot). Order matters: reverse shift would lose q_prev.
            q_prev2 = np.copy(q_prev)
            q_prev = np.copy(q)

            # Update CoM history for W^c (only when centroidal is active to avoid
            # the pinocchio FK cost when the feature is unused).
            if self.activate_centroidal:
                _c_new = self.pin.com(self.pin.qpos_mj_to_q_pin(q[:36]))
                _c_prev2 = _c_prev   # shift: t-2 slot <- t-1
                _c_prev = _c_new     # t-1 slot <- newly solved t

            # Update solved object pose history for W^o (shift two-frame window).
            if self.activate_tm and _frame_solved_obj is not None:
                _obj_prev2 = _obj_prev
                _obj_prev = _frame_solved_obj

            # Update cross-frame persistence state after the solved frame.
            if self._p_state is not None and _frame_solved_obj is not None:
                from HoloNew.src.test_socp.interaction import (
                    robot_control_points, query_entities, frame_references, _activation,
                )
                _q_pin_solved = self.pin.qpos_mj_to_q_pin(q[:36])
                _L_obj = self.L_object
                _L_flr = self.L_floor
                _p_world = robot_control_points(self, _q_pin_solved)
                # Persistence state reflects the SOLVED object pose (not the warm-start).
                _fobj_s, _fflr_s = query_entities(
                    self, _p_world, _frame_solved_obj, margin_obj=_L_obj, margin_flr=_L_flr)
                _d_obj_ref, _, _d_flr_ref, _, _ = frame_references(self, t)
                self._p_state["p_prev_world"] = _p_world
                self._p_state["obj_prev"] = _frame_solved_obj.copy()
                self._p_state["d_prev_obj"] = np.asarray(_fobj_s.distance, dtype=np.float64)
                self._p_state["d_prev_flr"] = np.asarray(_fflr_s.distance, dtype=np.float64)
                self._p_state["a_prev_obj"] = np.array(
                    [_activation(float(_d_obj_ref[i]), _L_obj) for i in range(len(_d_obj_ref))])
                self._p_state["a_prev_flr"] = np.array(
                    [_activation(float(_d_flr_ref[i]), _L_flr) for i in range(len(_d_flr_ref))])
            if urdf is not None:
                Tw = link_world_transforms(urdf, q, self.correspondence.link_names)
                g1_pts.append(transported_points(
                    Tw, self.correspondence.link_idx,
                    self.correspondence.offset_local, self.correspondence.link_names))
            # The SOLVED object pose is part of the result: write it back into the object
            # free-joint qpos so res.qpos[:, -7:] carries the OPTIMIZED object (not the
            # reference set pre-solve at line ~1053). Reorder pose7 [qw,qx,qy,qz,x,y,z] ->
            # MuJoCo [x,y,z,qw,qx,qy,qz]. Gated on has_dynamic_object: only then does q
            # have the trailing 7 object DOFs (otherwise q[-7:] are robot joints).
            if self.has_dynamic_object and _frame_solved_obj is not None:
                q[-7:] = np.concatenate([_frame_solved_obj[4:7], _frame_solved_obj[0:4]])
            out.append(np.copy(q))
            costs.append(float(_frame_cost))

        res = RetargetResult(qpos=np.array(out), stages={}, cost=0.0)
        res.per_frame_cost = np.asarray(costs, dtype=np.float64)
        if probe_pts:
            res.human_probe_pts = np.stack(probe_pts)
            res.human_obj_dist = np.stack(probe_obj)
            res.human_flr_dist = np.stack(probe_flr)
            res.human_witness = np.stack(probe_wit)
            res.human_flr_witness = np.stack(probe_flr_wit)
            if g1_pts:
                res.g1_transport_pts = np.stack(g1_pts)
                res.human_idx = self.correspondence.human_idx
        # Object-as-carrier surface samples (object<->floor channel). Static object-local
        # set; the viewer lifts it per frame by the solved/reference object pose.
        if getattr(self, "object_surface_local", None) is not None:
            res.object_surface_local = np.asarray(self.object_surface_local)
        # Diagnostics for the viewer. The solved object pose is already collected at
        # no extra cost; CoM / angular momentum / foot-slip are computed post-hoc
        # from the solved trajectory only when collect_diagnostics is on (keeps the
        # normal solve fast).
        if self._obj_solved_poses:
            res.solved_object_poses = np.asarray(self._obj_solved_poses)
        if getattr(self, "collect_diagnostics", False):
            self._fill_diagnostics(res)
        return res

    def _fill_diagnostics(self, res) -> None:
        """Post-hoc CoM, centroidal angular momentum, and foot slip from res.qpos.

        Cheap one-pass diagnostics for the viewer (no effect on the solve). CoM via
        pinocchio; L via the centroidal map A_G on the causal joint velocity; foot
        slip = mean tangential motion of active floor foot points vs the reference.
        """
        import pinocchio as pin
        T = res.qpos.shape[0]
        com = np.zeros((T, 3))
        L = np.zeros((T, 3))
        for t in range(T):
            q_pin = self.pin.qpos_mj_to_q_pin(res.qpos[t, :36])
            com[t] = self.pin.com(q_pin)
            if t >= 1:
                q_prev = self.pin.qpos_mj_to_q_pin(res.qpos[t - 1, :36])
                v = pin.difference(self.pin.model, q_prev, q_pin) / self._dt
                L[t] = (self.pin.centroidal_map(q_pin) @ v)[3:6]
        res.com = com
        res.angular_momentum = L
        # Targets used by the centroidal weights, for the viewer to draw alongside the
        # solved quantities (the grounded reference CoM and the W^L reference L).
        res.com_ref = getattr(self, "_c_ref_all", None)
        res.angular_momentum_ref = getattr(self, "_L_ref_all", None)
        # Foot slip needs the floor probe + correspondence (the floor field is always built).
        if (getattr(self, "smplx_ground_probe", None) is not None
                and getattr(self, "correspondence", None) is not None):
            from HoloNew.src.test_socp.interaction import (
                _activation, frame_references, query_entities, robot_control_points)
            corr = self.correspondence
            M = corr.link_idx.shape[0]
            Lm = self.L_floor  # foot-slip is a floor-channel diagnostic
            foot = [i for i in range(M) if any(k in corr.link_names[corr.link_idx[i]].lower()
                    for k in ("ankle", "foot", "toe"))]
            I3 = np.eye(3)
            slip = np.zeros(T)
            obj_raw = getattr(self, "_obj_poses_raw", None)
            for t in range(1, T):
                qp = self.pin.qpos_mj_to_q_pin(res.qpos[t, :36])
                qpm = self.pin.qpos_mj_to_q_pin(res.qpos[t - 1, :36])
                Pt = robot_control_points(self, qp)
                Ptm = robot_control_points(self, qpm)
                op = obj_raw[t] if obj_raw is not None else None
                _, fflr = query_entities(self, Pt, op, margin=Lm)
                _, _, d_flr_t, _, pr_t = frame_references(self, t)
                _, _, _, _, pr_tm = frame_references(self, t - 1)
                vals = []
                for i in foot:
                    if _activation(float(d_flr_t[i]), Lm) > 0 and bool(fflr.active[i]):
                        n0 = np.asarray(fflr.direction[i], float)
                        Pi = I3 - np.outer(n0, n0)
                        vals.append(np.linalg.norm(Pi @ ((Pt[i]-Ptm[i]) - (pr_t[i]-pr_tm[i]))))
                slip[t] = float(np.mean(vals)) if vals else 0.0
            res.foot_slip = slip

    # ------------------------------------------------------------------
    # Class method factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg) -> "TestSocpRetargeter":
        """Build a TestSocpRetargeter from a config (see builder.build_from_config)."""
        from .builder import build_from_config
        return build_from_config(cls, cfg)
