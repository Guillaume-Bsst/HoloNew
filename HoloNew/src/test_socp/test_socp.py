"""GMR-SOCP retargeter v2 — identical copy of v1, to be evolved later.

Derived from src/holosoma/interaction_mesh_retargeter.py (InteractionMeshRetargeter).
Strips all visualization, self-collision, foot-lock, and interaction-mesh
machinery and replaces the Laplacian objective with a GMR tracking objective
(one term per robot frame, weighted by the IK table pos_weight and rot_weight).

Both position and orientation tracking are included in this version.
"""
from __future__ import annotations

from pathlib import Path
from types import ModuleType

import cvxpy as cp
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from HoloNew.config_types.retargeter import FootLockConfig, SelfCollisionConfig
from .tables import IK_MATCH_TABLE1

# Body name remapping: keys are IK table frame names; values are actual G1
# MuJoCo body names.  Only entries that differ from the table key are listed.
# GMR's smplx_to_g1.json uses "left_toe_link" / "right_toe_link" but the G1
# model (g1_29dof.xml) does not have those bodies — the most distal foot body
# is left_ankle_roll_link / right_ankle_roll_link.
_BODY_NAME_REMAP: dict[str, str] = {
    "left_toe_link": "left_ankle_roll_link",
    "right_toe_link": "right_ankle_roll_link",
}


class TestSocpRetargeter:
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
        activate_obj_non_penetration: bool = False,
        activate_self_collision: bool = False,
        activate_foot_sticking: bool = False,
        penetration_tolerance: float = 1e-3,
        foot_sticking_tolerance: float = 1e-3,
        foot_lock=None,            # FootLockConfig | None
        self_collision=None,       # SelfCollisionConfig | None
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
        self.visualize = False
        self.demo_joints = task_constants.DEMO_JOINTS

        # object_name must be known before the model is loaded so the gated xml
        # selection below can reference it.
        self.object_name = getattr(task_constants, "OBJECT_NAME", "ground")

        # Load MuJoCo model.  Default (flag off or ground): plain robot xml.
        # When activate_obj_non_penetration is on AND the task has a real object,
        # swap to the object-scene xml so the collision geometry is present.
        robot_xml_path = task_constants.ROBOT_URDF_FILE.replace(".urdf", ".xml")
        if activate_obj_non_penetration and self.object_name not in (None, "ground"):
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
        self.q_a_lb = complete_lower[self.q_a_indices]
        self.q_a_ub = complete_upper[self.q_a_indices]

        if task_constants.MANUAL_LB:
            self.q_a_lb[np.array(list(task_constants.MANUAL_LB.keys()), dtype=int)] = list(
                task_constants.MANUAL_LB.values()
            )
        if task_constants.MANUAL_UB:
            self.q_a_ub[np.array(list(task_constants.MANUAL_UB.keys()), dtype=int)] = list(
                task_constants.MANUAL_UB.values()
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

        # Contact assets (loaded by from_config from the bundled artifacts).
        # None until from_config populates them; not used in the solve yet.
        self.object_sdf = None
        self.contact_fields = None

        # Online SMPL-X -> SDF probe + its per-frame outputs (set by from_config).
        self.smplx_ground_probe = None
        self.smplx_sdf_fields: list = []

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

    # ------------------------------------------------------------------
    # Holosoma-style constraint init helpers (copied verbatim from
    # src/holosoma/interaction_mesh_retargeter.py)
    # ------------------------------------------------------------------

    def _init_foot_lock(self, foot_lock: FootLockConfig | None) -> None:
        """Initialize foot lock configuration and normalize window mappings."""
        self.foot_lock = foot_lock or FootLockConfig()
        self._foot_lock_windows: dict[str, tuple[tuple[int, int], ...]] = {"left": (), "right": ()}
        if self.foot_lock.windows is None:
            return
        for key, windows in self.foot_lock.windows.items():
            key_lower = key.lower()
            side = None
            if key_lower.startswith("l") or ("left" in key_lower):
                side = "left"
            elif key_lower.startswith("r") or ("right" in key_lower):
                side = "right"
            if side is None:
                continue

            normalized_windows: list[tuple[int, int]] = []
            for window in windows:
                if len(window) != 2:
                    raise ValueError(f"Invalid foot lock window for {key}: {window}")
                start, end = int(window[0]), int(window[1])
                if end < start:
                    raise ValueError(f"Invalid foot lock window with end < start for {key}: {window}")
                normalized_windows.append((start, end))
            self._foot_lock_windows[side] = tuple(normalized_windows)

    def _init_self_collision(self, self_collision: SelfCollisionConfig | None) -> None:
        """Initialize self-collision configuration and precompute geom pairs."""
        sc = self_collision or SelfCollisionConfig()
        self._self_collision_enabled = sc.enable and len(sc.pairs) > 0
        self._self_collision_tolerance = sc.tolerance
        self._self_collision_windows: list[tuple[int, int]] | None = sc.windows
        self._self_collision_geom_pairs: list[tuple[int, int]] = []

        self._sc_last_vis_frame = -1

        if not self._self_collision_enabled:
            return

        m = self.robot_model

        # Build body_name → [geom_ids] mapping (only geoms with collision enabled)
        body_to_geoms: dict[str, list[int]] = {}
        for g in range(m.ngeom):
            if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
                continue
            body_id = m.geom_bodyid[g]
            body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            body_to_geoms.setdefault(body_name, []).append(g)

        # Build geom pairs from body name pairs
        for body_a, body_b in sc.pairs:
            geoms_a = body_to_geoms.get(body_a, [])
            geoms_b = body_to_geoms.get(body_b, [])
            if not geoms_a:
                print(f"[SelfCollision] Warning: no collision geoms found for body '{body_a}'")
            if not geoms_b:
                print(f"[SelfCollision] Warning: no collision geoms found for body '{body_b}'")
            for ga in geoms_a:
                for gb in geoms_b:
                    self._self_collision_geom_pairs.append((ga, gb))

        print(
            f"[SelfCollision] Initialized with {len(self._self_collision_geom_pairs)} geom pairs "
            f"from {len(sc.pairs)} body pairs, tolerance={sc.tolerance}m"
        )

    def _prefilter_pairs_with_mj_collision(self, threshold: float):
        m, d = self.robot_model, self.robot_data
        ngeom = m.ngeom

        self._geom_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(ngeom)]

        if not hasattr(self, "_saved_margins"):
            self._saved_margins = np.empty_like(m.geom_margin)
        self._saved_margins[:] = m.geom_margin

        m.geom_margin[:] = threshold

        # Run collision. This runs broad→narrow and fills d.contact.
        mujoco.mj_collision(m, d)

        # Collect unique candidate pairs that involve at least one masked geom
        candidates = set()
        for k in range(d.ncon):
            c = d.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 < 0 or g2 < 0:
                continue
            candidates.add((min(g1, g2), max(g1, g2)))

        # Restore margins to keep physics untouched
        m.geom_margin[:] = self._saved_margins

        return candidates

    def _compute_jacobian_for_contact_relative(self, geom1, geom2, geom1_name, geom2_name, fromto, dist):
        # Get closest points from fromto buffer
        pos1 = fromto[:3]  # closest point on geom1
        pos2 = fromto[3:]  # closest point on geom2

        v = pos1 - pos2
        norm_v = np.linalg.norm(v)

        if norm_v > 1e-12:
            nhat_BA_W = np.sign(dist) * (v / norm_v)
        # Degenerate: points coincide. Heuristics fallback.
        # If one side is a plane/ground, use its known normal.
        elif "ground" in geom2_name.lower():
            nhat_BA_W = np.array([0.0, 0.0, 1.0]) * (1.0 if dist >= 0 else -1.0)
        elif "ground" in geom1_name.lower():
            nhat_BA_W = np.array([0.0, 0.0, -1.0]) * (1.0 if dist >= 0 else -1.0)
        else:
            nhat_BA_W = np.array([0.0, 0.0, 0.0])

        J_bodyA = self._calc_contact_jacobian_from_point(geom1.bodyid, pos1, input_world=True)
        J_bodyB = self._calc_contact_jacobian_from_point(geom2.bodyid, pos2, input_world=True)

        # Compute relative Jacobian
        Jc = J_bodyA - J_bodyB

        return nhat_BA_W @ Jc

    def _is_foot_locked_in_window(self, foot_link_key: str, frame_idx: int) -> bool:
        """Check whether a foot link is locked by configured frame windows."""
        key_lower = foot_link_key.lower()
        side = None
        if "left" in key_lower:
            side = "left"
        elif "right" in key_lower:
            side = "right"
        if side is None:
            return False

        return any(start <= frame_idx <= end for start, end in self._foot_lock_windows.get(side, ()))

    def _compute_self_collision_constraints(self, frame_idx: int):
        """Compute Jacobians and distances for self-collision body pairs.

        Assumes ``mj_forward`` has already been called with the current q
        (done by ``_update_jacobians_and_phis_from_q`` which runs first).

        Returns:
            Js: dict mapping (geom_a, geom_b) -> relative Jacobian (1 x nq)
            phis: dict mapping (geom_a, geom_b) -> signed distance
        """
        if not self._self_collision_enabled:
            return {}, {}

        # Check frame windows
        if self._self_collision_windows is not None:
            if not any(start <= frame_idx <= end for start, end in self._self_collision_windows):
                return {}, {}

        m, d = self.robot_model, self.robot_data
        threshold = float(self.collision_detection_threshold)

        Js, phis = {}, {}
        fromto = np.zeros(6, dtype=float)

        if not hasattr(self, "_geom_names"):
            raise RuntimeError(
                "[SelfCollision] _geom_names not initialized. Please run _prefilter_pairs_with_mj_collision first."
            )

        _first_iter = self._sc_last_vis_frame != frame_idx
        if _first_iter:
            self._sc_last_vis_frame = frame_idx

        for geom_a, geom_b in self._self_collision_geom_pairs:
            fromto[:] = 0.0
            dist = mujoco.mj_geomDistance(m, d, geom_a, geom_b, threshold, fromto)
            if dist <= threshold:
                J_rel = self._compute_jacobian_for_contact_relative(
                    m.geom(geom_a),
                    m.geom(geom_b),
                    self._geom_names[geom_a],
                    self._geom_names[geom_b],
                    fromto,
                    dist,
                )
                key = ("self", geom_a, geom_b)
                Js[key] = J_rel
                phis[key] = float(dist)

        if _first_iter and self.visualize:
            self._draw_self_collision_geoms()

        return Js, phis

    def _update_jacobians_and_phis_from_q(self, q: np.ndarray):
        self.robot_data.qpos[:] = q

        mujoco.mj_forward(self.robot_model, self.robot_data)  # kinematics & AABBs valid

        m, d = self.robot_model, self.robot_data
        threshold = float(self.collision_detection_threshold)

        # 1) Fast prefilter via mj_collision with temporary margins
        candidates = self._prefilter_pairs_with_mj_collision(threshold)

        Js, phis = {}, {}
        fromto = np.zeros(6, dtype=float)

        # 2) Precise distance only on candidates (early-exit at threshold)
        contype, conaff = m.geom_contype, m.geom_conaffinity

        def masks_ok(g1, g2):
            if contype[g1] == 0 and conaff[g1] == 0:
                return False
            if contype[g2] == 0 and conaff[g2] == 0:
                return False
            if self.object_name in self._geom_names[g1] and "ground" in self._geom_names[g2]:
                return False
            if "ground" in self._geom_names[g1] and self.object_name in self._geom_names[g2]:
                return False
            return (
                self.object_name in self._geom_names[g1]
                or self.object_name in self._geom_names[g2]
                or "ground" in self._geom_names[g1]
                or "ground" in self._geom_names[g2]
            )

        for g1, g2 in candidates:
            # Optional: keep your own filters here (e.g., skip object-ground, only keep interaction with object/ground)
            if not masks_ok(g1, g2):
                continue

            fromto[:] = 0.0
            dist = mujoco.mj_geomDistance(m, d, g1, g2, threshold, fromto)
            if dist <= threshold:
                J_rel = self._compute_jacobian_for_contact_relative(
                    m.geom(g1), m.geom(g2), self._geom_names[g1], self._geom_names[g2], fromto, dist
                )
                Js[(g1, g2)] = J_rel
                phis[(g1, g2)] = float(dist)

                # For debug
                # self.draw_mesh_pair_with_contact(self.robot_model, self.robot_data, g1, g2,   \
                #     self._geom_names[g1], self._geom_names[g2], fromto=fromto)

        return Js, phis

    # ------------------------------------------------------------------
    # Jacobian helpers (kept verbatim from InteractionMeshRetargeter)
    # ------------------------------------------------------------------

    def _build_transform_qdot_to_qvel_fast(self, use_world_omega: bool = True) -> np.ndarray:
        """Return T(q) (nv x nq) such that v = T(q) @ qdot."""
        nq, nv = self.robot_model.nq, self.robot_model.nv
        T = np.zeros((nv, nq), dtype=float)

        j0 = 0
        assert self.robot_model.jnt_type[j0] == mujoco.mjtJoint.mjJNT_FREE

        def get_e_world(qw, qx, qy, qz):
            return np.array(
                [
                    [-qx, qw, qz, -qy],
                    [-qy, -qz, qw, qx],
                    [-qz, qy, -qx, qw],
                ]
            )

        def get_e_body(qw, qx, qy, qz):
            return np.array(
                [
                    [-qx, qw, -qz, qy],
                    [-qy, qz, qw, -qx],
                    [-qz, -qy, qx, qw],
                ]
            )

        E_fn = get_e_world if use_world_omega else get_e_body

        j_free1 = 0
        assert self.robot_model.jnt_type[j_free1] == mujoco.mjtJoint.mjJNT_FREE
        qadr1 = int(self.robot_model.jnt_qposadr[j_free1])
        dadr1 = int(self.robot_model.jnt_dofadr[j_free1])

        qw, qx, qy, qz = self.robot_data.qpos[qadr1 + 3: qadr1 + 7]
        E1 = 2.0 * E_fn(qw, qx, qy, qz)
        T[dadr1 + 0: dadr1 + 3, qadr1 + 0: qadr1 + 3] = np.eye(3)
        T[dadr1 + 3: dadr1 + 6, qadr1 + 3: qadr1 + 7] = E1

        if self.has_dynamic_object:
            free_joints = [
                j for j in range(self.robot_model.njnt)
                if self.robot_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
            ]
            assert len(free_joints) >= 2, "Expected two FREE joints (robot + object)."
            j_free2 = free_joints[1]
            qadr2 = int(self.robot_model.jnt_qposadr[j_free2])
            dadr2 = int(self.robot_model.jnt_dofadr[j_free2])
            qw, qx, qy, qz = self.robot_data.qpos[qadr2 + 3: qadr2 + 7]
            E2 = 2.0 * E_fn(qw, qx, qy, qz)
            T[dadr2 + 0: dadr2 + 3, qadr2 + 0: qadr2 + 3] = np.eye(3)
            T[dadr2 + 3: dadr2 + 6, qadr2 + 3: qadr2 + 7] = E2

        for j in range(1, self.robot_model.njnt):
            jt = self.robot_model.jnt_type[j]
            if jt in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                qa = self.robot_model.jnt_qposadr[j]
                da = self.robot_model.jnt_dofadr[j]
                T[da, qa] = 1.0
            elif jt == mujoco.mjtJoint.mjJNT_BALL:
                raise NotImplementedError("BALL joint block not implemented.")

        return T

    def _calc_contact_jacobian_from_point(
        self, body_idx: int, p_body: np.ndarray, input_world: bool = False
    ) -> np.ndarray:
        """Translational Jacobian J(q) (3 x nq) s.t. v_point_world = J @ qdot."""
        p_body = np.asarray(p_body, dtype=float).reshape(3)
        mujoco.mj_forward(self.robot_model, self.robot_data)

        R_WB = self.robot_data.xmat[body_idx].reshape(3, 3)
        p_WB = self.robot_data.xpos[body_idx]

        if input_world:
            p_W = p_body.astype(np.float64).reshape(3, 1)
        else:
            p_W = (p_WB + R_WB @ p_body).astype(np.float64).reshape(3, 1)

        Jp = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
        Jr = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
        mujoco.mj_jac(self.robot_model, self.robot_data, Jp, Jr, p_W, int(body_idx))

        T_mat = self._build_transform_qdot_to_qvel_fast()
        return Jp @ T_mat

    def _calc_manipulator_jacobians(
        self,
        q: np.ndarray,
        links: dict[str, str],
        obj_frame: bool = False,
        point_offsets: np.ndarray | None = None,
    ):
        """Compute position Jacobians (3 x nq_a) and world positions per frame.

        Returns (J_dict, p_dict, P_WO) matching the native retargeter's API.
        """
        J_XC_dict: dict[str, np.ndarray] = {}
        p_XC_dict: dict[str, np.ndarray] = {}

        if obj_frame:
            if self.has_dynamic_object:
                obj_quat = q[-4:]
                obj_pos = q[-7:-4]
                obj_rot = Rotation.from_quat(
                    [obj_quat[1], obj_quat[2], obj_quat[3], obj_quat[0]]
                ).as_matrix()
                obj_rot_inv = obj_rot.T
            else:
                obj_rot = Rotation.from_quat([0, 0, 0, 1]).as_matrix()
                obj_rot_inv = obj_rot.T
                obj_pos = np.zeros(3)
        else:
            obj_pos = np.zeros(3)
            obj_rot = None
            obj_rot_inv = None

        self.robot_data.qpos[:] = q.copy()
        mujoco.mj_forward(self.robot_model, self.robot_data)

        for name, link_name in links.items():
            body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
            pC_B = point_offsets if point_offsets is not None else np.zeros(3)

            J = self._calc_contact_jacobian_from_point(body_id, pC_B)
            pos_world = self.robot_data.xpos[body_id]

            if obj_frame and obj_rot_inv is not None:
                p_XC = obj_rot_inv @ (pos_world - obj_pos)
                J_XC = obj_rot_inv @ J
            else:
                p_XC = pos_world
                J_XC = J

            J_XC_dict[name] = np.array(J_XC[:, self.q_a_indices], dtype=float, copy=True)
            p_XC_dict[name] = np.array(p_XC, dtype=float, copy=True)

        P_WO = ({"position": obj_pos, "rotation": obj_rot}
                if obj_frame else None)
        return J_XC_dict, p_XC_dict, P_WO

    def _get_robot_link_positions(self, q: np.ndarray, link_names) -> np.ndarray:
        """World positions for each link name, computed via pinocchio FK.

        Args:
            q: Full configuration vector (length nq; robot part is q[:36]).
            link_names: Iterable of link (body) names.

        Returns:
            Array of shape (N, 3) with world positions.
        """
        q_pin = self.pin.qpos_mj_to_q_pin(q[:36])
        return np.array([self.pin.body_position(q_pin, n) for n in link_names])

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
            q_a_n_last: Actuated-DoF slice at the current iterate (length nq_a).
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

        dqa = cp.Variable(self.nv_a, name="dqa")

        obj_terms = []
        for frame, (p_t, R_t, w_p, w_r) in frame_targets.items():
            body = self.robot_link_names[frame]
            Jp, Jr = self._body_jac(q, body)

            if w_p > 0:
                p_c = self.body_position(q, body)
                obj_terms.append(
                    w_p * cp.sum_squares(Jp @ dqa - (p_t - p_c))
                )

            if w_r > 0:
                R_c = self.body_rotation(q, body)
                # Orientation error in body frame; body-frame angular Jacobian.
                # Convention (b): body-frame Jacobian = R_c.T @ Jr,
                # body-frame error = log(R_c.T @ R_t).
                # This is the convention that makes the linearisation consistent:
                # body_omega ≈ (R_c.T @ Jr) @ dqa  and  e = log(R_c.T @ R_t).
                e = Rotation.from_matrix(R_c.T @ R_t).as_rotvec()
                Jr_body = R_c.T @ Jr
                obj_terms.append(
                    w_r * cp.sum_squares(Jr_body @ dqa - e)
                )

        constraints = [cp.SOC(self.step_size, dqa)]
        if self.activate_joint_limits:
            # Tangent-space joint limit box: precomputed absolute bounds minus
            # current joint values.  For hinge/slide joints, the pinocchio tangent
            # increment equals (target - current) directly (same as subtraction).
            # Base DOFs (tangent 0:6) use large bounds and are effectively unconstrained.
            # Tangent index vi >= 6 maps to pinocchio q index vi + 1 (offset by 1
            # because the root FREE joint occupies 7 qpos DOFs but only 6 velocity DOFs).
            q_pin_cur = self.pin.qpos_mj_to_q_pin(q[:36])
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

                        Jxy = J_WF[:2, self.q_a_indices]  # (2 x nq_act)
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
                    Jz = J_WF[2, self.q_a_indices]
                    constraints += [
                        Jz @ dqa >= z_delta - self.foot_lock.tolerance,
                        Jz @ dqa <= z_delta + self.foot_lock.tolerance,
                    ]

        # Self-collision constraints (holosoma-style, default off)
        if self.activate_self_collision and self._self_collision_enabled:
            Js_sc, phis_sc = self._compute_self_collision_constraints(frame_idx)
            for key, phi in phis_sc.items():
                Ja_n_full = Js_sc[key]
                Ja_n = Ja_n_full[self.q_a_indices]
                # Enforce: new_distance >= tolerance  =>  phi + J @ dqa >= tol
                rhs = self._self_collision_tolerance - phi
                constraints += [Ja_n @ dqa >= rhs]

        # Non-penetration constraints (holosoma-style, default off)
        if self.activate_obj_non_penetration:
            Js, phis = self._update_jacobians_and_phis_from_q(q)
            for key, phi in phis.items():
                Ja_n_full = Js[key]
                Ja_n = Ja_n_full[self.q_a_indices]
                # Enforce: phi + J @ dqa >= -tol  (keep signed distance above -tolerance).
                rhs = -phi - self.penetration_tolerance
                constraints += [Ja_n @ dqa >= rhs]

        prob = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)
        prob.solve(solver=cp.CLARABEL)

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"GMR-SOCP solve failed: {prob.status}")

        import pinocchio as pin
        v_full = np.zeros(self.pin.model.nv)
        v_full[self.v_a_indices] = dqa.value
        q_pin_new = pin.integrate(
            self.pin.model,
            self.pin.qpos_mj_to_q_pin(q[:36]),
            v_full,
        )
        q_star = np.copy(q)
        q_star[:36] = self.pin.q_pin_to_qpos_mj(q_pin_new)
        # pin.integrate keeps the quaternion unit; no manual renormalisation needed.
        return q_star, float(prob.value)

    def iterate(
        self,
        q_locked: np.ndarray,
        q_n: np.ndarray,
        q_t_last: np.ndarray,
        frame_targets: dict,
        n_iter: int = 10,
        frame_idx: int = 0,
        foot_sticking: tuple[bool, bool] | None = None,
    ):
        """Iterate solve_single_iteration until convergence or n_iter steps."""
        last = np.inf
        cost = 0.0
        for _ in range(n_iter):
            q_n, cost = self.solve_single_iteration(
                q_locked, q_n[self.q_a_indices], q_t_last, frame_targets,
                frame_idx=frame_idx, foot_sticking=foot_sticking,
            )
            if np.isclose(cost, last):
                break
            last = cost
        return q_n, cost

    def retarget(self):
        """Run the full two-pass GMR solve over all frames.

        Requires from_config to have been called first (sets self.gmr_ground,
        self.q_init_full).

        Returns:
            RetargetResult with qpos (T, 7+DOF) trajectory.
        """
        from tqdm import tqdm

        from HoloNew.src.retarget_result import RetargetResult
        from .tables import IK_MATCH_TABLE1, IK_MATCH_TABLE2
        from .targets import ground_frame_targets

        gpos = self.gmr_ground["pos"]
        gquat = self.gmr_ground["quat"]
        T = gpos.shape[0]
        q = np.copy(self.q_init_full)
        out = []
        # q_prev: previous-frame foot anchor for foot-sticking; both passes use the same anchor,
        # updated once per frame (after both passes). Frame 0 anchors to init config.
        q_prev = np.copy(self.q_init_full)
        pelvis_grounded = self.gmr_grounded[:, 0]   # (T, 3) grounded SMPL-X pelvis per frame

        probe_pts, probe_obj, probe_flr, probe_wit, g1_pts = [], [], [], [], []
        urdf = None
        if self.smplx_ground_probe is not None:
            from HoloNew.src.test_socp.contact.backends.floor import floor_field
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
                pf = self.smplx_ground_probe(t, self.human_quat[t], pelvis_grounded[t])
                self.smplx_sdf_fields.append(pf.field)
                probe_pts.append(pf.points)
                probe_obj.append(pf.field.distance.copy())
                probe_wit.append(pf.field.witness.copy())
                probe_flr.append(floor_field(pf.points, self.smplx_ground_probe.margin).distance.copy())
            # GMR fidelity: both passes track the SAME table1-offset 'ground' targets;
            # only the cost weights differ (table1 -> pass 1, table2 -> pass 2).
            tg1 = ground_frame_targets(gpos[t], gquat[t], IK_MATCH_TABLE1)
            tg2 = ground_frame_targets(gpos[t], gquat[t], IK_MATCH_TABLE2)
            _fs = self.foot_sticking_sequences[t] if self.foot_sticking_sequences else None
            q, _ = self.iterate(q, q, q_prev, tg1, n_iter=(50 if t == 0 else 10),
                                frame_idx=t, foot_sticking=_fs)
            q, _ = self.iterate(q, q, q_prev, tg2, n_iter=(50 if t == 0 else 10),
                                frame_idx=t, foot_sticking=_fs)
            q_prev = np.copy(q)
            if urdf is not None:
                Tw = link_world_transforms(urdf, q, self.correspondence.link_names)
                g1_pts.append(transported_points(
                    Tw, self.correspondence.link_idx,
                    self.correspondence.offset_local, self.correspondence.link_names))
            out.append(np.copy(q))

        res = RetargetResult(qpos=np.array(out), stages={}, cost=0.0)
        if probe_pts:
            res.human_probe_pts = np.stack(probe_pts)
            res.human_obj_dist = np.stack(probe_obj)
            res.human_flr_dist = np.stack(probe_flr)
            res.human_witness = np.stack(probe_wit)
            if g1_pts:
                res.g1_transport_pts = np.stack(g1_pts)
                res.human_idx = self.correspondence.human_idx
        return res

    # ------------------------------------------------------------------
    # Class method factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg) -> "TestSocpRetargeter":
        """Build a TestSocpRetargeter and populate its motion inputs.

        Loads motion data directly from the .pt file without going through
        holosoma's preprocess_motion_data or initialize_robot_pose, since the
        GMR retarget uses only compute_stages' 'ground' output and the base
        init is fully overridden by the ground pelvis position and orientation.

        Args:
            cfg: RetargetingConfig instance (task_type must be "robot_only",
                data_format must be "smplh" or None).

        Returns:
            Configured TestSocpRetargeter ready to call .retarget().
        """
        from HoloNew.config_types.data_type import MotionDataConfig
        from HoloNew.config_types.robot import RobotConfig
        from HoloNew.examples.robot_retarget import (
            DEFAULT_DATA_FORMATS,
            build_retargeter_kwargs_from_config,
            create_task_constants,
        )
        from HoloNew.src.holosoma.preprocess import calculate_scale_factor, ground_to_floor
        from .preprocess import compute_stages
        from .tables import HUMAN_ROOT_NAME, MAPPED_BODY_NAMES
        from .targets import load_pt_joints, load_pt_quaternions

        task_type = cfg.task_type
        data_format = cfg.data_format or DEFAULT_DATA_FORMATS[task_type]

        # Ensure robot / motion configs are consistent
        if cfg.robot_config.robot_type != cfg.robot:
            cfg.robot_config = RobotConfig(robot_type=cfg.robot)
        if (cfg.motion_data_config.robot_type != cfg.robot
                or cfg.motion_data_config.data_format != data_format):
            cfg.motion_data_config = MotionDataConfig(
                data_format=data_format, robot_type=cfg.robot
            )

        constants = create_task_constants(
            robot_config=cfg.robot_config,
            motion_data_config=cfg.motion_data_config,
            task_config=cfg.task_config,
            task_type=task_type,
        )

        # Build retargeter kwargs and construct the retargeter
        kwargs = build_retargeter_kwargs_from_config(
            cfg.retargeter, constants, object_urdf_path=None, task_type=task_type
        )
        # Holosoma-style constraints are OPT-IN for TEST-SOCP and default OFF
        # (holosoma's RetargeterConfig defaults them ON). Take the activate_*
        # flags from a TEST-SOCP-specific config: honor one if the caller passed it,
        # else force OFF so the default solve is unchanged.
        from .config import TestSocpRetargeterConfig
        sc = cfg.retargeter if isinstance(cfg.retargeter, TestSocpRetargeterConfig) else TestSocpRetargeterConfig()
        kwargs["activate_obj_non_penetration"] = sc.activate_obj_non_penetration
        kwargs["activate_foot_sticking"] = sc.activate_foot_sticking
        kwargs["activate_self_collision"] = sc.activate_self_collision
        rt = cls(**kwargs)

        # Load raw joint positions and per-joint quaternions from the .pt file
        pt_path = cfg.data_path / f"{cfg.task_name}.pt"
        raw_joints = load_pt_joints(pt_path)    # (T, 52, 3) raw positions
        human_quat = load_pt_quaternions(pt_path)  # (T, 52, 4) wxyz

        # Align T between raw_joints and human_quat (both come from the same
        # file so they are equal in length, but guard against edge cases)
        T = min(raw_joints.shape[0], human_quat.shape[0])
        raw_joints = raw_joints[:T]
        human_quat = human_quat[:T]

        # Build per-frame foot sticking sequence from SMPL joint velocities.
        # Gated by activate_foot_sticking (False by default) in solve; building the
        # sequence here is harmless and does not affect the default solve path.
        from HoloNew.src.utils import extract_foot_sticking_sequence_velocity
        toe_names = cfg.motion_data_config.toe_names
        rt.foot_sticking_sequences = extract_foot_sticking_sequence_velocity(
            raw_joints, constants.DEMO_JOINTS, toe_names)

        # All joints are zero; base is overridden below from the ground pelvis
        q_init_full = np.zeros(rt.nq)

        rt.human_quat = human_quat    # (T, 52, 4) wxyz
        rt.q_init_full = q_init_full  # (nq,) — base will be set from ground below

        # Ground the raw input onto the floor first (like holosoma) so every downstream
        # stage lives in the grounded world. GMR's own floor correction (the 'ground'
        # stage) re-grounds afterwards — a constant z-shift it cancels out — so the solved
        # targets are unchanged, but the mapped/scaled/offset stages now follow the
        # grounded input and the 'Grounded' display stage is the real chain input.
        toe_indices = [constants.DEMO_JOINTS.index(n) for n in cfg.motion_data_config.toe_names]
        rt.gmr_grounded = ground_to_floor(raw_joints, toe_indices)
        # Pull the root toward the world centre by holosoma's scale factor
        # (ROBOT_HEIGHT / human_height) so the GMR base XY matches holosoma's
        # globally-scaled placement. compute_stages applies this as a rigid XY
        # translation, preserving GMR's body proportions and the Z floor-drop.
        smpl_scale = calculate_scale_factor(cfg.task_name, constants.ROBOT_HEIGHT)
        rt.gmr_stages = compute_stages(
            rt.gmr_grounded, human_quat, anchor_root_xy=True, root_xy_scale=smpl_scale
        )
        rt.gmr_ground = rt.gmr_stages["ground"]
        ground = rt.gmr_ground
        _pelvis_bi = MAPPED_BODY_NAMES.index(HUMAN_ROOT_NAME)
        rt.q_init_full[:3] = ground["pos"][0, _pelvis_bi]    # base at frame-0 pelvis target
        rt.q_init_full[3:7] = ground["quat"][0, _pelvis_bi]  # base orientation at frame-0 target

        # Load object poses in MuJoCo qpos order for per-frame object qpos drive.
        # Only when the flag is on and the task has a real object; otherwise leave
        # None so the retarget loop's object-qpos block is always skipped (parity).
        rt._obj_poses_mj = None
        if sc.activate_obj_non_penetration and rt.object_name not in (None, "ground"):
            from HoloNew.examples.robot_retarget import convert_object_poses_to_mujoco_order
            from HoloNew.src.utils import load_intermimic_data
            _, obj_poses = load_intermimic_data(str(pt_path))   # (T, 7) [qw,qx,qy,qz,x,y,z]
            obj_poses = obj_poses[:T]
            # Convert from [qw,qx,qy,qz,x,y,z] to MuJoCo order [x,y,z,qw,qx,qy,qz]
            rt._obj_poses_mj = convert_object_poses_to_mujoco_order(obj_poses)

        # Fail loudly on a misconfigured object scene: poses loaded but the model
        # has no object free joint would silently skip the per-frame object drive.
        if rt._obj_poses_mj is not None and not rt.has_dynamic_object:
            raise RuntimeError(
                f"[{cls.__name__}] Object poses loaded but has_dynamic_object is False: "
                f"the scene xml for '{rt.object_name}' did not add a free joint. "
                "Check SCENE_XML_FILE / robot_urdf_file naming."
            )

        # Load the bundled human->G1 correspondence table (data only,
        # NOT used in the solve yet — will be wired in a later task).
        from pathlib import Path
        from HoloNew.src.test_socp.correspondence.build_correspondence import load_correspondence, build_table
        from HoloNew.src.test_socp.correspondence.constants import (
            G1_29DOF_URDF, SMPLX_MODEL_DIR_DEFAULT, HUMAN_GRID_DENSITY, G1_DENSITY, OT_REG,
        )
        _bundled = Path(__file__).resolve().parent.parent.parent / "assets" / "correspondence" / "corr_neutral.npz"
        if _bundled.exists():
            rt.correspondence = load_correspondence(_bundled)
        elif Path(SMPLX_MODEL_DIR_DEFAULT).is_dir():
            rt.correspondence = build_table(SMPLX_MODEL_DIR_DEFAULT, "neutral", None,
                                            G1_29DOF_URDF, HUMAN_GRID_DENSITY, G1_DENSITY, OT_REG)

        # Load bundled contact assets (data only — NOT used in the solve yet;
        # will be wired into the objective in a later task).
        from HoloNew.src.test_socp.contact.backends.sdf import load_object_sdf
        from HoloNew.src.test_socp.contact.contact_io import load_contact_fields
        _contact_assets = Path(__file__).resolve().parent.parent.parent / "assets" / "contact"
        _sdf_path = _contact_assets / "largebox_sdf.npz"
        _contact_path = _contact_assets / f"contact_{cfg.task_name}.npz"
        if _sdf_path.exists():
            rt.object_sdf = load_object_sdf(_sdf_path)
        if _contact_path.exists():
            rt.contact_fields = load_contact_fields(_contact_path)

        # Online SMPL-X -> object-SDF probe (causal, per frame). Built only when the
        # object SDF is available: sample the subject SMPL-X surface once. The human is
        # placed at its Grounded pose in retarget(); the object pose is used as-is (the
        # raw human floats, the object sits correctly, so only the human is grounded).
        if rt.object_sdf is not None:
            from HoloNew.src.test_socp.contact.constants import CONTACT_MARGIN_M, OMOMO_DIR_DEFAULT
            from HoloNew.src.test_socp.contact.smplx_field import build_smplx_ground_probe
            from HoloNew.src.test_socp.correspondence.human_body import PointCloudCache
            from HoloNew.src.utils import load_intermimic_data
            _, obj_poses = load_intermimic_data(str(pt_path))   # (T, 7) [qw,qx,qy,qz,x,y,z]
            corr_cache = None
            if rt.correspondence is not None:
                corr_cache = PointCloudCache(tri_idx=rt.correspondence.tri_idx,
                                             bary=rt.correspondence.bary)
            rt.smplx_ground_probe = build_smplx_ground_probe(
                cfg.task_name, OMOMO_DIR_DEFAULT, SMPLX_MODEL_DIR_DEFAULT,
                rt.object_sdf, obj_poses[:T], CONTACT_MARGIN_M, HUMAN_GRID_DENSITY,
                cache=corr_cache)

        return rt
