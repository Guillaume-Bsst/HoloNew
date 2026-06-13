"""GMR-SOCP retargeter v1 — position + orientation tracking objective.

Derived from src/holosoma/interaction_mesh_retargeter.py (InteractionMeshRetargeter).
Strips all visualization, self-collision, foot-lock, and interaction-mesh
machinery and replaces the Laplacian objective with a GMR tracking objective
(one term per robot frame, weighted by the IK table pos_weight and rot_weight).

Both position and orientation tracking are included in this version.
"""
from __future__ import annotations

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


class GmrSocpRetargeter:
    """Position + orientation tracking GMR-SOCP retargeter (v1).

    Solves a two-pass linearised IK problem using a trust-region SOCP.
    The objective is a sum of weighted squared-error terms (one per robot
    frame), combining position and orientation residuals.  Frame targets
    are produced by build_frame_targets and have the form:

        {frame: (p_target (3,), R_target (3,3), w_p, w_r)}

    where ``w_p`` weights the translational term and ``w_r`` the rotational
    term.  Either weight may be zero to disable that term.
    """

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

        # Load MuJoCo model (robot_only uses the plain .xml, no object)
        robot_xml_path = task_constants.ROBOT_URDF_FILE.replace(".urdf", ".xml")
        self.robot_model = mujoco.MjModel.from_xml_path(robot_xml_path)
        print(f"[GmrSocp] Loading robot model from: {robot_xml_path}")
        self.robot_data = mujoco.MjData(self.robot_model)

        if self.robot_data.qpos.shape[0] > 7 + task_constants.ROBOT_DOF:
            self.has_dynamic_object = True
        else:
            self.has_dynamic_object = False

        self.nq = self.robot_model.nq
        self.q_a_init_idx = q_a_init_idx
        self.q_a_indices = np.arange(7 + q_a_init_idx, 7 + task_constants.ROBOT_DOF)
        self.nq_a = len(self.q_a_indices)

        # Joint limits
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

        # ===== Holosoma-style optional constraints (default OFF; copied verbatim
        # from src/holosoma/interaction_mesh_retargeter.py). When every flag is
        # off the solve is unchanged. =====
        self.activate_obj_non_penetration = activate_obj_non_penetration
        self.activate_self_collision = activate_self_collision
        self.activate_foot_sticking = activate_foot_sticking
        self.penetration_tolerance = penetration_tolerance
        self.foot_sticking_tolerance = foot_sticking_tolerance
        self.object_name = getattr(task_constants, "OBJECT_NAME", "ground")
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
                    f"[GmrSocp] Body '{actual}' (mapped from table key '{frame}') "
                    f"not found in model. Available: {sorted(available_bodies)}"
                )
            if actual != frame:
                print(f"[GmrSocp] Remapped body: '{frame}' -> '{actual}'")
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
        """Get world positions for each link name given configuration q."""
        self.robot_data.qpos[:] = q.copy()
        mujoco.mj_forward(self.robot_model, self.robot_data)
        positions = []
        for link_name in link_names:
            body_id = mujoco.mj_name2id(
                self.robot_model, mujoco.mjtObj.mjOBJ_BODY, link_name
            )
            if body_id == -1:
                raise ValueError(f"Body '{link_name}' not found in MuJoCo model")
            positions.append(self.robot_data.xpos[body_id].copy())
        return np.array(positions)

    def _body_jac(self, q: np.ndarray, body_name: str):
        """World-frame (Jp, Jr) for a body, reduced to actuated columns.

        Both Jacobians map actuated-DoF increments (dqa, shape nq_a) to
        world-frame body velocity / angular velocity.

        Args:
            q: Full configuration vector (length nq).
            body_name: MuJoCo body name.

        Returns:
            Tuple (Jp, Jr) each of shape (3, nq_a).
        """
        self.robot_data.qpos[:] = q
        mujoco.mj_forward(self.robot_model, self.robot_data)

        bid = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        jacp = np.zeros((3, self.robot_model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.robot_model.nv), dtype=np.float64)
        mujoco.mj_jacBody(self.robot_model, self.robot_data, jacp, jacr, bid)

        # T shape: (nv, nq) — maps qdot -> qvel (same convention used in
        # _calc_contact_jacobian_from_point for the position Jacobian).
        T = self._build_transform_qdot_to_qvel_fast()
        Jp_qa = (jacp @ T)[:, self.q_a_indices]  # (3, nq_a)
        Jr_qa = (jacr @ T)[:, self.q_a_indices]  # (3, nq_a)
        return Jp_qa, Jr_qa

    def body_position(self, q: np.ndarray, body_name: str) -> np.ndarray:
        """World position of ``body_name`` at configuration ``q``.

        Args:
            q: Full configuration vector (length nq).
            body_name: MuJoCo body name.

        Returns:
            Position array of shape (3,).
        """
        self.robot_data.qpos[:] = q
        mujoco.mj_forward(self.robot_model, self.robot_data)
        bid = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.robot_data.xpos[bid].copy()

    def body_rotation(self, q: np.ndarray, body_name: str) -> np.ndarray:
        """World rotation matrix of ``body_name`` at configuration ``q``.

        Args:
            q: Full configuration vector (length nq).
            body_name: MuJoCo body name.

        Returns:
            Rotation matrix of shape (3, 3).
        """
        self.robot_data.qpos[:] = q
        mujoco.mj_forward(self.robot_model, self.robot_data)
        bid = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.robot_data.xmat[bid].reshape(3, 3).copy()

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
                compatibility; unused in v1 which has no smoothness cost).
            frame_targets: {frame: (p_target(3,), R_target(3,3), w_p, w_r)}
                as returned by build_frame_targets.
            init_t: True on the very first frame (unused in v1, kept for compat).
            frame_idx: Index of the current frame; used for window filtering by
                the holosoma-style constraints (self-collision / foot) when enabled.
            foot_sticking: Per-foot sticking flags (left, right) for this frame;
                used by the foot-sticking constraint when enabled.

        Returns:
            (q_star, cost): updated full config and objective value.
        """
        q = np.copy(q_locked)
        q[self.q_a_indices] = q_a_n_last

        dqa = cp.Variable(self.nq_a, name="dqa")

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
            constraints += [
                dqa >= (self.q_a_lb - q_a_n_last),
                dqa <= (self.q_a_ub - q_a_n_last),
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

        q_star = np.copy(q)
        q_star[self.q_a_indices] = dqa.value + q_a_n_last
        # Renormalise quaternion
        q_star[3:7] /= np.linalg.norm(q_star[3:7]) + 1e-12
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

        for t in tqdm(range(T), desc="GMR-SOCP"):
            # GMR fidelity: both passes track the SAME table1-offset 'ground' targets;
            # only the cost weights differ (table1 -> pass 1, table2 -> pass 2).
            tg1 = ground_frame_targets(gpos[t], gquat[t], IK_MATCH_TABLE1)
            tg2 = ground_frame_targets(gpos[t], gquat[t], IK_MATCH_TABLE2)
            _fs = self.foot_sticking_sequences[t] if self.foot_sticking_sequences else None
            q, _ = self.iterate(q, q, q, tg1, n_iter=(50 if t == 0 else 10),
                                frame_idx=t, foot_sticking=_fs)
            q, _ = self.iterate(q, q, q, tg2, n_iter=(50 if t == 0 else 10),
                                frame_idx=t, foot_sticking=_fs)
            out.append(np.copy(q))

        return RetargetResult(qpos=np.array(out), stages={}, cost=0.0)

    # ------------------------------------------------------------------
    # Class method factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg) -> "GmrSocpRetargeter":
        """Build a GmrSocpRetargeter and populate its motion inputs.

        Loads motion data directly from the .pt file without going through
        holosoma's preprocess_motion_data or initialize_robot_pose, since the
        GMR retarget uses only compute_stages' 'ground' output and the base
        init is fully overridden by the ground pelvis position and orientation.

        Args:
            cfg: RetargetingConfig instance (task_type must be "robot_only",
                data_format must be "smplh" or None).

        Returns:
            Configured GmrSocpRetargeter ready to call .retarget().
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
        # Holosoma-style constraints are OPT-IN for GMR-SOCP and default OFF
        # (holosoma's RetargeterConfig defaults them ON). Take the activate_*
        # flags from a GMR-specific config: honor one if the caller passed it,
        # else force OFF so the default solve is unchanged.
        from .config import GmrSocpRetargeterConfig
        sc = cfg.retargeter if isinstance(cfg.retargeter, GmrSocpRetargeterConfig) else GmrSocpRetargeterConfig()
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

        return rt
