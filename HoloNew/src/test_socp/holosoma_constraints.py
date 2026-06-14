"""Holosoma-style optional constraints + their MuJoCo Jacobians (mixin).

Foot-lock, self-collision, and the MuJoCo-based contact/manipulator Jacobians used
by TestSocpRetargeter's default-off holosoma-style constraints. Extracted verbatim
as a mixin to keep the core solver file focused. Every method operates on ``self``
(the retargeter instance) and uses attributes set in TestSocpRetargeter.__init__
(self.robot_model, self.robot_data, self.pin, self.foot_lock, ...).
"""
from __future__ import annotations

import mujoco
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from HoloNew.config_types.retargeter import FootLockConfig, SelfCollisionConfig


class HolosomaConstraintsMixin:
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
            Js: dict mapping (geom_a, geom_b) -> relative Jacobian (nv,) in pinocchio tangent order
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
    # Jacobian helpers
    # ------------------------------------------------------------------

    def _calc_contact_jacobian_from_point(
        self, body_idx: int, p_body: np.ndarray, input_world: bool = False
    ) -> np.ndarray:
        """Translational Jacobian J(q) (3 x nv) in pinocchio tangent space.

        Computes the Jacobian of a fixed point on a body using pinocchio's
        point_translational_jacobian.  MuJoCo body_idx is converted to a body
        name; the contact point is expressed in the body's local frame.

        Static bodies (i.e. 'world' / ground, not present in the URDF-derived
        pinocchio model) have a zero Jacobian because they cannot move.

        Args:
            body_idx: MuJoCo body index (integer).
            p_body: Contact point.  If input_world is False (default), the
                point is in the body-local frame.  If input_world is True, the
                point is in the world frame and is first projected into the
                body-local frame via the current pinocchio FK pose.
            input_world: Whether p_body is given in world coordinates.

        Returns:
            Translational Jacobian of shape (3, nv) in pinocchio tangent order.
        """
        p_body = np.asarray(p_body, dtype=float).reshape(3)
        body_idx_int = int(np.asarray(body_idx).flat[0])
        body_name = mujoco.mj_id2name(
            self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_idx_int
        ) or ""
        # Static bodies (e.g. world/ground) are not in the URDF pinocchio model;
        # their velocity Jacobian is identically zero.
        fid = self.pin.model.getFrameId(body_name)
        if fid >= self.pin.model.nframes:
            return np.zeros((3, self.pin.model.nv))
        q_pin = self.pin.qpos_mj_to_q_pin(self.robot_data.qpos[:36])
        if input_world:
            # Convert world-frame point to body-local frame using pinocchio FK pose.
            R_WB = self.pin.body_rotation(q_pin, body_name)
            p_WB = self.pin.body_position(q_pin, body_name)
            offset_local = R_WB.T @ (p_body - p_WB)
        else:
            offset_local = p_body
        return self.pin.point_translational_jacobian(q_pin, body_name, offset_local)

    def _calc_manipulator_jacobians(
        self,
        q: np.ndarray,
        links: dict[str, str],
        obj_frame: bool = False,
        point_offsets: np.ndarray | None = None,
    ):
        """Compute position Jacobians (3 x nv_a) and world positions per frame.

        Uses pinocchio point_translational_jacobian internally; Jacobians are
        sliced to v_a_indices and expressed in pinocchio tangent order.

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

        # Sync MuJoCo data for world positions (collision/SDF still uses MuJoCo).
        self.robot_data.qpos[:] = q.copy()
        mujoco.mj_forward(self.robot_model, self.robot_data)

        q_pin = self.pin.qpos_mj_to_q_pin(q[:36])

        for name, link_name in links.items():
            body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
            pC_B = point_offsets if point_offsets is not None else np.zeros(3)

            # Jacobian (3, nv) in pinocchio tangent order.
            J_nv = self.pin.point_translational_jacobian(q_pin, link_name, pC_B)
            pos_world = self.robot_data.xpos[body_id]

            if obj_frame and obj_rot_inv is not None:
                p_XC = obj_rot_inv @ (pos_world - obj_pos)
                J_XC = obj_rot_inv @ J_nv
            else:
                p_XC = pos_world
                J_XC = J_nv

            # Slice to active tangent indices (nv_a columns).
            J_XC_dict[name] = np.array(J_XC[:, self.v_a_indices], dtype=float, copy=True)
            p_XC_dict[name] = np.array(p_XC, dtype=float, copy=True)

        P_WO = ({"position": obj_pos, "rotation": obj_rot}
                if obj_frame else None)
        return J_XC_dict, p_XC_dict, P_WO
