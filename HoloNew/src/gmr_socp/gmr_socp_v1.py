"""GMR-SOCP retargeter v1 — position + orientation tracking objective.

Derived from src/interaction_mesh_retargeter.py (InteractionMeshRetargeter).
Strips all visualization, self-collision, foot-lock, and interaction-mesh
machinery and replaces the Laplacian objective with a GMR tracking objective
(one term per robot frame, weighted by the IK table pos_weight and rot_weight).

Both position and orientation tracking are included in this version.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import cvxpy as cp
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

# Fix up sys.path so that mujoco_utils, utils, viser_utils (in src/) are
# importable. This file lives in src/gmr_socp/, so parent.parent is src/.
_src_path = Path(__file__).parent.parent
sys.path.insert(0, str(_src_path))

from .tables import IK_MATCH_TABLE1  # noqa: E402

# Body name remapping: keys are IK table frame names; values are actual G1
# MuJoCo body names.  Only entries that differ from the table key are listed.
# GMR's smplx_to_g1.json uses "left_toe_link" / "right_toe_link" but the G1
# model (g1_29dof.xml) does not have those bodies — the most distal foot body
# is left_ankle_roll_link / right_ankle_roll_link.
_BODY_NAME_REMAP: dict[str, str] = {
    "left_toe_link": "left_ankle_roll_link",
    "right_toe_link": "right_ankle_roll_link",
}


class GmrSocpRetargeterV1:
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
        print(f"[GmrSocpV1] Loading robot model from: {robot_xml_path}")
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

        # Build robot_link_names: map each IK table frame -> actual G1 body name,
        # applying the remap for the two missing toe bodies.
        available_bodies = {self.robot_model.body(i).name for i in range(self.robot_model.nbody)}
        self.robot_link_names: dict[str, str] = {}
        for frame in IK_MATCH_TABLE1:
            actual = _BODY_NAME_REMAP.get(frame, frame)
            bid = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, actual)
            if bid == -1:
                raise ValueError(
                    f"[GmrSocpV1] Body '{actual}' (mapped from table key '{frame}') "
                    f"not found in model. Available: {sorted(available_bodies)}"
                )
            if actual != frame:
                print(f"[GmrSocpV1] Remapped body: '{frame}' -> '{actual}'")
            self.robot_link_names[frame] = actual

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
    ):
        """Iterate solve_single_iteration until convergence or n_iter steps."""
        last = np.inf
        cost = 0.0
        for _ in range(n_iter):
            q_n, cost = self.solve_single_iteration(
                q_locked, q_n[self.q_a_indices], q_t_last, frame_targets
            )
            if np.isclose(cost, last):
                break
            last = cost
        return q_n, cost

    def retarget(self):
        """Run the full two-pass GMR solve over all frames.

        Requires from_config to have been called first (sets self.human_pos,
        self.human_quat, self.q_init_full).

        Returns:
            RetargetResult with qpos (T, 7+DOF) trajectory.
        """
        from HoloNew.src.retarget_result import RetargetResult
        from .tables import IK_MATCH_TABLE1, IK_MATCH_TABLE2
        from .targets import build_frame_targets

        T = self.human_pos.shape[0]
        q = np.copy(self.q_init_full)
        out = []

        for t in range(T):
            tg1 = build_frame_targets(self.human_pos[t], self.human_quat[t], IK_MATCH_TABLE1)
            tg2 = build_frame_targets(self.human_pos[t], self.human_quat[t], IK_MATCH_TABLE2)
            q, _ = self.iterate(q, q, q, tg1, n_iter=(50 if t == 0 else 10))
            q, _ = self.iterate(q, q, q, tg2, n_iter=(50 if t == 0 else 10))
            out.append(np.copy(q))

        return RetargetResult(qpos=np.array(out), stages={}, cost=0.0)

    # ------------------------------------------------------------------
    # Class method factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg) -> "GmrSocpRetargeterV1":
        """Build a GmrSocpRetargeterV1 and populate its motion inputs.

        Replicates the input-prep sequence of examples/robot_retarget.py main()
        for the robot_only / smplh path.

        Args:
            cfg: RetargetingConfig instance (task_type must be "robot_only",
                data_format must be "smplh" or None).

        Returns:
            Configured GmrSocpRetargeterV1 ready to call .retarget().
        """
        from pathlib import Path as _Path

        from HoloNew.config_types.data_type import MotionDataConfig
        from HoloNew.config_types.robot import RobotConfig
        from HoloNew.examples.robot_retarget import (
            DEFAULT_DATA_FORMATS,
            build_retargeter_kwargs_from_config,
            create_task_constants,
            initialize_robot_pose,
            load_motion_data,
        )
        from HoloNew.src.utils import preprocess_motion_data
        from .targets import load_pt_quaternions

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

        # Load motion data (human_joints, object_poses, smpl_scale)
        human_joints, object_poses, smpl_scale = load_motion_data(
            task_type, data_format, cfg.data_path, cfg.task_name,
            constants, cfg.motion_data_config
        )

        toe_names = cfg.motion_data_config.toe_names

        # Build retargeter kwargs and construct the retargeter
        kwargs = build_retargeter_kwargs_from_config(
            cfg.retargeter, constants, object_urdf_path=None, task_type=task_type
        )
        rt = cls(**kwargs)

        # Preprocess human joints (scale + ground-normalise)
        human_joints = preprocess_motion_data(
            human_joints, rt, toe_names, smpl_scale
        )

        # Initialize robot pose (pelvis translation + orientation)
        q_init, _q_nominal, object_poses_mj, human_joints, _obj_poses = initialize_robot_pose(
            task_type=task_type,
            data_format=data_format,
            human_joints=human_joints,
            object_poses=object_poses,
            constants=constants,
            retargeter=rt,
            task_config=cfg.task_config,
            augmentation=cfg.augmentation,
            save_dir=(cfg.save_dir or _Path("demo_results/g1/robot_only/omomo")),
            task_name=cfg.task_name,
        )

        # Build q_init_full: fill a zero array of length nq with q_init
        q_init_full = np.zeros(rt.nq)
        q_init_full[:len(q_init)] = q_init

        # Load per-joint quaternions from the .pt file
        pt_path = cfg.data_path / f"{cfg.task_name}.pt"
        human_quat = load_pt_quaternions(pt_path)  # (T_q, 52, 4) wxyz

        # Align T between human_joints (T_j, J, 3) and human_quat (T_q, 52, 4)
        T_j = human_joints.shape[0]
        T_q = human_quat.shape[0]
        T = min(T_j, T_q)
        human_joints = human_joints[:T]
        human_quat = human_quat[:T]

        rt.human_pos = human_joints   # (T, J, 3)
        rt.human_quat = human_quat    # (T, 52, 4) wxyz
        rt.q_init_full = q_init_full  # (nq,)

        return rt
