"""Owns the viser scene: robot instance(s), object, grid, keypoint layers.

Extracted from InteractionMeshRetargeter so rendering is separate from compute
and several trajectories can share one viser session. qpos layout matches
holosoma: [0:3] pos, [3:7] wxyz quat, [7:7+dof] actuated joints, optional
trailing [-7:] dynamic-object pose.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np
import trimesh
import viser
from viser.extras import ViserUrdf
import yourdfpy

from . import skeleton
from .stages import ROBOT_STAGE, stages_for_method
from HoloNew.src.holosoma.interaction_mesh import transform_points_local_to_world
from HoloNew.src.test_socp.contact.viz import signed_distance_colors
from HoloNew.src.test_socp.contact.constants import CONTACT_MARGIN_M

# Joint-frame axis radius as a fraction of its length, so the RGB triads stay
# slender at any "Frame size" value (matches test_pipe's add_batched_axes sizing).
AXIS_RADIUS_FRAC = 0.035


@dataclass
class RobotHandle:
    urdf: ViserUrdf
    base: object   # viser frame handle
    dof: int


@dataclass
class MethodViz:
    """A method's robot trajectory plus its named skeleton stages.

    Args:
        label: Method dropdown label (must be one of ``method_labels()``).
        robot_key: Robot instance key; selects ``self.robots[robot_key]``.
        qpos: ``(T, 7 + dof)`` solved robot trajectory.
        stages: Mapping of ``{stage_label: (T, B, 3)}`` skeleton point clouds.
        stage_bones: Optional ``{stage_label: [(i, j), ...]}`` bone index pairs
            for the non-52-joint (mapped) stages and the robot stage, so they
            render as skeletons; 52-joint stages use the built-in SMPLH topology.
        robot_skeleton: Optional ``(T, K, 3)`` solved-robot link world positions
            (drawn on the Robot stage, keyed by ``stage_bones["Robot"]``).
        stage_quats: Optional ``{stage_label: (T, B, 4)}`` per-joint orientations
            (wxyz) for stages that carry them (the mapped GMR stages), so their
            joint frames can be drawn.
        robot_quats: Optional ``(T, K, 4)`` solved-robot link orientations (wxyz),
            for the Robot stage joint frames.
        g1_points: Optional ``(T, K, 3)`` world positions of the G1 morphological-graph
            keypoints (holosoma's interaction-mesh robot links) at the solved pose.
    """
    label: str
    robot_key: str
    qpos: np.ndarray
    stages: dict = field(default_factory=dict)
    stage_bones: dict = field(default_factory=dict)
    robot_skeleton: np.ndarray | None = None
    stage_quats: dict = field(default_factory=dict)
    robot_quats: np.ndarray | None = None
    g1_points: np.ndarray | None = None
    # The object placement THIS method actually used on its scaled stages (MuJoCo order
    # [xyz, wxyz]); the object follows the active method here (GMR centres it by
    # smpl_scale, TEST keeps it raw). None falls back to the viewer's global scaled pose.
    object_pose_scaled: np.ndarray | None = None
    human_probe_pts: np.ndarray | None = None   # (T, N, 3) Grounded-pose SMPL-X probes
    human_dist: np.ndarray | None = None        # (T, N)    signed distance (min(object, floor))
    g1_transport_pts: np.ndarray | None = None  # (T, M, 3) transported points on the robot
    g1_dist: np.ndarray | None = None           # (T, M)    contact colour distance = min(object, floor)
    human_witness: np.ndarray | None = None     # (T, N, 3) object-local witness per human probe
    human_flr_witness: np.ndarray | None = None # (T, N, 3) WORLD-frame floor witness (probe projected to z=0)
    human_obj_dist: np.ndarray | None = None    # (T, N)    signed distance to the object (SDF)
    human_flr_dist: np.ndarray | None = None    # (T, N)    signed distance to the floor (analytic)
    g1_obj_witness: np.ndarray | None = None    # (T, M, 3) object-local witness per G1 point
    g1_obj_dist: np.ndarray | None = None       # (T, M)    each G1 point's human-source object distance
    g1_flr_dist: np.ndarray | None = None       # (T, M)    each G1 point's human-source floor distance
    g1_flr_witness: np.ndarray | None = None    # (T, M, 3) WORLD-frame floor witness per G1 point
    object_surface_local: np.ndarray | None = None  # (M, 3) object-local surface samples (object<->floor carrier)
    # TEST-SOCP solve diagnostics.
    solved_object_poses: np.ndarray | None = None  # (T, 7) the method's SOLVED object pose [wxyz,xyz]
    com: np.ndarray | None = None                  # (T, 3) robot CoM
    com_ref: np.ndarray | None = None              # (T, 3) grounded CoM target (W^c_pos reference)
    angular_momentum: np.ndarray | None = None     # (T, 3) centroidal angular momentum L
    angular_momentum_ref: np.ndarray | None = None # (T, 3) grounded L target (W^L reference)
    foot_slip: np.ndarray | None = None            # (T,)   mean tangential foot slip (no-slip diagnostic)


class Viewer:
    def __init__(self, robot_model_path: str, object_model_path: str | None,
                 stage_keys: tuple[str, ...] = ("socp",),
                 has_dynamic_object: bool = False,
                 original_joints: np.ndarray | None = None,
                 original_quats: np.ndarray | None = None,
                 original_bones: list[tuple[int, int]] | None = None,
                 object_mesh_verts: np.ndarray | None = None,
                 object_mesh_faces: np.ndarray | None = None,
                 object_points_local: np.ndarray | None = None,
                 object_pose_raw: np.ndarray | None = None,
                 object_pose_scaled: np.ndarray | None = None,
                 object_scaled_stages: tuple[str, ...] = (),
                 object_sdf_pts: np.ndarray | None = None,
                 object_sdf_cols: np.ndarray | None = None,
                 object_sdf_floor_pts: np.ndarray | None = None,
                 object_sdf_floor_cols: np.ndarray | None = None,
                 interaction_L_floor: float = CONTACT_MARGIN_M,
                 interaction_L_object: float = CONTACT_MARGIN_M,
                 human_body: object | None = None) -> None:
        # Contact bands used for the overlays' active masks + colour scales, per channel
        # (the run's L_floor / L_object), so the viewer matches what the solver activates.
        self._L_flr = float(interaction_L_floor)
        self._L_obj = float(interaction_L_object)
        self.robot_model_path = robot_model_path
        self.object_model_path = object_model_path
        self.has_dynamic_object = has_dynamic_object
        # Original source motion, shared by every method's "Original" stage.
        self.original_joints = original_joints
        self.original_quats = original_quats
        # Bone topology for the Original source skeleton. None -> the full 52-joint
        # SMPLH layout (BODY_BONES + finger bones); set (e.g. SMPLX_BODY_BONES) when
        # the source carries a different, finger-less joint set (the smplx path).
        self.original_bones = original_bones
        # Stage-dependent object: the mesh + surface samples are kept in the object's
        # local frame and lifted to world per frame by the active stage's pose. On a
        # scaled stage (object_scaled_stages) the centred pose is used, otherwise the raw
        # pose. The object keeps its native SIZE in both — holosoma scales the object's
        # position toward the centre, not its geometry — so only the placement changes.
        self.object_mesh_verts = object_mesh_verts
        self.object_mesh_faces = object_mesh_faces
        self.object_points_local = object_points_local
        self.object_pose_raw = object_pose_raw
        self.object_pose_scaled = object_pose_scaled
        self.object_scaled_stages = frozenset(object_scaled_stages)
        # Object SDF near-surface band shell: object-local points + precomputed
        # signed-distance colours, placed at the active stage's object pose.
        self.object_sdf_pts = object_sdf_pts
        self.object_sdf_cols = object_sdf_cols
        self.object_sdf_floor_pts = object_sdf_floor_pts
        self.object_sdf_floor_cols = object_sdf_floor_cols
        self.human_body = human_body
        self._smplx_handle = None
        # Persistent object handles, updated in place each frame (never removed/recreated)
        # so the object does not flicker during playback, like the SMPL-X mesh.
        self._object_mesh_handle = None
        self._object_pts_handle = None
        self._g1_pts_handle = None
        self._sdf_handle = None
        self._human_handle = None
        self._g1_transport_handle = None
        self._object_contact_handle = None
        self._floor_contact_handle = None
        self._object_surface_handle = None
        self._object_floor_contact_handle = None
        self._sdf_floor_handle = None
        # Persistent centroidal-diagnostic handles (CoM marker + its ground shadow),
        # plus the grounded CoM target marker + shadow drawn alongside.
        self._com_handle = None
        self._com_shadow_handle = None
        self._com_ref_handle = None
        self._com_ref_shadow_handle = None
        self._dynamic_handles: list = []
        self.server = viser.ViserServer()

        # Ensure a world frame exists (absolute path)
        try:
            self.server.scene.add_frame("/world", show_axes=False)
        except Exception:
            print("Starting viser")

        self.robots: dict[str, RobotHandle] = {k: self._add_robot(k) for k in stage_keys}

        self.object_base = None
        self.viser_object = None
        if object_model_path:
            self.object_base = self.server.scene.add_frame("/world/object", show_axes=False)
            obj = yourdfpy.URDF.load(object_model_path, load_meshes=True, build_scene_graph=True)
            self.viser_object = ViserUrdf(self.server, urdf_or_path=obj, root_node_name="/world/object")

        # Add grid
        self.server.scene.add_grid("/world/grid", width=8, height=8, position=(0.0, 0.0, 0.0))

    def _add_robot(self, key: str) -> RobotHandle:
        root = f"/world/robot_{key}"
        base = self.server.scene.add_frame(root, show_axes=False)
        urdf = yourdfpy.URDF.load(self.robot_model_path, load_meshes=True, build_scene_graph=True)
        vu = ViserUrdf(self.server, urdf_or_path=urdf, root_node_name=root)
        dof = len(vu.get_actuated_joint_limits())
        vu.update_cfg(np.zeros(dof))
        return RobotHandle(urdf=vu, base=base, dof=dof)

    def draw_q(self, q: np.ndarray, stage: str = "socp") -> None:
        h = self.robots[stage]
        h.urdf.update_cfg(q[7:7 + h.dof])
        h.base.position = q[:3]
        h.base.wxyz = q[3:7]
        if self.viser_object is not None and self.object_base is not None:
            if self.has_dynamic_object:
                self.object_base.position = q[-7:-4]
                self.object_base.wxyz = q[-4:]
            else:
                self.object_base.position = np.zeros(3)
                self.object_base.wxyz = np.asarray([1.0, 0.0, 0.0, 0.0])

    def draw_keypoints(self, p: np.ndarray, name: str = "keypoint", rgba=(0, 0, 1, 1)):
        sphere = trimesh.primitives.Sphere(radius=0.02)
        color = tuple(int(c * 255) for c in rgba[:3])
        opacity = float(rgba[3])
        if p.ndim == 1:
            return self.server.scene.add_mesh_simple(
                f"/{name}", vertices=sphere.vertices, faces=sphere.faces,
                position=p, color=color, opacity=opacity)
        return self.server.scene.add_batched_meshes_simple(
            f"/{name}", vertices=sphere.vertices, faces=sphere.faces,
            batched_positions=p,
            batched_wxyzs=np.tile(np.array([1, 0, 0, 0]), (p.shape[0], 1)),
            batched_colors=color, opacity=opacity)

    def bind_methods(self, methods: list) -> None:
        """Bind a list of MethodViz and build a Frame slider + Method/Stage dropdowns."""
        self._methods = {m.label: m for m in methods}
        # Dropdowns offer only the methods actually bound (e.g. when --methods
        # selects a subset); otherwise selecting an unsolved method KeyErrors.
        bound_labels = [m.label for m in methods]
        T = min(len(m.qpos) for m in methods)
        self._n_frames = T
        self._playing = False
        self._prog_update = False

        with self.server.gui.add_folder("Playback"):
            self._slider = self.server.gui.add_slider(
                "Frame", min=0, max=max(0, T - 1), step=1, initial_value=0)
            self._play_btn = self.server.gui.add_button("Play / Pause")
            self._fps_in = self.server.gui.add_number("FPS", initial_value=30, min=1, max=240, step=1)
        with self.server.gui.add_folder("Display"):
            first = methods[0].label
            self._method_dd = self.server.gui.add_dropdown(
                "Method", options=bound_labels, initial_value=first)
            self._stage_dd = self.server.gui.add_dropdown(
                "Stage", options=stages_for_method(first), initial_value=ROBOT_STAGE)
            self._tog_urdf = self.server.gui.add_checkbox("Show G1 URDF", True)
            self._tog_smplx = self.server.gui.add_checkbox("SMPL-X mesh", False)
            self._tog_object = self.server.gui.add_checkbox("Object mesh", False)

        with self.server.gui.add_folder("Skeleton"):
            self._tog_body_bones = self.server.gui.add_checkbox("Body bones", True)
            self._tog_finger_bones = self.server.gui.add_checkbox("Finger bones", True)
            self._tog_body_joints = self.server.gui.add_checkbox("Body joints", True)
            self._tog_finger_joints = self.server.gui.add_checkbox("Finger joints", False)

        for _cb in (self._tog_body_bones, self._tog_finger_bones,
                    self._tog_body_joints, self._tog_finger_joints):
            @_cb.on_update
            def _(_evt):
                self._redraw(int(self._slider.value))

        with self.server.gui.add_folder("Joint frames"):
            self._tog_axes = self.server.gui.add_checkbox("Show joint frames", False)
            self._axis_size = self.server.gui.add_slider(
                "Frame size", min=0.02, max=0.5, step=0.01, initial_value=0.1)
        for _cb in (self._tog_axes, self._axis_size):
            @_cb.on_update
            def _(_evt):
                self._redraw(int(self._slider.value))

        with self.server.gui.add_folder("Holosoma"):
            # Holosoma's interaction mesh: object samples paired with the G1
            # morphological-graph keypoints (the robot links the Laplacian matches).
            self._tog_object_pts = self.server.gui.add_checkbox("Object points", False)
            self._tog_g1_pts = self.server.gui.add_checkbox("G1 points", False)
        for _cb in (self._tog_smplx, self._tog_object, self._tog_object_pts, self._tog_g1_pts):
            @_cb.on_update
            def _(_evt):
                self._redraw(int(self._slider.value))

        with self.server.gui.add_folder("Test"):
            # test_pipe frame: object SDF near-surface band shell, coloured by signed distance.
            self._tog_sdf = self.server.gui.add_checkbox("SDF Object", False)
            self._tog_sdf_floor = self.server.gui.add_checkbox("SDF Floor", False)
            self._tog_human = self.server.gui.add_checkbox("Human contact", False)
            self._tog_g1_transport = self.server.gui.add_checkbox("G1 transport", False)
            self._tog_dir_object = self.server.gui.add_checkbox("Object directions", False)
            self._tog_dir_floor = self.server.gui.add_checkbox("Floor directions", False)
            self._tog_object_contact = self.server.gui.add_checkbox("Object contact", False)
            self._tog_floor_contact = self.server.gui.add_checkbox("Floor contact", False)
            # Object-as-carrier <-> floor channel (follows the solved/reference object pose).
            self._tog_obj_surface = self.server.gui.add_checkbox("Object surface pts", False)
            self._tog_dir_obj_floor = self.server.gui.add_checkbox("Object->Floor directions", False)
            self._tog_obj_floor_contact = self.server.gui.add_checkbox("Object->Floor contact", False)

        # TEST-SOCP solve diagnostics: the SOLVED object pose (movable/inertia) and
        # the per-frame CoM / foot-slip / angular-momentum readout.
        with self.server.gui.add_folder("Test diagnostics"):
            self._tog_solved_obj = self.server.gui.add_checkbox("Solved object pose", True)
            self._tog_com = self.server.gui.add_checkbox("CoM + trail", False)
            self._tog_L = self.server.gui.add_checkbox("Angular momentum L", False)
            self._diag_text = self.server.gui.add_text("Solve state", initial_value="", disabled=True)
        for _cb in (self._tog_sdf, self._tog_sdf_floor, self._tog_human,
                    self._tog_g1_transport, self._tog_dir_object, self._tog_dir_floor,
                    self._tog_object_contact, self._tog_floor_contact,
                    self._tog_obj_surface, self._tog_dir_obj_floor, self._tog_obj_floor_contact,
                    self._tog_solved_obj, self._tog_com, self._tog_L):
            @_cb.on_update
            def _(_evt):
                self._redraw(int(self._slider.value))

        def _ghost_stages(label: str) -> list:
            return ["Off"] + [s for s in stages_for_method(label) if s != ROBOT_STAGE]

        with self.server.gui.add_folder("Ghost"):
            first = methods[0].label
            self._ghost_method_dd = self.server.gui.add_dropdown(
                "Method", options=bound_labels, initial_value=first)
            self._ghost_stage_dd = self.server.gui.add_dropdown(
                "Stage", options=_ghost_stages(first), initial_value="Off")

        @self._ghost_method_dd.on_update
        def _(_evt):
            self._ghost_stage_dd.options = _ghost_stages(self._ghost_method_dd.value)
            self._ghost_stage_dd.value = "Off"
            self._redraw(int(self._slider.value))

        @self._ghost_stage_dd.on_update
        def _(_evt):
            self._redraw(int(self._slider.value))

        @self._method_dd.on_update
        def _(_evt):
            self._stage_dd.options = stages_for_method(self._method_dd.value)
            self._stage_dd.value = ROBOT_STAGE
            self._redraw(int(self._slider.value))

        @self._slider.on_update
        def _(_evt):
            if not self._prog_update:
                self._playing = False   # user scrubbing pauses playback
            self._redraw(int(self._slider.value))

        @self._stage_dd.on_update
        def _(_evt): self._redraw(int(self._slider.value))

        @self._tog_urdf.on_update
        def _(_evt): self._redraw(int(self._slider.value))

        @self._play_btn.on_click
        def _(_evt):
            self._playing = not self._playing

        self._redraw(0)
        threading.Thread(target=self._player_loop, daemon=True).start()

    def _hide_all_robots(self) -> None:
        for h in self.robots.values():
            h.urdf.show_visual = False

    def _advance_frame(self) -> int:
        """Advance the Frame slider by one (wrapping); returns the new frame.

        The programmatic slider write is guarded so its on_update does not pause
        playback; the on_update still redraws the new frame.
        """
        frame = (int(self._slider.value) + 1) % self._n_frames
        self._prog_update = True
        self._slider.value = frame
        self._prog_update = False
        return frame

    def _player_loop(self) -> None:
        """Background playback: step frames at the chosen FPS while playing."""
        if self._n_frames <= 1:
            return
        while True:
            if self._playing:
                self._advance_frame()
                time.sleep(1.0 / max(1, int(self._fps_in.value)))
            else:
                time.sleep(0.02)

    def _original_frame(self, frame: int) -> np.ndarray:
        return self.original_joints[frame].astype(np.float32)

    def _draw_skeleton(self, prefix: str, pos: np.ndarray, *, ghost: bool,
                       bones: list[tuple[int, int]] | None = None) -> None:
        """Source skeleton: body/finger bones + joints, toggle-gated. ``bones`` None
        is the full 52-joint SMPLH layout (with fingers); pass an explicit bone list
        (e.g. SMPLX_BODY_BONES) for a finger-less source, drawn over all its joints."""
        body_col = skeleton.COLOR_GHOST_BODY if ghost else skeleton.COLOR_BODY
        finger_col = skeleton.COLOR_GHOST_FINGER if ghost else skeleton.COLOR_FINGER
        lw = 1.5 if ghost else 3.5
        body_bones = skeleton.BODY_BONES if bones is None else bones
        has_fingers = bones is None

        segs, seg_cols = [], []
        if self._tog_body_bones.value:
            segs += [[pos[a], pos[b]] for a, b in body_bones]
            seg_cols += [body_col] * len(body_bones)
        if has_fingers and self._tog_finger_bones.value:
            segs += [[pos[a], pos[b]] for a, b in skeleton.FINGER_BONES]
            seg_cols += [finger_col] * len(skeleton.FINGER_BONES)
        if segs:
            arr = np.asarray(segs, dtype=np.float32)
            cols = np.repeat(np.asarray(seg_cols, np.uint8)[:, None, :], 2, axis=1)
            h = self.server.scene.add_line_segments(
                f"{prefix}/bones", arr, cols, line_width=lw)
            self._dynamic_handles.append(h)

        j_idx, j_cols = [], []
        if self._tog_body_joints.value:
            body_joints = (skeleton.BODY_JOINT_INDICES if has_fingers
                           else list(range(pos.shape[0])))
            j_idx += body_joints
            j_cols += [body_col] * len(body_joints)
        if has_fingers and self._tog_finger_joints.value:
            j_idx += skeleton.FINGER_JOINT_INDICES
            j_cols += [finger_col] * len(skeleton.FINGER_JOINT_INDICES)
        if j_idx:
            h = self.server.scene.add_point_cloud(
                f"{prefix}/joints", pos[j_idx].astype(np.float32),
                np.asarray(j_cols, np.uint8), point_size=0.025)
            self._dynamic_handles.append(h)

    def _active_human_pelvis(self, frame: int) -> np.ndarray:
        """Root the SMPL-X mesh follows: the active stage's pelvis when that stage is a full
        52-joint human skeleton (Original -> raw/in the air, Grounded -> lowered, Scaled),
        else the raw pelvis. Stages differ from Original by a rigid offset, so the mesh
        tracks the displayed skeleton by translating its root onto this pelvis."""
        methods = getattr(self, "_methods", None)
        if methods is not None:
            method = methods.get(self._method_dd.value)
            pos = method.stages.get(self._stage_dd.value) if method is not None else None
            if pos is not None and pos.ndim == 3 and pos.shape[1] == 52:
                return pos[frame, 0]
        return self.original_joints[frame, 0]

    # Stages where the full human is shown at human scale, so the SMPL-X mesh aligns.
    SMPLX_STAGES = ("Original", "Grounded")

    def _draw_smplx_mesh(self, frame: int) -> None:
        """Render the SMPL-X body mesh for the given frame; no-op when data absent. Shown
        only on the human-scale stages (Original, Grounded), where it follows the skeleton
        by snapping its pelvis onto the active stage's root (see _active_human_pelvis);
        hidden elsewhere, where a full-size human mesh would not match the stage."""
        show = (self._tog_smplx.value and self.human_body is not None
                and self.original_quats is not None and self.original_joints is not None
                and self._stage_dd.value in self.SMPLX_STAGES)
        if not show:
            if self._smplx_handle is not None:
                self._smplx_handle.visible = False
            return
        raw_pelvis = self.original_joints[frame, 0]
        # placed_verts caches by frame; a stage only shifts the mesh rigidly, so pose once
        # at the raw pelvis (cache hit on toggles) then translate onto the stage's root.
        base = self.human_body.placed_verts(self.original_quats[frame], raw_pelvis, frame_idx=frame)
        verts = (base + (self._active_human_pelvis(frame) - raw_pelvis)).astype(np.float32)
        if self._smplx_handle is None:
            self._smplx_handle = self.server.scene.add_mesh_simple(
                "/human/mesh", vertices=verts, faces=self.human_body.faces,
                color=(150, 150, 150), opacity=0.7)
        else:
            self._smplx_handle.vertices = verts
            self._smplx_handle.visible = True

    def _solved_or_ref_object_pose(self, frame: int, stage: str):
        """Object (quat_wxyz, trans, ok) honouring the 'Solved object pose' toggle:
        the SOLVED pose when on (order [qw,qx,qy,qz, x,y,z], quat at [:4]), else the
        stage REFERENCE pose (order [x,y,z, qw,qx,qy,qz], quat at [3:7]) — the same pose
        the box is drawn at for this stage. ok=False when neither is available."""
        method = self._methods.get(self._method_dd.value) if hasattr(self, "_methods") else None
        sp = method.solved_object_poses if method is not None else None
        if self._tog_solved_obj.value and sp is not None and frame < len(sp):
            return sp[frame][:4], sp[frame][4:7], True
        pose = self._object_pose(stage)
        if pose is not None:
            return pose[frame, 3:7], pose[frame, :3], True
        return None, None, False

    def _object_pose(self, stage: str):
        """Object pose for the given stage: the active method's placed pose on a scaled
        stage (GMR centres it by smpl_scale, TEST keeps it raw), else the raw pose. The
        object keeps its native size in both; only the placement changes."""
        if stage in self.object_scaled_stages:
            methods = getattr(self, "_methods", None)
            if methods is not None:
                method = methods.get(self._method_dd.value)
                if method is not None and method.object_pose_scaled is not None:
                    return method.object_pose_scaled
            return self.object_pose_scaled
        return self.object_pose_raw

    def _draw_object(self, frame: int) -> None:
        """Native-size mesh + surface samples of the object for the given frame, placed
        with the active stage's pose (centred on a scaled stage, raw otherwise). The mesh
        ("Object mesh") and points ("Object points") are independently toggled.

        Both use persistent handles updated in place (not the per-frame add/remove of
        _dynamic_handles) so the object does not flicker during playback."""
        pose = None if self.object_mesh_verts is None else self._object_pose(self._stage_dd.value)

        if pose is not None and self._tog_object.value:
            verts = transform_points_local_to_world(
                pose[frame, 3:7], pose[frame, :3], self.object_mesh_verts).astype(np.float32)
            if self._object_mesh_handle is None:
                self._object_mesh_handle = self.server.scene.add_mesh_simple(
                    "/object/mesh", vertices=verts, faces=self.object_mesh_faces,
                    color=(180, 180, 190), opacity=0.6)
            else:
                self._object_mesh_handle.vertices = verts
                self._object_mesh_handle.visible = True
        elif self._object_mesh_handle is not None:
            self._object_mesh_handle.visible = False

        if pose is not None and self._tog_object_pts.value and self.object_points_local is not None:
            pts = transform_points_local_to_world(
                pose[frame, 3:7], pose[frame, :3], self.object_points_local).astype(np.float32)
            if self._object_pts_handle is None:
                col = np.broadcast_to((255, 140, 0), (len(pts), 3)).astype(np.uint8)
                self._object_pts_handle = self.server.scene.add_point_cloud(
                    "/object/pts", pts, col, point_size=0.012)
            else:
                self._object_pts_handle.points = pts
                self._object_pts_handle.visible = True
        elif self._object_pts_handle is not None:
            self._object_pts_handle.visible = False

    def _draw_diagnostics(self, frame: int) -> None:
        """TEST-SOCP solve diagnostics. (1) When "Solved object pose" is on, re-place
        the object mesh/points at TEST's SOLVED object pose (movable/inertia), overriding
        the reference pose drawn by _draw_object so the solved-vs-reference gap is visible.
        (2) Report the per-frame CoM height, foot slip and |L| in the "Solve state" text.
        No-op for methods without diagnostics (the fields stay None)."""
        method = self._methods.get(self._method_dd.value)
        if method is None:
            return
        # (1) Solved object pose: the solved pose is [qw,qx,qy,qz, x,y,z]; the mesh/points
        # handles were created (or hidden) by _draw_object for the active stage/pose.
        sp_seq = method.solved_object_poses
        if (self._tog_solved_obj.value and sp_seq is not None and frame < len(sp_seq)
                and self.object_mesh_verts is not None):
            quat, trans = sp_seq[frame][:4], sp_seq[frame][4:7]
            if self._object_mesh_handle is not None and self._tog_object.value:
                self._object_mesh_handle.vertices = transform_points_local_to_world(
                    quat, trans, self.object_mesh_verts).astype(np.float32)
                self._object_mesh_handle.visible = True
            if (self._object_pts_handle is not None and self._tog_object_pts.value
                    and self.object_points_local is not None):
                self._object_pts_handle.points = transform_points_local_to_world(
                    quat, trans, self.object_points_local).astype(np.float32)
                self._object_pts_handle.visible = True
        # (2) Solve-state readout: CoM height (solved vs grounded target), foot slip
        # (mm), and |L| (solved vs reference). Targets shown as "(→tgt)" when present.
        parts = []
        if method.com is not None and frame < len(method.com):
            txt = f"CoM z={method.com[frame][2]:.3f}m"
            if method.com_ref is not None and frame < len(method.com_ref):
                txt += f" (→{method.com_ref[frame][2]:.3f})"
            parts.append(txt)
        if method.foot_slip is not None and frame < len(method.foot_slip):
            parts.append(f"slip={method.foot_slip[frame] * 1000:.1f}mm")
        if method.angular_momentum is not None and frame < len(method.angular_momentum):
            txt = f"|L|={np.linalg.norm(method.angular_momentum[frame]):.2f}"
            if method.angular_momentum_ref is not None and frame < len(method.angular_momentum_ref):
                txt += f" (→{np.linalg.norm(method.angular_momentum_ref[frame]):.2f})"
            parts.append(txt)
        self._diag_text.value = "  ".join(parts) if parts else "(no Test diagnostics)"

    def _draw_centroidal(self, frame: int) -> None:
        """Make TEST's centroidal quantities geometric (they are otherwise only numbers).
        "CoM + trail": a magenta SOLVED CoM marker + grey shadow + ballistic-arc polyline,
        and the grounded CoM TARGET (W^c_pos reference) drawn alongside in green so the
        solved-vs-target gap is visible. "Angular momentum L": a cyan arrow along the solved
        L plus an orange arrow along the reference L (W^L target). The two L arrows are each
        normalised to their OWN clip peak (~0.5 m): the reference is a lumped/orbital quantity
        whose absolute scale differs from the full centroidal L, so only direction/trend is
        comparable. All gate on the corresponding fields; markers/shadows are persistent (no
        flicker), trails/arrows are dynamic."""
        method = self._methods.get(self._method_dd.value)
        com = None if method is None else method.com
        com_ref = None if method is None else method.com_ref
        show_com = self._tog_com.value and com is not None and frame < len(com)
        # Persistent CoM marker + ground shadow, updated in place. Magenta = solved.
        if show_com:
            c = np.asarray(com[frame], dtype=np.float32)
            shadow = np.array([[c[0], c[1], 0.0]], dtype=np.float32)
            if self._com_handle is None:
                self._com_handle = self.server.scene.add_point_cloud(
                    "/test/com", c[None], np.array([[230, 0, 230]], np.uint8), point_size=0.05)
                self._com_shadow_handle = self.server.scene.add_point_cloud(
                    "/test/com_shadow", shadow, np.array([[90, 90, 90]], np.uint8), point_size=0.04)
            else:
                self._com_handle.points = c[None]
                self._com_handle.visible = True
                self._com_shadow_handle.points = shadow
                self._com_shadow_handle.visible = True
            # Whole-clip CoM trail (ballistic arc) as a dynamic polyline.
            if len(com) > 1:
                tr = np.asarray(com, dtype=np.float32)
                segs = np.stack([tr[:-1], tr[1:]], axis=1)
                cols = np.broadcast_to((230, 0, 230), (len(segs), 2, 3)).astype(np.uint8)
                self._dynamic_handles.append(self.server.scene.add_line_segments(
                    "/test/com_trail", segs, cols, line_width=2.0))
        elif self._com_handle is not None:
            self._com_handle.visible = False
            self._com_shadow_handle.visible = False
        # Grounded CoM TARGET marker + shadow + trail drawn alongside in green.
        show_ref = self._tog_com.value and com_ref is not None and frame < len(com_ref)
        if show_ref:
            cr = np.asarray(com_ref[frame], dtype=np.float32)
            shr = np.array([[cr[0], cr[1], 0.0]], dtype=np.float32)
            if self._com_ref_handle is None:
                self._com_ref_handle = self.server.scene.add_point_cloud(
                    "/test/com_ref", cr[None], np.array([[40, 200, 90]], np.uint8), point_size=0.05)
                self._com_ref_shadow_handle = self.server.scene.add_point_cloud(
                    "/test/com_ref_shadow", shr, np.array([[60, 130, 80]], np.uint8), point_size=0.04)
            else:
                self._com_ref_handle.points = cr[None]
                self._com_ref_handle.visible = True
                self._com_ref_shadow_handle.points = shr
                self._com_ref_shadow_handle.visible = True
            if len(com_ref) > 1:
                trr = np.asarray(com_ref, dtype=np.float32)
                segsr = np.stack([trr[:-1], trr[1:]], axis=1)
                colsr = np.broadcast_to((40, 200, 90), (len(segsr), 2, 3)).astype(np.uint8)
                self._dynamic_handles.append(self.server.scene.add_line_segments(
                    "/test/com_ref_trail", segsr, colsr, line_width=2.0))
        elif self._com_ref_handle is not None:
            self._com_ref_handle.visible = False
            self._com_ref_shadow_handle.visible = False
        # Angular-momentum arrows from the CoM: cyan = solved L, orange = reference L.
        # Anchor both at the solved CoM when present, else the target CoM.
        anchor = com if com is not None else com_ref
        if self._tog_L.value and anchor is not None and frame < len(anchor):
            c = np.asarray(anchor[frame], dtype=np.float32)
            L = None if method is None else method.angular_momentum
            if L is not None and frame < len(L):
                peak = float(np.linalg.norm(np.asarray(L), axis=1).max())
                k = 0.5 / peak if peak > 1e-9 else 0.0
                tip = (c + k * np.asarray(L[frame], dtype=np.float32)).astype(np.float32)
                seg = np.stack([c, tip])[None]
                cols = np.broadcast_to((0, 180, 230), (1, 2, 3)).astype(np.uint8)
                self._dynamic_handles.append(self.server.scene.add_line_segments(
                    "/test/L_arrow", seg, cols, line_width=4.0))
                self._dynamic_handles.append(self.server.scene.add_point_cloud(
                    "/test/L_tip", tip[None], np.array([[0, 180, 230]], np.uint8), point_size=0.04))
            L_ref = None if method is None else method.angular_momentum_ref
            if L_ref is not None and frame < len(L_ref):
                peak_r = float(np.linalg.norm(np.asarray(L_ref), axis=1).max())
                k_r = 0.5 / peak_r if peak_r > 1e-9 else 0.0
                tip_r = (c + k_r * np.asarray(L_ref[frame], dtype=np.float32)).astype(np.float32)
                seg_r = np.stack([c, tip_r])[None]
                cols_r = np.broadcast_to((255, 150, 0), (1, 2, 3)).astype(np.uint8)
                self._dynamic_handles.append(self.server.scene.add_line_segments(
                    "/test/L_ref_arrow", seg_r, cols_r, line_width=4.0))
                self._dynamic_handles.append(self.server.scene.add_point_cloud(
                    "/test/L_ref_tip", tip_r[None], np.array([[255, 150, 0]], np.uint8), point_size=0.04))

    def _draw_g1_points(self, frame: int) -> None:
        """Holosoma's G1 morphological-graph keypoints (the robot links the interaction
        mesh pairs with the object points) at the active method's solved pose. Shown only
        on the Robot stage (the solved robot), hidden elsewhere. Persistent handle, updated
        in place like the object points so it does not flicker."""
        method = self._methods.get(self._method_dd.value)
        g1 = None if method is None else method.g1_points
        if self._tog_g1_pts.value and g1 is not None and self._stage_dd.value == ROBOT_STAGE:
            pts = g1[frame].astype(np.float32)
            if self._g1_pts_handle is None:
                col = np.broadcast_to((0, 200, 0), (len(pts), 3)).astype(np.uint8)
                self._g1_pts_handle = self.server.scene.add_point_cloud(
                    "/holosoma/g1_pts", pts, col, point_size=0.02)
            else:
                self._g1_pts_handle.points = pts
                self._g1_pts_handle.visible = True
        elif self._g1_pts_handle is not None:
            self._g1_pts_handle.visible = False

    def _draw_sdf(self, frame: int) -> None:
        """Object SDF near-surface band shell, coloured by signed distance, placed at the
        active stage's object pose (object-local points lifted to world). Persistent handle,
        updated in place; no-op when no SDF / object pose is available."""
        pose = None if self.object_sdf_pts is None else self._object_pose(self._stage_dd.value)
        if self._tog_sdf.value and pose is not None:
            pts = transform_points_local_to_world(
                pose[frame, 3:7], pose[frame, :3], self.object_sdf_pts).astype(np.float32)
            if self._sdf_handle is None:
                self._sdf_handle = self.server.scene.add_point_cloud(
                    "/interaction/sdf", pts, self.object_sdf_cols, point_size=0.008)
            else:
                self._sdf_handle.points = pts
                self._sdf_handle.visible = True
        elif self._sdf_handle is not None:
            self._sdf_handle.visible = False

    def _draw_signed_cloud(self, handle_attr, name, pts, dist, show, margin=CONTACT_MARGIN_M):
        """Persistent signed-distance point cloud, updated in place; hidden when show=False.
        `margin` is the contact band used for the colour scale (the run's L per channel)."""
        h = getattr(self, handle_attr)
        if not show:
            if h is not None:
                h.visible = False
            return
        cols = signed_distance_colors(dist, margin)
        if h is None:
            h = self.server.scene.add_point_cloud(
                name, points=pts.astype(np.float32), colors=cols, point_size=0.01)
            setattr(self, handle_attr, h)
        else:
            h.points = pts.astype(np.float32)
            h.colors = cols
            h.visible = True

    def _draw_interaction(self, frame: int) -> None:
        """TEST-SOCP overlays: human contact (Grounded stage) + G1 transport (Robot stage)."""
        method = self._methods.get(self._method_dd.value)
        stage = self._stage_dd.value
        _Lmix = max(self._L_flr, self._L_obj)   # combined min(obj,flr) clouds: widest band
        show_human = (method is not None and self._tog_human.value
                      and method.human_probe_pts is not None and stage == "Grounded")
        self._draw_signed_cloud(
            "_human_handle", "/test/human",
            method.human_probe_pts[frame] if show_human else None,
            method.human_dist[frame] if show_human else None, show_human, margin=_Lmix)
        show_g1 = (method is not None and self._tog_g1_transport.value
                   and method.g1_transport_pts is not None and stage == ROBOT_STAGE)
        self._draw_signed_cloud(
            "_g1_transport_handle", "/test/g1_transport",
            method.g1_transport_pts[frame] if show_g1 else None,
            method.g1_dist[frame] if show_g1 else None, show_g1, margin=_Lmix)
        # Object / floor "contact" footprints (SDF mode, like test_pipe's witness_cloud):
        # the witness points of the human channels for the ACTIVE probes — the contact
        # footprint on the object / floor. Gated to the Grounded stage (they belong with
        # the Grounded human probes the witnesses came from).
        pose = self._object_pose(stage) if method is not None else None
        # Object footprint: human_object witness (object-local), lifted by the object pose.
        show_o = (method is not None and self._tog_object_contact.value and stage == "Grounded"
                  and method.human_witness is not None and method.human_obj_dist is not None
                  and pose is not None)
        if show_o:
            d = method.human_obj_dist[frame]
            a = d < self._L_obj
            wit = transform_points_local_to_world(
                pose[frame, 3:7], pose[frame, :3], method.human_witness[frame][a])
            self._draw_signed_cloud("_object_contact_handle", "/test/object_contact",
                                    wit, d[a], bool(a.any()), margin=self._L_obj)
        else:
            self._draw_signed_cloud("_object_contact_handle", "/test/object_contact",
                                    None, None, False)
        # Floor footprint: the active probes projected onto z=0 (world).
        show_f = (method is not None and self._tog_floor_contact.value and stage == "Grounded"
                  and method.human_probe_pts is not None and method.human_flr_dist is not None)
        if show_f:
            d = method.human_flr_dist[frame]
            a = d < self._L_flr
            fw = method.human_probe_pts[frame][a].copy()
            fw[:, 2] = 0.0
            self._draw_signed_cloud("_floor_contact_handle", "/test/floor_contact",
                                    fw, d[a], bool(a.any()), margin=self._L_flr)
        else:
            self._draw_signed_cloud("_floor_contact_handle", "/test/floor_contact",
                                    None, None, False)

    def _draw_segments(self, name, a, b, dist, margin=CONTACT_MARGIN_M):
        segs = np.stack([a, b], axis=1).astype(np.float32)            # (K, 2, 3)
        cols = np.repeat(signed_distance_colors(dist, margin)[:, None, :], 2, axis=1)
        h = self.server.scene.add_line_segments(name, segs, cols, line_width=1.5)
        self._dynamic_handles.append(h)

    def _draw_directions(self, frame: int) -> None:
        """Probe -> witness lines for the ACTIVE probes, split into two independent
        channels. Object directions (human hands / G1) lift an object-local witness by
        the object pose; floor directions use the world-frame floor witness directly
        (probe projected to z=0), so feet point along the ground normal, not at the object."""
        show_obj = self._tog_dir_object.value
        show_flr = self._tog_dir_floor.value
        if not (show_obj or show_flr):
            return
        method = self._methods.get(self._method_dd.value)
        if method is None:
            return
        stage = self._stage_dd.value
        # Object pose used to lift the object-local witnesses, honouring "Solved object
        # pose" (solved when on, else the stage reference pose) — see helper.
        obj_quat, obj_trans, has_obj_pose = self._solved_or_ref_object_pose(frame, stage)
        if stage == "Grounded" and method.human_probe_pts is not None:
            probes = method.human_probe_pts[frame]
            # Object channel: object-local witness, lifted by the object pose.
            if (show_obj and has_obj_pose and method.human_witness is not None
                    and method.human_obj_dist is not None):
                d_obj = method.human_obj_dist[frame]
                a_obj = d_obj < self._L_obj
                if a_obj.any():
                    wit = transform_points_local_to_world(
                        obj_quat, obj_trans, method.human_witness[frame][a_obj])
                    self._draw_segments("/test/dir_human_obj", probes[a_obj], wit,
                                        d_obj[a_obj], margin=self._L_obj)
            # Floor channel: world-frame witness, NOT lifted by the object pose.
            if (show_flr and method.human_flr_witness is not None
                    and method.human_flr_dist is not None):
                d_flr = method.human_flr_dist[frame]
                a_flr = d_flr < self._L_flr
                if a_flr.any():
                    self._draw_segments("/test/dir_human_flr", probes[a_flr],
                                        method.human_flr_witness[frame][a_flr], d_flr[a_flr],
                                        margin=self._L_flr)
        # G1 transport is a correspondence onto the solved robot; it has both an object
        # and a floor channel, each gated on its own toggle (mirrors the human probes).
        if stage == ROBOT_STAGE and method.g1_transport_pts is not None:
            g1 = method.g1_transport_pts[frame]
            # Object channel: object-local witness, lifted by the object pose (solved
            # when "Solved object pose" is on, else reference — see obj_quat/obj_trans above).
            if (show_obj and has_obj_pose and method.g1_obj_witness is not None
                    and method.g1_obj_dist is not None):
                d = method.g1_obj_dist[frame]
                a = d < self._L_obj
                if a.any():
                    wit = transform_points_local_to_world(
                        obj_quat, obj_trans, method.g1_obj_witness[frame][a])
                    self._draw_segments("/test/dir_g1_obj", g1[a], wit, d[a], margin=self._L_obj)
            # Floor channel: world-frame witness, NOT lifted by the object pose.
            if (show_flr and method.g1_flr_witness is not None
                    and method.g1_flr_dist is not None):
                d = method.g1_flr_dist[frame]
                a = d < self._L_flr
                if a.any():
                    self._draw_segments("/test/dir_g1_flr", g1[a],
                                        method.g1_flr_witness[frame][a], d[a], margin=self._L_flr)

    def _draw_object_floor(self, frame: int) -> None:
        """Object-as-carrier <-> floor channel (mirrors the human/G1 floor channel for the
        object). The static object-local surface samples are lifted by the object pose —
        SOLVED or stage-reference per 'Solved object pose', exactly like the box and the
        robot/box directions — then split into three independently-toggled overlays:
        all surface points (height-coloured), near-floor probe->witness directions, and the
        near-floor contact footprint on the floor (witness = the point projected to z=0)."""
        method = self._methods.get(self._method_dd.value)
        surf = method.object_surface_local if method is not None else None
        show_pts = self._tog_obj_surface.value
        show_dir = self._tog_dir_obj_floor.value
        show_con = self._tog_obj_floor_contact.value
        quat, trans, ok = (self._solved_or_ref_object_pose(frame, self._stage_dd.value)
                           if surf is not None else (None, None, False))
        if surf is None or not ok or not (show_pts or show_dir or show_con):
            self._draw_signed_cloud("_object_surface_handle", "/test/object_surface", None, None, False)
            self._draw_signed_cloud("_object_floor_contact_handle", "/test/object_floor_contact",
                                    None, None, False)
            return
        world = transform_points_local_to_world(quat, trans, surf)   # (M, 3)
        d = world[:, 2]                                              # floor signed distance = height
        # Object<->floor is the FLOOR channel for the object carrier -> band = L_floor.
        _Lf = self._L_flr
        # All surface points, height-coloured (independent of the contact band).
        self._draw_signed_cloud("_object_surface_handle", "/test/object_surface",
                                world if show_pts else None, d if show_pts else None, show_pts,
                                margin=_Lf)
        near = d < _Lf
        wit = world.copy(); wit[:, 2] = 0.0                         # floor witness (project to z=0)
        if show_dir and near.any():
            self._draw_segments("/test/dir_object_floor", world[near], wit[near], d[near], margin=_Lf)
        show_con_now = show_con and bool(near.any())
        self._draw_signed_cloud("_object_floor_contact_handle", "/test/object_floor_contact",
                                wit[near] if show_con_now else None,
                                d[near] if show_con_now else None, show_con_now, margin=_Lf)

    def _draw_sdf_floor(self) -> None:
        """Static analytic floor SDF band (world), toggled visible/invisible."""
        show = (self._tog_sdf_floor.value and self.object_sdf_floor_pts is not None)
        if not show:
            if self._sdf_floor_handle is not None:
                self._sdf_floor_handle.visible = False
            return
        if self._sdf_floor_handle is None:
            self._sdf_floor_handle = self.server.scene.add_point_cloud(
                "/test/sdf_floor", points=self.object_sdf_floor_pts.astype(np.float32),
                colors=self.object_sdf_floor_cols, point_size=0.008)
        else:
            self._sdf_floor_handle.visible = True

    def _draw_stage_skeleton(self, prefix: str, pos: np.ndarray, bones, *, ghost: bool) -> None:
        """A non-source stage (mapped bodies or solved robot links) as a skeleton:
        bones (when a topology is given) plus joints, both toggle-gated, in orange."""
        col = skeleton.COLOR_GHOST_STAGE if ghost else skeleton.COLOR_STAGE
        lw = 1.5 if ghost else 3.5
        if bones and self._tog_body_bones.value:
            segs = np.asarray([[pos[a], pos[b]] for a, b in bones], dtype=np.float32)
            cols = np.broadcast_to(col, (len(bones), 2, 3)).astype(np.uint8)
            h = self.server.scene.add_line_segments(f"{prefix}/bones", segs, cols, line_width=lw)
            self._dynamic_handles.append(h)
        if self._tog_body_joints.value:
            h = self.server.scene.add_point_cloud(
                f"{prefix}/joints", pos.astype(np.float32), col, point_size=0.025)
            self._dynamic_handles.append(h)

    def _draw_axes(self, prefix: str, pos: np.ndarray, quat: np.ndarray, *, ghost: bool) -> None:
        """Per-joint orientation frames (RGB triads) at each joint. ``quat`` is (B, 4)
        wxyz, ``pos`` is (B, 3). Length comes from the Frame size slider (ghost = half),
        radius scales with length so the axes stay slender."""
        length = float(self._axis_size.value) * (0.5 if ghost else 1.0)
        h = self.server.scene.add_batched_axes(
            f"{prefix}/axes",
            batched_wxyzs=np.asarray(quat, dtype=np.float32),
            batched_positions=np.asarray(pos, dtype=np.float32),
            axes_length=length, axes_radius=length * AXIS_RADIUS_FRAC)
        self._dynamic_handles.append(h)

    def _draw_stage(self, method, stage: str, prefix: str, frame: int, *, ghost: bool) -> None:
        """Draw a method's stage skeleton: full SMPLH topology for 52-joint stages,
        else the stage's own bone topology over its mapped joints (plus joint frames
        when the stage carries per-joint orientations)."""
        pos = method.stages[stage][frame]
        if pos.shape[0] == 52:
            self._draw_skeleton(prefix, pos.astype(np.float32), ghost=ghost)
        else:
            self._draw_stage_skeleton(prefix, pos, method.stage_bones.get(stage), ghost=ghost)
            quat = method.stage_quats.get(stage)
            if self._tog_axes.value and quat is not None:
                self._draw_axes(prefix, pos, quat[frame], ghost=ghost)

    def _clear_dynamic(self) -> None:
        for h in self._dynamic_handles:
            h.remove()
        self._dynamic_handles = []

    def _redraw(self, frame: int) -> None:
        method = self._methods[self._method_dd.value]
        stage = self._stage_dd.value
        self._clear_dynamic()
        self._hide_all_robots()
        if stage == ROBOT_STAGE:
            if self._tog_urdf.value:
                self.robots[method.robot_key].urdf.show_visual = True
                self.draw_q(method.qpos[frame], stage=method.robot_key)
            if method.robot_skeleton is not None:
                self._draw_stage_skeleton(
                    "/active", method.robot_skeleton[frame],
                    method.stage_bones.get(ROBOT_STAGE), ghost=False)
                if self._tog_axes.value and method.robot_quats is not None:
                    self._draw_axes("/active", method.robot_skeleton[frame],
                                    method.robot_quats[frame], ghost=False)
        elif stage == "Original" and self.original_joints is not None:
            self._draw_skeleton("/active", self._original_frame(frame), ghost=False,
                                bones=self.original_bones)
        else:
            self._draw_stage(method, stage, "/active", frame, ghost=False)
        self._draw_smplx_mesh(frame)
        self._draw_object(frame)
        self._draw_g1_points(frame)
        self._draw_sdf(frame)
        self._draw_interaction(frame)
        self._draw_directions(frame)
        self._draw_object_floor(frame)
        self._draw_sdf_floor()
        self._draw_diagnostics(frame)
        self._draw_centroidal(frame)
        g_stage = self._ghost_stage_dd.value
        if g_stage != "Off":
            g_method = self._methods[self._ghost_method_dd.value]
            if g_stage == "Original" and self.original_joints is not None:
                self._draw_skeleton("/ghost", self._original_frame(frame), ghost=True,
                                    bones=self.original_bones)
            elif g_stage in g_method.stages:
                self._draw_stage(g_method, g_stage, "/ghost", frame, ghost=True)

    def close(self) -> None:
        self.server.stop()
