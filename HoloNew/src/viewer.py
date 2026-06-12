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
    """
    label: str
    robot_key: str
    qpos: np.ndarray
    stages: dict = field(default_factory=dict)
    stage_bones: dict = field(default_factory=dict)
    robot_skeleton: np.ndarray | None = None


class Viewer:
    def __init__(self, robot_model_path: str, object_model_path: str | None,
                 stage_keys: tuple[str, ...] = ("socp",),
                 has_dynamic_object: bool = False,
                 original_joints: np.ndarray | None = None,
                 original_quats: np.ndarray | None = None,
                 object_mesh_verts: np.ndarray | None = None,
                 object_mesh_faces: np.ndarray | None = None,
                 object_points_local: np.ndarray | None = None,
                 object_pose_raw: np.ndarray | None = None,
                 object_pose_scaled: np.ndarray | None = None,
                 object_scaled_stages: tuple[str, ...] = (),
                 human_body: object | None = None) -> None:
        self.robot_model_path = robot_model_path
        self.object_model_path = object_model_path
        self.has_dynamic_object = has_dynamic_object
        # Original source motion, shared by every method's "Original" stage.
        self.original_joints = original_joints
        self.original_quats = original_quats
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
        self.human_body = human_body
        self._smplx_handle = None
        # Persistent object handles, updated in place each frame (never removed/recreated)
        # so the object does not flicker during playback, like the SMPL-X mesh.
        self._object_mesh_handle = None
        self._object_pts_handle = None
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

        with self.server.gui.add_folder("Meshes"):
            self._tog_smplx = self.server.gui.add_checkbox("SMPL-X mesh", False)
            self._tog_object = self.server.gui.add_checkbox("Object mesh", False)
            self._tog_object_pts = self.server.gui.add_checkbox("Object points", False)
        for _cb in (self._tog_smplx, self._tog_object, self._tog_object_pts):
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

    def _draw_skeleton(self, prefix: str, pos: np.ndarray, *, ghost: bool) -> None:
        """52-joint source skeleton: body/finger bones + joints, toggle-gated."""
        body_col = skeleton.COLOR_GHOST_BODY if ghost else skeleton.COLOR_BODY
        finger_col = skeleton.COLOR_GHOST_FINGER if ghost else skeleton.COLOR_FINGER
        lw = 1.5 if ghost else 3.5

        segs, seg_cols = [], []
        if self._tog_body_bones.value:
            segs += [[pos[a], pos[b]] for a, b in skeleton.BODY_BONES]
            seg_cols += [body_col] * len(skeleton.BODY_BONES)
        if self._tog_finger_bones.value:
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
            j_idx += skeleton.BODY_JOINT_INDICES
            j_cols += [body_col] * len(skeleton.BODY_JOINT_INDICES)
        if self._tog_finger_joints.value:
            j_idx += skeleton.FINGER_JOINT_INDICES
            j_cols += [finger_col] * len(skeleton.FINGER_JOINT_INDICES)
        if j_idx:
            h = self.server.scene.add_point_cloud(
                f"{prefix}/joints", pos[j_idx].astype(np.float32),
                np.asarray(j_cols, np.uint8), point_size=0.025)
            self._dynamic_handles.append(h)

    def _draw_smplx_mesh(self, frame: int) -> None:
        """Render the SMPL-X body mesh for the given frame; no-op when data absent."""
        show = (self._tog_smplx.value and self.human_body is not None
                and self.original_quats is not None and self.original_joints is not None)
        if not show:
            if self._smplx_handle is not None:
                self._smplx_handle.visible = False
            return
        verts = self.human_body.placed_verts(
            self.original_quats[frame], self.original_joints[frame, 0],
            frame_idx=frame).astype(np.float32)
        if self._smplx_handle is None:
            self._smplx_handle = self.server.scene.add_mesh_simple(
                "/human/mesh", vertices=verts, faces=self.human_body.faces,
                color=(150, 150, 150), opacity=0.7)
        else:
            self._smplx_handle.vertices = verts
            self._smplx_handle.visible = True

    def _object_pose(self, stage: str):
        """Object pose for the given stage: the centred pose on a scaled stage, else the
        raw pose. The object keeps its native size in both (holosoma scales the object's
        position toward the centre, not its geometry); only the placement changes."""
        if stage in self.object_scaled_stages:
            return self.object_pose_scaled
        return self.object_pose_raw

    def _draw_object(self, frame: int) -> None:
        """Native-size mesh + surface samples of the object for the given frame, placed
        with the active stage's pose (centred on a scaled stage, raw otherwise). The mesh
        ("Object mesh") and points ("Object points") are independently toggled.

        Both use persistent handles updated in place (not the per-frame add/remove of
        _dynamic_handles) so the object does not flicker during playback."""
        pose = None if self.object_mesh_verts is None else self._object_pose(self._stage_dd.value)
        from HoloNew.src.holosoma.interaction_mesh import transform_points_local_to_world

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

    def _draw_stage(self, method, stage: str, prefix: str, frame: int, *, ghost: bool) -> None:
        """Draw a method's stage skeleton: full SMPLH topology for 52-joint stages,
        else the stage's own bone topology over its mapped joints."""
        pos = method.stages[stage][frame]
        if pos.shape[0] == 52:
            self._draw_skeleton(prefix, pos.astype(np.float32), ghost=ghost)
        else:
            self._draw_stage_skeleton(prefix, pos, method.stage_bones.get(stage), ghost=ghost)

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
        elif stage == "Original" and self.original_joints is not None:
            self._draw_skeleton("/active", self._original_frame(frame), ghost=False)
        else:
            self._draw_stage(method, stage, "/active", frame, ghost=False)
        self._draw_smplx_mesh(frame)
        self._draw_object(frame)
        g_stage = self._ghost_stage_dd.value
        if g_stage != "Off":
            g_method = self._methods[self._ghost_method_dd.value]
            if g_stage == "Original" and self.original_joints is not None:
                self._draw_skeleton("/ghost", self._original_frame(frame), ghost=True)
            elif g_stage in g_method.stages:
                self._draw_stage(g_method, g_stage, "/ghost", frame, ghost=True)

    def close(self) -> None:
        self.server.stop()
