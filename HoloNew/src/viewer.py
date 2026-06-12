"""Owns the viser scene: robot instance(s), object, grid, keypoint layers.

Extracted from InteractionMeshRetargeter so rendering is separate from compute
and several trajectories can share one viser session. qpos layout matches
holosoma: [0:3] pos, [3:7] wxyz quat, [7:7+dof] actuated joints, optional
trailing [-7:] dynamic-object pose.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh
import viser
from viser.extras import ViserUrdf
import yourdfpy

from .stages import ROBOT_STAGE, method_labels, stages_for_method


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
    """
    label: str
    robot_key: str
    qpos: "np.ndarray"
    stages: dict = field(default_factory=dict)


class Viewer:
    def __init__(self, robot_model_path: str, object_model_path: str | None,
                 stage_keys: tuple[str, ...] = ("socp",),
                 has_dynamic_object: bool = False) -> None:
        self.robot_model_path = robot_model_path
        self.object_model_path = object_model_path
        self.has_dynamic_object = has_dynamic_object
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
        """Bind a list of MethodViz and build Frame + Method + Stage dropdowns."""
        self._methods = {m.label: m for m in methods}
        T = min(len(m.qpos) for m in methods)

        with self.server.gui.add_folder("Playback"):
            self._slider = self.server.gui.add_slider(
                "Frame", min=0, max=max(0, T - 1), step=1, initial_value=0)
        with self.server.gui.add_folder("Display"):
            first = methods[0].label
            self._method_dd = self.server.gui.add_dropdown(
                "Method", options=method_labels(), initial_value=first)
            self._stage_dd = self.server.gui.add_dropdown(
                "Stage", options=stages_for_method(first), initial_value=ROBOT_STAGE)

        @self._method_dd.on_update
        def _(_evt):
            self._stage_dd.options = stages_for_method(self._method_dd.value)
            self._stage_dd.value = ROBOT_STAGE
            self._redraw(int(self._slider.value))

        @self._slider.on_update
        def _(_evt): self._redraw(int(self._slider.value))

        @self._stage_dd.on_update
        def _(_evt): self._redraw(int(self._slider.value))

        self._redraw(0)

    def _hide_all_robots(self) -> None:
        for h in self.robots.values():
            h.urdf.show_visual = False

    def _redraw(self, frame: int) -> None:
        method = self._methods[self._method_dd.value]
        stage = self._stage_dd.value
        # Clear the skeleton layer each redraw.
        self._clear_skeleton()
        self._hide_all_robots()
        if stage == ROBOT_STAGE:
            self.robots[method.robot_key].urdf.show_visual = True
            self.draw_q(method.qpos[frame], stage=method.robot_key)
        else:
            pts = method.stages[stage][frame].astype(np.float32)
            self._skeleton = self.draw_keypoints(
                pts, name="stage_skeleton", rgba=(1.0, 0.4, 0.0, 1.0))

    def _clear_skeleton(self) -> None:
        # draw_keypoints batches over points, so an empty array would error;
        # remove the previously drawn skeleton handle instead.
        handle = getattr(self, "_skeleton", None)
        if handle is not None:
            handle.remove()
            self._skeleton = None

    def close(self) -> None:
        self.server.stop()
