"""FrameTrace viewer — the main ``viz`` viewer (one big viser server, many toggles).

PURE CONSUMER (golden rule 6): it drives ``prepare`` once to get the static assets, BAKES the
per-frame ``FrameTrace`` (``targets.pipeline.trace_frame``) for smooth playback, and only READS those
typed artifacts to draw togglable layers. ZERO hooks in the compute; viser is confined to this module;
one ``_draw_<layer>`` method per checkbox-gated layer (static layers added once, dynamic layers updated
per slider tick). See docs/VIZ.md.

Layers implemented here (the HUMAN side — everything available from a ``FrameTrace`` without a SOLVED
robot):
  - Playback : frame slider · play · fps
  - Static   : ground plane (z=0) · SMPL ghost mesh (per-frame) · posed object clouds
  - Skeleton : SMPL bones from ``trace.pose`` (parent -> child segments)
  - Style    : the 14 ``StyleTargets`` link targets — points (colour by ``weight_pos``) + per-link
               orientation frames (3 short xyz axes) + link-name labels, so one can SEE each link
               target sit on the skeleton (e.g. ``*_toe_link`` near the ankle)
  - Interaction (human) : the posed human cloud coloured by the SELECTED channel of ``human_field``
               (uniform / distance heatmap / active mask), with optional witness lines + normals
  - Selectors: channel dropdown · colour mode · point size

DEFERRED (robot-side, out of scope here): the transported field on the G1 correspondence points, the
SMPL<->robot correspondence lines, and the SDF iso-surface. They all need a SOLVED robot configuration
(``solve/`` will place the M correspondence points via robot FK); a ``FrameTrace`` alone cannot put
them in world. No robot URDF / robot FK is loaded here.

Run:
    fuser -k 8080/tcp   # free the port FIRST (never pkill -f this script: it self-kills)
    python -m src.viz.viewer --motion-path <smplx.npz> --model-dir <smplx_models> \
        [--dataset hodome --dataset-root <root> --port 8080 --frame-step 2 --max-frames 200]
"""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as _Rot

from ..prepare.config import PrepareConfig
from ..prepare.contracts import SceneSpec
from ..prepare.runner import prepare
from ..targets.contracts import FrameTrace
from ..targets.pipeline import trace_frame
from ._scene_args import add_scene_args, scene_from_args

# colour anchors (uint8 RGB)
_NEAR = np.array([40, 90, 255], np.float64)    # distance heatmap: near/penetrating (blue)
_FAR = np.array([255, 60, 50], np.float64)     # distance heatmap: far within margin (red)
_AXIS_COLORS = np.array([[255, 80, 80], [80, 230, 80], [80, 130, 255]], np.uint8)  # x r, y g, z b


def _quat_wxyz_to_R(quat: np.ndarray) -> np.ndarray:
    """(L, 4) wxyz quaternions -> (L, 3, 3) rotation matrices (scipy is xyzw)."""
    q = np.asarray(quat, np.float64)
    return _Rot.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()


def _heat_distance(dist: np.ndarray, margin: float) -> np.ndarray:
    """(P,) signed distance -> (P, 3) uint8. blue = near/penetrating (d<=0), red = far (d ~ margin)."""
    t = np.clip(np.asarray(dist, np.float64) / max(margin, 1e-9), 0.0, 1.0)[:, None]
    return (t * _FAR + (1.0 - t) * _NEAR).astype(np.uint8)


def _active_colors(active: np.ndarray) -> np.ndarray:
    """(P,) bool -> (P, 3) uint8. bright green where active (in contact band), dim grey elsewhere."""
    col = np.tile(np.array([70, 70, 80], np.uint8), (len(active), 1))
    col[np.asarray(active, bool)] = (90, 255, 130)
    return col


class Viewer:
    """One viser viewer over a BAKED list of ``FrameTrace``. Construction prepares the scene, bakes the
    traces and precomputes the per-frame SMPL ghost; ``run()`` builds the GUI + handles and serves."""

    def __init__(self, spec: SceneSpec, *, port: int = 8080, frame_step: int = 2,
                 max_frames: int = 200) -> None:
        self.port = port
        # --- 1) prepared (static) context: load -> calibration -> channels -> clouds -> correspondence
        self.grounded, self.ctx = prepare(spec, PrepareConfig())
        self.robot = spec.robot
        body = self.grounded.body
        if body is None:
            raise ValueError("the FrameTrace viewer needs a parametric body (SMPL params)")

        # --- 2) BAKE: precompute every shown frame's FrameTrace (smooth playback; get(i) indexes this)
        self.frames = list(range(0, self.grounded.n_frames, frame_step))[:max_frames]
        print(f"baking {len(self.frames)} frames (trace_frame) ...")
        self.traces: list[FrameTrace] = [
            trace_frame(self.grounded, self.ctx, self.robot, f) for f in self.frames]
        # SMPL ghost mesh is per-frame too (full forward) — precompute alongside the traces.
        self.faces = body.faces
        self.parents = body.parents
        self.n_bones = body.n_bones
        self.ghost_verts = [body.posed_vertices(self.grounded.smpl_params, f).astype(np.float32)
                            for f in self.frames]
        self.bone_pairs = [(int(self.parents[j]), j) for j in range(self.n_bones)
                           if self.parents[j] >= 0]

        self.channel_names = self.ctx.channel_names
        self.margin = float(self.ctx.margin)
        self.style_links = self.traces[0].targets.style.link_names
        print(f"baked: T_shown={len(self.frames)}, channels={self.channel_names}, "
              f"human points={self.traces[0].human_cloud_world.shape[0]}, "
              f"style links={len(self.style_links)}, objects={self.grounded.n_objects}")

    # frame provider (VIZ.md): get(i) -> FrameTrace, indexing the baked list (a live trace_frame on
    # the fly is the alternative the design allows; the bake is used here for smooth playback).
    def get(self, i: int) -> FrameTrace:
        return self.traces[int(i)]

    # ---------------------------------------------------------------- layers (one _draw_ per toggle)
    def _draw_ground(self) -> None:
        """STATIC ground plane (z=0). Added once in ``run``; here we only toggle visibility."""
        self._ground.visible = self.cb_ground.value

    def _draw_ghost(self, trace: FrameTrace, i: int) -> None:
        """SMPL ghost mesh — per-frame (re-added like scene.py), low opacity so it stays a backdrop."""
        if self.cb_ghost.value:
            self.srv.scene.add_mesh_simple("/ghost", self.ghost_verts[i], self.faces,
                                           color=(200, 200, 210), opacity=0.45, side="double")
        else:
            self.srv.scene.add_mesh_simple("/ghost", np.zeros((3, 3), np.float32),
                                           np.array([[0, 1, 2]]), opacity=0.0)

    def _draw_objects(self, trace: FrameTrace) -> None:
        """Posed object clouds (``trace.object_clouds_world``), coloured by their OWN env field
        (``trace.targets.env_interaction`` — object<->ground / object<->object) on the SELECTED channel,
        so the object's contact is visible like the human side (pick channel ``ground`` to see the
        object resting on the floor). ``uniform`` keeps them plain orange."""
        c = self._channel_idx()
        mode = self.color_mode.value
        env = trace.targets.env_interaction.per_object
        for k, h in enumerate(self._obj_handles):
            pts = trace.object_clouds_world[k].astype(np.float32)
            if mode == "distance":
                col = _heat_distance(env[k].distance[c], self.margin)
            elif mode == "active":
                col = _active_colors(env[k].active[c])
            else:                                                        # uniform
                col = np.tile(np.array([255, 140, 0], np.uint8), (pts.shape[0], 1))
            h.points = pts
            h.colors = col
            h.point_size = float(self.size.value)
            h.visible = self.cb_objects.value

    def _draw_skeleton(self, trace: FrameTrace) -> None:
        """SMPL bones from ``trace.pose.bone_pos`` — parent->child line segments (see scene.py)."""
        if self.cb_skel.value and self.bone_pairs:
            bp = np.asarray(trace.pose.bone_pos, np.float32)
            seg = np.stack([np.stack([bp[a], bp[b]]) for a, b in self.bone_pairs]).astype(np.float32)
            col = np.tile([[[0, 120, 255]]], (len(self.bone_pairs), 2, 1)).astype(np.uint8)
            self.srv.scene.add_line_segments("/skeleton", seg, col, line_width=3.0)
        else:
            self.srv.scene.add_line_segments("/skeleton", np.zeros((1, 2, 3), np.float32),
                                             np.zeros((1, 2, 3), np.uint8), line_width=0.1)

    def _draw_style(self, trace: FrameTrace) -> None:
        """The 14 ``StyleTargets`` link targets — the KEY validation layer.

        Points sit at ``style.position`` (coloured by ``weight_pos``: orange = planted/high weight,
        cyan = position-free); per-link orientation frames draw 3 short xyz axes from ``style.orientation``
        (wxyz); per-link labels name the link. With the skeleton on, this shows each link target landing
        on the body — e.g. ``left_toe_link`` (mapped to the SMPL ANKLE + a toe pos-offset) sits at the
        ankle, the foot->ankle mapping made visible."""
        style = trace.targets.style
        pos = np.asarray(style.position, np.float32)                      # (L, 3)
        wp = np.asarray(style.weight_pos, np.float64)
        # colour by weight_pos: high (planted) -> orange, low (free) -> cyan.
        t = (wp / max(float(wp.max()), 1e-9))[:, None]
        col = (t * np.array([255, 170, 0]) + (1.0 - t) * np.array([0, 200, 220])).astype(np.uint8)
        self._style_pts.points = pos
        self._style_pts.colors = col
        self._style_pts.point_size = max(float(self.size.value) * 2.0, 0.02)
        self._style_pts.visible = self.cb_style.value

        # orientation frames (3 short axes per link).
        if self.cb_style_frames.value and style.orientation is not None:
            rots = _quat_wxyz_to_R(style.orientation)                    # (L, 3, 3)
            axis_len = 0.08
            segs, cols = [], []
            for i in range(len(self.style_links)):
                for a in range(3):
                    d = rots[i][:, a]                                    # world dir of body axis a
                    segs.append([pos[i], pos[i] + d * axis_len])
                    cols.append([_AXIS_COLORS[a], _AXIS_COLORS[a]])
            self.srv.scene.add_line_segments("/style_frames", np.asarray(segs, np.float32),
                                             np.asarray(cols, np.uint8), line_width=2.5)
        else:
            self.srv.scene.add_line_segments("/style_frames", np.zeros((1, 2, 3), np.float32),
                                             np.zeros((1, 2, 3), np.uint8), line_width=0.1)
        # link-name labels (persistent handles, repositioned per frame).
        for i, h in enumerate(self._style_labels):
            h.position = tuple(float(v) for v in pos[i])
            h.visible = self.cb_style_labels.value

    def _draw_human(self, trace: FrameTrace) -> None:
        """Posed human cloud ``trace.human_cloud_world`` coloured by the SELECTED channel of
        ``trace.human_field`` (uniform / distance heatmap / active mask)."""
        pc = trace.human_cloud_world.astype(np.float32)                  # (N, 3) world
        c = self._channel_idx()
        field = trace.human_field
        mode = self.color_mode.value
        if mode == "distance":
            col = _heat_distance(field.distance[c], self.margin)
        elif mode == "active":
            col = _active_colors(field.active[c])
        else:                                                            # uniform
            col = np.tile(np.array([185, 185, 195], np.uint8), (pc.shape[0], 1))
        self._human.points = pc
        self._human.colors = col
        self._human.point_size = float(self.size.value)
        self._human.visible = self.cb_human.value

    def _draw_field_lines(self, trace: FrameTrace) -> None:
        """Witness lines (point -> nearest surface) and normals (short segment along ``direction``) for
        the ACTIVE probes of the selected channel. The OBJECT channels store witness/direction in the
        object-LOCAL frame (the field's per-channel natural frame), so map them to world by that
        object's per-frame ``(R, t)``; the GROUND channel is already world."""
        c = self._channel_idx()
        field = trace.human_field
        ch = self.ctx.channels[c]
        active = np.asarray(field.active[c], bool)
        idx = np.where(active)[0]
        want_w, want_n = self.cb_witness.value, self.cb_normals.value
        if len(idx) and (want_w or want_n):
            if len(idx) > 400:                                           # bound the segment count
                idx = np.random.default_rng(0).choice(idx, 400, replace=False)
            pts = np.asarray(trace.human_cloud_world, np.float64)[idx]   # (S, 3) world
            wit = np.asarray(field.witness[c], np.float64)[idx]          # (S, 3) channel-local
            dirn = np.asarray(field.direction[c], np.float64)[idx]       # (S, 3) channel-local
            if ch.object_idx is not None:                               # object-local -> world
                R = np.asarray(trace.pose.object_rot[ch.object_idx], np.float64)
                tt = np.asarray(trace.pose.object_pos[ch.object_idx], np.float64)
                wit = wit @ R.T + tt
                dirn = dirn @ R.T
        else:
            pts = wit = dirn = np.zeros((0, 3))

        if want_w and len(pts):
            seg = np.stack([pts, wit], axis=1).astype(np.float32)        # (S, 2, 3)
            col = np.tile([[[230, 230, 60]]], (len(pts), 2, 1)).astype(np.uint8)
            self.srv.scene.add_line_segments("/witness", seg, col, line_width=1.5)
        else:
            self.srv.scene.add_line_segments("/witness", np.zeros((1, 2, 3), np.float32),
                                             np.zeros((1, 2, 3), np.uint8), line_width=0.1)
        if want_n and len(pts):
            seg = np.stack([pts, pts + dirn * 0.05], axis=1).astype(np.float32)
            col = np.tile([[[60, 220, 200]]], (len(pts), 2, 1)).astype(np.uint8)
            self.srv.scene.add_line_segments("/normals", seg, col, line_width=2.0)
        else:
            self.srv.scene.add_line_segments("/normals", np.zeros((1, 2, 3), np.float32),
                                             np.zeros((1, 2, 3), np.uint8), line_width=0.1)

    # ---------------------------------------------------------------- helpers
    def _channel_idx(self) -> int:
        return self.channel_names.index(self.channel.value)

    def _update_info(self, trace: FrameTrace, i: int) -> None:
        c = self._channel_idx()
        n_active = int(np.asarray(trace.human_field.active[c]).sum())
        self.info.content = (
            f"**frame {self.frames[i]}** ({i + 1}/{len(self.frames)})\n\n"
            f"channel **{self.channel.value}** · colour **{self.color_mode.value}** · "
            f"active probes **{n_active}** / {trace.human_field.n_points}\n\n"
            f"style points: orange = planted (high w_p), cyan = position-free · "
            f"distance heatmap: blue near/penetrating .. red far (margin {self.margin:.3f} m)\n\n"
            f"deferred (need a solved robot): transported G1 field · SMPL<->robot lines · SDF iso-surface")

    # ---------------------------------------------------------------- server
    def run(self) -> None:
        import viser

        self.srv = viser.ViserServer(port=self.port)
        self.srv.scene.add_grid("/grid", width=4.0, height=4.0)

        with self.srv.gui.add_folder("Playback"):
            self.sld = self.srv.gui.add_slider("frame", 0, len(self.frames) - 1, 1, 0)
            self.play = self.srv.gui.add_checkbox("play", False)
            self.fps = self.srv.gui.add_number("fps", 20, min=1, max=120, step=1)
        with self.srv.gui.add_folder("Static"):
            self.cb_ground = self.srv.gui.add_checkbox("ground plane", True)
            self.cb_ghost = self.srv.gui.add_checkbox("SMPL ghost", True)
            self.cb_objects = self.srv.gui.add_checkbox("object clouds", True)
        with self.srv.gui.add_folder("Skeleton"):
            self.cb_skel = self.srv.gui.add_checkbox("skeleton", True)
        with self.srv.gui.add_folder("Style targets"):
            self.cb_style = self.srv.gui.add_checkbox("link points", True)
            self.cb_style_frames = self.srv.gui.add_checkbox("orientation frames", True)
            self.cb_style_labels = self.srv.gui.add_checkbox("link labels", False)
        with self.srv.gui.add_folder("Interaction - human"):
            self.cb_human = self.srv.gui.add_checkbox("human cloud", True)
            self.cb_witness = self.srv.gui.add_checkbox("witness lines", False)
            self.cb_normals = self.srv.gui.add_checkbox("normals", False)
        with self.srv.gui.add_folder("Selectors"):
            self.channel = self.srv.gui.add_dropdown("channel", self.channel_names,
                                                     initial_value=self.channel_names[0])
            self.color_mode = self.srv.gui.add_dropdown("colour mode",
                                                        ("uniform", "distance", "active"),
                                                        initial_value="distance")
            self.size = self.srv.gui.add_number("point size", 0.012, min=0.002, max=0.05, step=0.002)
        self.info = self.srv.gui.add_markdown("")

        # --- static handles added ONCE (kept; toggled/updated per frame) ---
        t0 = self.traces[0]
        self._ground = self.srv.scene.add_box("/ground", color=(170, 170, 178),
                                              dimensions=(4.0, 4.0, 0.004), position=(0.0, 0.0, 0.0))
        self._human = self.srv.scene.add_point_cloud(
            "/human", t0.human_cloud_world.astype(np.float32),
            np.tile(np.array([185, 185, 195], np.uint8), (t0.human_cloud_world.shape[0], 1)),
            point_size=float(self.size.value))
        self._obj_handles = [
            self.srv.scene.add_point_cloud(
                f"/obj{k}", t0.object_clouds_world[k].astype(np.float32),
                np.tile(np.array([255, 140, 0], np.uint8), (t0.object_clouds_world[k].shape[0], 1)),
                point_size=float(self.size.value))
            for k in range(self.grounded.n_objects)]
        sp = np.asarray(t0.targets.style.position, np.float32)
        self._style_pts = self.srv.scene.add_point_cloud(
            "/style_pts", sp, np.tile(np.array([255, 170, 0], np.uint8), (sp.shape[0], 1)),
            point_size=0.03)
        self._style_labels = [self.srv.scene.add_label(f"/style_label/{name}", name,
                                                       position=tuple(float(v) for v in sp[i]))
                              for i, name in enumerate(self.style_links)]

        def render(_=None):
            i = int(self.sld.value)
            trace = self.get(i)
            self._draw_ground()
            self._draw_ghost(trace, i)
            self._draw_objects(trace)
            self._draw_skeleton(trace)
            self._draw_style(trace)
            self._draw_human(trace)
            self._draw_field_lines(trace)
            self._update_info(trace, i)

        for h in (self.sld, self.cb_ground, self.cb_ghost, self.cb_objects, self.cb_skel,
                  self.cb_style, self.cb_style_frames, self.cb_style_labels, self.cb_human,
                  self.cb_witness, self.cb_normals, self.channel, self.color_mode, self.size):
            h.on_update(render)
        render()

        def loop():
            while True:
                if self.play.value:
                    self.sld.value = (int(self.sld.value) + 1) % len(self.frames)
                    render()
                time.sleep(1.0 / float(self.fps.value))
        threading.Thread(target=loop, daemon=True).start()
        print(f"viser ready -> http://localhost:{self.port}")
        while True:
            time.sleep(1)


def view_trace(spec: SceneSpec, *, port: int = 8080, frame_step: int = 2,
               max_frames: int = 200) -> None:
    """Build + bake + serve the FrameTrace viewer for ``spec`` (the ``view_scene``-style entry)."""
    Viewer(spec, port=port, frame_step=frame_step, max_frames=max_frames).run()


def main() -> None:
    ap = argparse.ArgumentParser()
    add_scene_args(ap)
    a = ap.parse_args()
    spec = scene_from_args(a)
    view_trace(spec, port=a.port, frame_step=a.frame_step, max_frames=a.max_frames)


if __name__ == "__main__":
    main()
