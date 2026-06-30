"""Visualiseur FrameTrace — le principal visualiseur ``viz`` (un gros serveur viser, beaucoup de toggles).

CONSOMMATEUR PUR (règle d'or 6) : pilote ``prepare`` une fois pour récupérer les assets statiques, CUIT les
``FrameTrace`` par-frame (``targets.pipeline.trace_frame``) pour une lecture fluide, et lit UNIQUEMENT ces
artefacts typés pour dessiner des couches basculables. ZÉRO hook dans le calcul ; viser confiné à ce module ;
une méthode ``_draw_<layer>`` par couche gated checkbox (couches statiques ajoutées une fois, couches dynamiques
mises à jour par tick slider). Voir docs/VIZ.md.

Couches implémentées ici (le côté HUMAIN — tout ce qui est disponible d'une ``FrameTrace`` sans un
robot RÉSOLU) :
  - Playback : curseur de frame · play · fps
  - Static   : plan sol (z=0) · maille SMPL fantôme (par-frame) · nuages d'objets posés
  - Skeleton : os SMPL de ``trace.pose`` (segments parent → enfant)
  - Style    : les 14 cibles de liens ``StyleTargets`` — points (couleur uniforme) + cadres
               d'orientation par-lien (3 courts axes xyz) + labels de nom de lien, pour SEE chaque
               cible de lien s'asseoir sur le squelette (p.ex. ``*_toe_link`` près de la cheville)
  - Interaction (human) : le nuage humain posé colorié par le canal SÉLECTIONNÉ de ``human_field``
               (uniforme / heatmap de distance / masque actif), avec lignes witness facultatives + normales
  - Selectors: dropdown canal · mode couleur · taille de point

DÉFÉRÉ (côté robot, hors de portée ici) : le champ transporté sur les points de correspondance G1, les
lignes de correspondance SMPL<->robot, et la surface iso SDF. Ils ont tous besoin d'une configuration de
robot RÉSOLU (``solve/`` placera les M points de correspondance via robot FK) ; une ``FrameTrace`` seule
ne peut les mettre dans le monde. Aucun URDF de robot / FK de robot chargé ici.

Exécution :
    fuser -k 8080/tcp   # libère le port D'ABORD (ne jamais pkill -f ce script : il se tue seul)
    python -m src.viz.viewer --motion-path <smplx.npz> --model-dir <smplx_models> \
        [--dataset hodome --dataset-root <root> --port 8080 --frame-step 2 --max-frames 200]
"""
from __future__ import annotations

import argparse
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as _Rot

from ..prepare.config import PrepareConfig
from ..prepare.contracts import SceneSpec
from ..prepare.runner import prepare
from ..targets.contracts import FrameTrace
from ..targets.pipeline import trace_frame
from ._scene_args import add_scene_args, scene_from_args

# points d'ancrage couleur (uint8 RGB)
_NEAR = np.array([40, 90, 255], np.float64)    # heatmap de distance : proche/pénétrant (bleu)
_FAR = np.array([255, 60, 50], np.float64)     # heatmap de distance : loin dans la marge (rouge)
_AXIS_COLORS = np.array([[255, 80, 80], [80, 230, 80], [80, 130, 255]], np.uint8)  # x r, y v, z b


def _quat_wxyz_to_R(quat: np.ndarray) -> np.ndarray:
    """(L, 4) quaternions wxyz → (L, 3, 3) matrices de rotation (scipy est xyzw)."""
    q = np.asarray(quat, np.float64)
    return _Rot.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()


def _heat_distance(dist: np.ndarray, margin: float) -> np.ndarray:
    """(P,) distance signée → (P, 3) uint8. bleu = proche/pénétrant (d<=0), rouge = loin (d ~ marge)."""
    t = np.clip(np.asarray(dist, np.float64) / max(margin, 1e-9), 0.0, 1.0)[:, None]
    return (t * _FAR + (1.0 - t) * _NEAR).astype(np.uint8)


def _active_colors(active: np.ndarray) -> np.ndarray:
    """(P,) bool → (P, 3) uint8. vert lumineux où actif (dans la bande de contact), gris tamisé ailleurs."""
    col = np.tile(np.array([70, 70, 80], np.uint8), (len(active), 1))
    col[np.asarray(active, bool)] = (90, 255, 130)
    return col


class Viewer:
    """Un visualiseur viser sur une liste CUITE de ``FrameTrace``. La construction prépare la scène, cuit les
    traces et précalcule le fantôme SMPL par-frame ; ``run()`` construit l'interface graphique + handles et sert."""

    def __init__(self, spec: SceneSpec, *, port: int = 8080, frame_step: int = 2,
                 max_frames: int = 200) -> None:
        self.port = port
        # --- 1) contexte préparé (statique) : load → calibration → canaux → nuages → correspondance
        self.grounded, self.ctx = prepare(spec, PrepareConfig())
        self.robot = spec.robot
        body = self.grounded.body
        if body is None:
            raise ValueError("the FrameTrace viewer needs a parametric body (SMPL params)")

        # --- 2) CUITE : précalcule la FrameTrace de chaque frame affichée (lecture fluide ; get(i) indexe ceci)
        self.frames = list(range(0, self.grounded.n_frames, frame_step))[:max_frames]
        print(f"baking {len(self.frames)} frames (trace_frame) ...")
        self.traces: list[FrameTrace] = [
            trace_frame(self.grounded, self.ctx, self.robot, f) for f in self.frames]
        # La maille fantôme SMPL est aussi par-frame (avance complète) — précalcule à côté des traces.
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

    # fournisseur de frame (VIZ.md) : get(i) → FrameTrace, indexe la liste cuite (une trace_frame en direct à la
    # volée est l'alternative que la conception permet ; la cuisson est utilisée ici pour une lecture fluide).
    def get(self, i: int) -> FrameTrace:
        return self.traces[int(i)]

    # ---------------------------------------------------------------- couches (une _draw_ par toggle)
    def _draw_ground(self) -> None:
        """Plan sol STATIQUE (z=0). Ajouté une fois dans ``run`` ; ici on ne bascule que la visibilité."""
        self._ground.visible = self.cb_ground.value

    def _draw_ghost(self, trace: FrameTrace, i: int) -> None:
        """Maille fantôme SMPL — par-frame (ré-ajoutée comme scene.py), opacité basse donc reste un arrière-plan."""
        if self.cb_ghost.value:
            self.srv.scene.add_mesh_simple("/ghost", self.ghost_verts[i], self.faces,
                                           color=(200, 200, 210), opacity=0.45, side="double")
        else:
            self.srv.scene.add_mesh_simple("/ghost", np.zeros((3, 3), np.float32),
                                           np.array([[0, 1, 2]]), opacity=0.0)

    def _draw_objects(self, trace: FrameTrace) -> None:
        """Nuages d'objets posés (``trace.object_clouds_world``), coloriés par leur PROPRE champ env
        (``trace.targets.env_interaction`` — objet<->sol / objet<->objet) sur le canal SÉLECTIONNÉ,
        donc le contact de l'objet est visible comme le côté humain (choisir canal ``ground`` pour voir l'objet
        reposant sur le sol). ``uniform`` les garde orange pur."""
        c = self._channel_idx()
        mode = self.color_mode.value
        env = trace.targets.env_interaction.per_object
        for k, h in enumerate(self._obj_handles):
            pts = trace.object_clouds_world[k].astype(np.float32)
            if mode == "distance":
                col = _heat_distance(env[k].distance[c], self.margin)
            elif mode == "active":
                col = _active_colors(env[k].active[c])
            else:                                                        # uniforme
                col = np.tile(np.array([255, 140, 0], np.uint8), (pts.shape[0], 1))
            h.points = pts
            h.colors = col
            h.point_size = float(self.size.value)
            h.visible = self.cb_objects.value

    def _draw_skeleton(self, trace: FrameTrace) -> None:
        """Os SMPL de ``trace.pose.bone_pos`` — segments de ligne parent→enfant (voir scene.py)."""
        if self.cb_skel.value and self.bone_pairs:
            bp = np.asarray(trace.pose.bone_pos, np.float32)
            seg = np.stack([np.stack([bp[a], bp[b]]) for a, b in self.bone_pairs]).astype(np.float32)
            col = np.tile([[[0, 120, 255]]], (len(self.bone_pairs), 2, 1)).astype(np.uint8)
            self.srv.scene.add_line_segments("/skeleton", seg, col, line_width=3.0)
        else:
            self.srv.scene.add_line_segments("/skeleton", np.zeros((1, 2, 3), np.float32),
                                             np.zeros((1, 2, 3), np.uint8), line_width=0.1)

    def _draw_style(self, trace: FrameTrace) -> None:
        """Les 14 cibles de liens ``StyleTargets`` — la couche de validation CLE.

        Les points s'assoient à ``style.position`` (orange uniforme) ; les cadres d'orientation par-lien dessinent 3
        courts axes xyz de ``style.orientation`` (wxyz) ; les labels par-lien nomment le lien. Avec le squelette activé,
        ceci montre chaque cible de lien atterrir sur le corps — p.ex. ``left_toe_link`` (mappé à la
        CHEVILLE SMPL + un décalage de pos-orteil) s'assoit à la cheville, le mapping pied→cheville devient visible.
        Les poids de tracking sont une préoccupation du SOLVEUR, pas dans ``StyleTargets`` — donc pas de split
        couleur planté/libre ici."""
        style = trace.targets.style
        pos = np.asarray(style.position, np.float32)                      # (L, 3)
        col = np.tile(np.array([255, 170, 0], np.uint8), (len(pos), 1))   # couleur style uniforme
        self._style_pts.points = pos
        self._style_pts.colors = col
        self._style_pts.point_size = max(float(self.size.value) * 2.0, 0.02)
        self._style_pts.visible = self.cb_style.value

        # cadres d'orientation (3 courts axes par lien).
        if self.cb_style_frames.value and style.orientation is not None:
            rots = _quat_wxyz_to_R(style.orientation)                    # (L, 3, 3)
            axis_len = 0.08
            segs, cols = [], []
            for i in range(len(self.style_links)):
                for a in range(3):
                    d = rots[i][:, a]                                    # dir monde de l'axe du corps a
                    segs.append([pos[i], pos[i] + d * axis_len])
                    cols.append([_AXIS_COLORS[a], _AXIS_COLORS[a]])
            self.srv.scene.add_line_segments("/style_frames", np.asarray(segs, np.float32),
                                             np.asarray(cols, np.uint8), line_width=2.5)
        else:
            self.srv.scene.add_line_segments("/style_frames", np.zeros((1, 2, 3), np.float32),
                                             np.zeros((1, 2, 3), np.uint8), line_width=0.1)
        # labels de nom de lien (handles persistants, repositionnés par-frame).
        for i, h in enumerate(self._style_labels):
            h.position = tuple(float(v) for v in pos[i])
            h.visible = self.cb_style_labels.value

    def _draw_human(self, trace: FrameTrace) -> None:
        """Nuage humain posé ``trace.human_cloud_world`` colorié par le canal SÉLECTIONNÉ de
        ``trace.human_field`` (uniforme / heatmap de distance / masque actif)."""
        pc = trace.human_cloud_world.astype(np.float32)                  # (N, 3) monde
        c = self._channel_idx()
        field = trace.human_field
        mode = self.color_mode.value
        if mode == "distance":
            col = _heat_distance(field.distance[c], self.margin)
        elif mode == "active":
            col = _active_colors(field.active[c])
        else:                                                            # uniforme
            col = np.tile(np.array([185, 185, 195], np.uint8), (pc.shape[0], 1))
        self._human.points = pc
        self._human.colors = col
        self._human.point_size = float(self.size.value)
        self._human.visible = self.cb_human.value

    def _draw_field_lines(self, trace: FrameTrace) -> None:
        """Lignes witness (point → surface la plus proche) et normales (court segment le long de ``direction``) pour
        les sondes ACTIVES du canal sélectionné. Les canaux OBJET stockent witness/direction dans le
        cadre LOCAL-objet (le cadre naturel par-canal du champ), donc les mapper au monde par l'objet
        par-frame ``(R, t)`` ; le canal SOL est déjà en monde."""
        c = self._channel_idx()
        field = trace.human_field
        ch = self.ctx.channels[c]
        active = np.asarray(field.active[c], bool)
        idx = np.where(active)[0]
        want_w, want_n = self.cb_witness.value, self.cb_normals.value
        if len(idx) and (want_w or want_n):
            if len(idx) > 400:                                           # borne le compte de segments
                idx = np.random.default_rng(0).choice(idx, 400, replace=False)
            pts = np.asarray(trace.human_cloud_world, np.float64)[idx]   # (S, 3) monde
            wit = np.asarray(field.witness[c], np.float64)[idx]          # (S, 3) canal-local
            dirn = np.asarray(field.direction[c], np.float64)[idx]       # (S, 3) canal-local
            if ch.object_idx is not None:                               # objet-local → monde
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

    # ---------------------------------------------------------------- assistants
    def _channel_idx(self) -> int:
        return self.channel_names.index(self.channel.value)

    def _update_info(self, trace: FrameTrace, i: int) -> None:
        c = self._channel_idx()
        n_active = int(np.asarray(trace.human_field.active[c]).sum())
        self.info.content = (
            f"**frame {self.frames[i]}** ({i + 1}/{len(self.frames)})\n\n"
            f"channel **{self.channel.value}** · colour **{self.color_mode.value}** · "
            f"active probes **{n_active}** / {trace.human_field.n_points}\n\n"
            f"style points: orange link targets (position + orientation) · "
            f"distance heatmap: blue near/penetrating .. red far (margin {self.margin:.3f} m)\n\n"
            f"deferred (need a solved robot): transported G1 field · SMPL<->robot lines · SDF iso-surface")

    # ---------------------------------------------------------------- serveur
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

        # --- handles statiques ajoutés UNE FOIS (conservés ; basculent/mettent à jour par-frame) ---
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
    """Construit + cuit + sert le visualiseur FrameTrace pour ``spec`` (l'entrée de style ``view_scene``)."""
    Viewer(spec, port=port, frame_step=frame_step, max_frames=max_frames).run()


def main() -> None:
    ap = argparse.ArgumentParser()
    add_scene_args(ap)
    a = ap.parse_args()
    spec = scene_from_args(a)
    view_trace(spec, port=a.port, frame_step=a.frame_step, max_frames=a.max_frames)


if __name__ == "__main__":
    main()
