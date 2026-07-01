"""Visualiseur d'aperçu de scène — débogage visuel de l'étape ``load`` (réécrit sur ``viz/core``).

Étant donné une ``SceneSpec``, charge le ``RawMotion`` + construit le ``BodyModel`` et montre, par-frame,
la maille SMPL-X posée, le squelette (os FK + articulations démo), les objet(s) posés par leurs poses
monde, le sol, ET l'overlay de débogage ANCRAGE (offsets de calibration, curseur pied-percentile live,
et marqueurs point-le-plus-bas que chaque entité doit reposer sur z=0). Viewer de débogage : il pilote
DÉLIBÉRÉMENT les internes ``load``/``calibration`` pour exposer ces intermédiaires non-contractuels
(exception debug-viewer de l'ARCHITECTURE.md) ; viser reste confiné à ``core/viser_ops`` + ce module ;
la boucle Playback/keep-alive vient de ``core.player.play_loop``.

Exécution :
    python -m src.viz.debug.scene --motion-path <smplx.npz> --model-dir <smplx_models> [--dataset hodome]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ...prepare.calibration import build_calibration
from ...prepare.config import CalibrationConfig
from ...prepare.contracts import SceneSpec
from ...prepare.load import load
from ...prepare.load.smpl import build_body_model
# Player importé pour l'assertion structurelle du smoke (scene.Player is core.player.Player) ;
# play_loop est le chemin effectif de lecture — les viewers debug n'ont ni Source ni VizFrame.
from ..core.player import Player, play_loop
from ..core.viser_ops import add_line_segments, add_point_cloud, hide, quat_wxyz_to_R
from ._args import add_scene_args, scene_from_args
from ._geometry import lowest_point, object_world_lowz


def view_scene(spec: SceneSpec, *, port: int = 8080, frame_step: int = 2, max_frames: int = 200) -> None:
    """Lance le visualiseur de scène interactif pour un ``SceneSpec`` donné.

    Charge le ``RawMotion``, précalcule les maillages/squelettes posés sur les frames sélectionnées,
    puis ouvre un serveur viser. Bloque jusqu'à interruption (keep-alive via ``play_loop``).

    Args:
        spec: Spécification de scène (dataset, chemin motion, répertoire modèle SMPL-X…).
        port: Port viser (défaut 8080).
        frame_step: Pas d'échantillonnage des frames (défaut 2).
        max_frames: Nombre maximal de frames affichées (défaut 200).
    """
    import trimesh
    import viser

    raw = load(spec)
    frames = list(range(0, raw.n_frames, frame_step))[:max_frames]
    F = len(frames)
    print(f"loaded {raw.source_format}: T={raw.n_frames}, showing {F} frames, "
          f"{len(raw.object_poses_raw)} object(s), parametric={raw.is_parametric}")

    # --- précalcule par-frame affiché (borné) ---
    body = build_body_model(raw.smpl_params, Path(spec.smpl_model_dir)) if raw.is_parametric else None
    faces = body.faces if body is not None else None
    parents = body.parents if body is not None else None
    n_demo = raw.joint_pos.shape[1]

    verts = None
    if body is not None:
        V = body.rest_vertices(raw.smpl_params).shape[0]
        verts = np.empty((F, V, 3), np.float32)
    demo_j = np.empty((F, n_demo, 3), np.float32)
    # Os squelette : translations des joints par-frame, (F, n_bones, 3)
    bones = np.empty((F, body.n_bones, 3), np.float32) if body is not None else None
    print("precomputing posed meshes/skeletons ...")
    for i, t in enumerate(frames):
        demo_j[i] = raw.joint_pos[t]
        if body is not None:
            verts[i] = body.posed_vertices(raw.smpl_params, t)
            bones[i] = body.bone_transforms(raw.smpl_params, t)[1]

    # Objets rigides : maille locale + pose monde par-frame ; la transformation est mise à jour frame par frame.
    objs = []  # liste de (verts_local (V,3), faces (F,3), poses_frames (F,7) pos_xyz+quat_wxyz)
    for k in range(len(raw.object_poses_raw)):
        m = trimesh.load(str(raw.object_mesh_paths[k]), force="mesh", process=False, skip_materials=True)
        vl = np.asarray(m.vertices, np.float32)
        fl = np.asarray(m.faces, np.int32)
        poses = np.asarray(raw.object_poses_raw[k], np.float32)[frames]
        objs.append((vl, fl, poses))

    # Paires parent→enfant pour les segments de squelette
    bone_pairs = [(int(parents[j]), j) for j in range(body.n_bones) if parents[j] >= 0] if body else []

    # --- débogage ancrage : calibration + dégagements au sol par-frame (BRUT, pré-ancrage) ---
    # L'ancrage est PAR ENTITÉ : l'humain chute par calib.human_offset, TOUS les objets par le
    # calib.object_offset partagé. Ces dégagements permettent de VOIR chaque entité atterrir sur z=0
    # (l'humain peut flotter tandis que les objets reposent déjà au sol, d'où les offsets scindés).
    calib = build_calibration(raw, CalibrationConfig())      # ancrage sans-corps
    human_offset = float(calib.human_offset)
    object_offset = float(calib.object_offset)               # partagé par tous les objets

    # Z monde le plus bas humain + point le plus bas par-frame (surface si paramétrique, sinon joints démo).
    src = verts if verts is not None else demo_j
    human_minz, human_low = lowest_point(src)                # (F,), (F, 3)

    # Offset sol humain = PERCENTILE du z pied-joint inférieur mocap sur le clip, réglé live par le curseur.
    # L'articulation du pied est robuste à la pénétration de la semelle SMPL (le curl-orteil plonge la
    # maille SOUS le niveau de repos, donc suivre le point le plus bas surélèverait trop l'humain) ;
    # le percentile cible le niveau REPOS/contact. ``sole_med`` (méthode actuelle) est gardé pour contraste.
    sole_med = float(np.median(human_minz))                  # méthode actuelle, pour contraste à l'écran
    _foot = [i for i, n in enumerate(raw.joint_names) if n in ("L_Foot", "R_Foot")]
    lower_foot = raw.joint_pos[:, _foot, 2].min(axis=1) if _foot else human_minz   # (T,) z pied inférieur

    obj_minz: list[np.ndarray] = []
    obj_low: list[np.ndarray] = []
    for (vl, _fl, poses) in objs:
        rot = quat_wxyz_to_R(poses[:, 3:7])                  # (F, 3, 3) wxyz -> R via core.viser_ops
        mz, lp = object_world_lowz(vl, rot, poses[:, :3])    # délègue à debug._geometry
        obj_minz.append(mz)
        obj_low.append(lp)

    hz_med = float(np.median(human_minz))
    oz_med = [float(np.median(z)) for z in obj_minz]
    stature_str = f"{body.stature:.3f} m" if body is not None else "n/a"
    print(f"calibration: human_offset={human_offset:+.4f} m, human_stature={stature_str}, "
          f"object_offset={object_offset:+.4f}")
    print(f"  RAW clip-median lowest z: human={hz_med:+.4f}" +
          "".join(f", obj{k}={m:+.4f}" for k, m in enumerate(oz_med)))

    print("done. starting viser ...")
    srv = viser.ViserServer(port=port)
    srv.scene.add_grid("/grid", width=4.0, height=4.0)

    # Dossiers Display et Grounding créés AVANT play_loop (qui ajoutera Playback après)
    with srv.gui.add_folder("Display"):
        show_mesh = srv.gui.add_checkbox("SMPL mesh", True)
        show_joints = srv.gui.add_checkbox("demo joints", True)
        show_bones = srv.gui.add_checkbox("skeleton", True)
        show_obj = srv.gui.add_checkbox("objects", True)
    with srv.gui.add_folder("Grounding"):
        apply_ground = srv.gui.add_checkbox("apply grounding", True)
        foot_pct = srv.gui.add_slider("foot offset pct", 0, 100, 1, 50)   # percentile du z pied inférieur
        show_floor = srv.gui.add_checkbox("floor plane z=0", True)
        show_low = srv.gui.add_checkbox("lowest-point markers", True)
    info = srv.gui.add_markdown("")

    # Plan au sol persistant
    floor_h = srv.scene.add_box("/floor", color=(170, 170, 178), dimensions=(4.0, 4.0, 0.004),
                                position=(0.0, 0.0, 0.0))
    # Nuage de joints démo (vert)
    hj = add_point_cloud(srv, "/joints", demo_j[0],
                         np.tile([[40, 200, 60]], (n_demo, 1)).astype(np.uint8), point_size=0.025)
    # Maillages objets (orange) — transform mis à jour par-frame via .position/.wxyz
    hobj = [srv.scene.add_mesh_simple(f"/obj{k}", vl, fl, color=(255, 140, 0), side="double")
            for k, (vl, fl, _) in enumerate(objs)]
    # Segments squelette (bleu) — .points mis à jour par-frame
    nseg = max(len(bone_pairs), 1)
    skel_h = add_line_segments(srv, "/skeleton", np.zeros((nseg, 2, 3), np.float32),
                               np.tile([[[0, 120, 255]]], (nseg, 2, 1)).astype(np.uint8), line_width=3.0)
    # Marqueurs point le plus bas : rouge = semelle humaine, jaune = chaque objet.
    # Regarder les deux séparément montre exactement comment un offset partagé diffère d'un offset scindé.
    low_h = add_point_cloud(srv, "/low_human", human_low[:1],
                            np.array([[255, 40, 40]], np.uint8), point_size=0.05)
    low_o = [add_point_cloud(srv, f"/low_obj{k}", obj_low[k][:1],
                             np.array([[255, 210, 0]], np.uint8), point_size=0.05)
             for k in range(len(objs))]
    # Maillage SMPL (ré-ajouté par-frame car les sommets changent) ; None jusqu'au premier rendu.
    body_h: list = [None]

    # Frame courante mémorisée pour re-déclencher render depuis les callbacks GUI sans le slider.
    _cur: list[int] = [0]

    def render(f: int) -> None:
        """Rafraîchit toutes les poignées de scène pour la frame ``f``.

        Appelé par ``play_loop`` à chaque pas du slider (et au rendu initial) avec l'index de frame.
        """
        _cur[0] = f
        on = apply_ground.value                               # ancrage = fait descendre chaque entité à z=0
        gh = float(np.percentile(lower_foot, foot_pct.value)) if on else 0.0   # décalage z humain foot-pct
        go = [object_offset if on else 0.0 for _ in range(len(objs))]           # décalage z objet partagé
        dzh = np.array([0.0, 0.0, gh], np.float32)

        # Maillage SMPL : ré-ajouté à chaque frame (sommets changent) ou masqué via hide()
        if body is not None and show_mesh.value:
            body_h[0] = srv.scene.add_mesh_simple(
                "/body", verts[f] - dzh, faces, color=(200, 200, 210), opacity=0.55, side="double")
        elif body_h[0] is not None:
            hide(body_h[0])

        # Articulations démo
        hj.points = demo_j[f] - dzh
        hj.visible = show_joints.value

        # Squelette : segments parent→enfant, masqués via hide() si désactivé
        if body is not None and show_bones.value and bone_pairs:
            seg = (np.stack([np.stack([bones[f, a], bones[f, b]]) for a, b in bone_pairs])
                   .astype(np.float32) - dzh)
            skel_h.points = seg
            skel_h.visible = True
        else:
            hide(skel_h)

        # Objets : transform world mis à jour, maille locale inchangée
        for k, (_, _, poses) in enumerate(objs):
            h = hobj[k]
            h.position = poses[f][:3] - np.array([0.0, 0.0, go[k]], np.float32)
            h.wxyz = poses[f][3:]
            h.visible = show_obj.value

        # Plan au sol et marqueurs point-le-plus-bas
        floor_h.visible = show_floor.value
        low_h.points = (human_low[f] - dzh)[None]
        low_h.visible = show_low.value
        for k in range(len(objs)):
            low_o[k].points = (obj_low[k][f] - np.array([0.0, 0.0, go[k]], np.float32))[None]
            low_o[k].visible = show_low.value

        # Panneau d'info : offsets courants + z-au-sol de cette frame
        hz = human_minz[f] - gh
        oz = [obj_minz[k][f] - go[k] for k in range(len(objs))]
        info.content = (
            f"**frame {frames[f]}** ({f + 1}/{F}) · grounding **{'ON' if on else 'OFF'}**\n\n"
            f"human offset = **foot-joint p{int(foot_pct.value)} = {gh:+.4f} m**  "
            f"(sole median for contrast: {sole_med:+.4f})\n\n"
            f"lowest z (this frame) — human sole **{hz:+.4f}**" +
            "".join(f", obj{k} **{z:+.4f}**" for k, z in enumerate(oz)) + " m\n\n"
            f"object offset (shared): {object_offset:+.4f} m" + ("" if objs else " (no objects)"))

    def _rerender(_=None) -> None:
        """Redéclenche render au frame courant quand un toggle GUI change (indépendamment du slider)."""
        render(_cur[0])

    # Câble les toggles Display/Grounding sur _rerender (le slider Playback est câblé par play_loop)
    for h in (show_mesh, show_joints, show_bones, show_obj, apply_ground, foot_pct,
              show_floor, show_low):
        h.on_update(_rerender)

    print(f"viser ready -> http://localhost:{port}")
    # play_loop gère le dossier Playback (slider/play/fps), le rendu initial et le keep-alive.
    # Contrairement à Player(source, layers), play_loop est prévu pour les viewers debug sans Source/VizFrame.
    play_loop(srv, n_frames=F, render=render, fps_default=20)


def main() -> None:
    """Point d'entrée CLI : parse les arguments et lance ``view_scene``."""
    ap = argparse.ArgumentParser(description="Visualiseur de scène — débogage étape load")
    add_scene_args(ap)
    a = ap.parse_args()
    spec = scene_from_args(a)
    view_scene(spec, port=a.port, frame_step=a.frame_step, max_frames=a.max_frames)


if __name__ == "__main__":
    main()
