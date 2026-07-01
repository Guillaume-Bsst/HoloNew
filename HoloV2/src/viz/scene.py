"""Visualiseur d'aperçu de scène — débogage visuel de l'étape ``load``.

Étant donné une ``SceneSpec``, charge le ``RawMotion`` + construit le ``BodyModel`` et montre, par-frame,
la maille SMPL-X posée, le squelette (os FK + articulations démo), les objet(s) posés par leurs poses monde,
et le sol. Consommateur pur (pilote ``load`` pour récupérer des artefacts ; pas de hooks de calcul). Le
visualiseur complet ``FrameTrace`` (``viewer.py``) vient plus tard, une fois que ``targets`` existe.

Exécution :
    python -m src.viz.scene --motion-path <smplx.npz> --model-dir <smplx_models> [--dataset hodome]
"""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as _Rot

from ..prepare.contracts import SceneSpec
from ..prepare.config import CalibrationConfig
from ..prepare.calibration import build_calibration
from ..prepare.load import load
from ..prepare.load.smpl import build_body_model
from .debug._args import add_scene_args, scene_from_args


def _object_world_lowz(vl: np.ndarray, poses: np.ndarray, cap: int = 8000):
    """Point le plus bas MONDE par-frame d'un objet rigide posé par ``poses`` (F,7) pos-d'abord wxyz.

    Retourne ``(min_z (F,), low_point (F,3))``. L'ensemble de vertices ``vl`` (V,3 local) est sous-échantillonné à
    ``cap`` pour borner le coût sur les scans denses — un point le plus bas quasi-exact, suffisant pour un marqueur débogage."""
    v = vl
    if vl.shape[0] > cap:
        v = vl[np.random.default_rng(0).choice(vl.shape[0], cap, replace=False)]
    rot = _Rot.from_quat(poses[:, [4, 5, 6, 3]]).as_matrix()          # wxyz → xyzw
    world = np.einsum("fij,vj->fvi", rot, v) + poses[:, None, :3]     # (F, V, 3)
    z = world[:, :, 2]
    lo = z.argmin(axis=1)
    return z.min(axis=1), world[np.arange(world.shape[0]), lo]


def view_scene(spec: SceneSpec, *, port: int = 8080, frame_step: int = 2, max_frames: int = 200) -> None:
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
    bones = np.empty((F, body.n_bones, 3), np.float32) if body is not None else None
    print("precomputing posed meshes/skeletons ...")
    for i, t in enumerate(frames):
        demo_j[i] = raw.joint_pos[t]
        if body is not None:
            verts[i] = body.posed_vertices(raw.smpl_params, t)
            bones[i] = body.bone_transforms(raw.smpl_params, t)[1]

    # les objets sont rigides : gardez la maille locale + pose monde par-frame, mettez à jour la transformation par-frame.
    objs = []  # (verts_local, faces, poses_frames (F, 7))
    for k in range(len(raw.object_poses_raw)):
        m = trimesh.load(str(raw.object_mesh_paths[k]), force="mesh", process=False, skip_materials=True)
        vl = np.asarray(m.vertices, np.float32)
        fl = np.asarray(m.faces, np.int32)
        poses = np.asarray(raw.object_poses_raw[k], np.float32)[frames]
        objs.append((vl, fl, poses))

    # segments de squelette (parent → enfant)
    bone_pairs = [(int(parents[j]), j) for j in range(body.n_bones) if parents[j] >= 0] if body else []

    # --- débogage d'ancrage : l'étalonnage + dégagements au sol par-frame (BRUT, pré-ancrage) ---
    # L'ancrage est PAR ENTITÉ : l'humain chute par calib.human_offset, TOUS les objets par l'
    # calib.object_offset partagée. Ces dégagements nous laissent VOIR chaque entité atterrir sur z=0
    # (l'humain peut flotter tandis que les objets reposent déjà sur le sol, d'où les décalages humain/objet scindés).
    calib = build_calibration(raw, CalibrationConfig())                # ancrage sans-corps
    human_offset = float(calib.human_offset)
    object_offset = float(calib.object_offset)                         # partagée par tous les objets
    # Z monde le plus bas humain + point le plus bas par-frame (surface si paramétrique, sinon articulations démo).
    src = verts if verts is not None else demo_j
    human_minz = src[:, :, 2].min(axis=1)                               # (F,)
    human_low = src[np.arange(F), src[:, :, 2].argmin(axis=1)]          # (F, 3)
    # Décalage du sol humain = un PERCENTILE de la hauteur INFÉRIEURE du PIED-ARTICULATION mocap sur le clip, réglé
    # en direct par un curseur. L'articulation du pied est robuste à la pénétration de la semelle SMPL (la curl-orteil plonge la
    # maille SOUS le niveau de repos, donc poursuivre le point le plus bas surélève trop l'humain) ; le percentile
    # nous permet de cibler le niveau de REPOS/contact à la place. ``sole_med`` (la méthode actuelle) est gardée comme
    # référence uniquement à l'écran.
    sole_med = float(np.median(human_minz))                            # méthode actuelle, pour contraste
    _foot = [i for i, n in enumerate(raw.joint_names) if n in ("L_Foot", "R_Foot")]
    lower_foot = raw.joint_pos[:, _foot, 2].min(axis=1) if _foot else human_minz   # (T,) z pied plus bas
    obj_minz, obj_low = [], []                                          # par objet : (F,), (F, 3)
    for (vl, _fl, poses) in objs:
        mz, lp = _object_world_lowz(vl, poses)
        obj_minz.append(mz); obj_low.append(lp)
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
    with srv.gui.add_folder("Playback"):
        sld = srv.gui.add_slider("frame", 0, F - 1, 1, 0)
        play = srv.gui.add_checkbox("play", False)
        fps = srv.gui.add_number("fps", 20, min=1, max=120, step=1)
    with srv.gui.add_folder("Display"):
        show_mesh = srv.gui.add_checkbox("SMPL mesh", True)
        show_joints = srv.gui.add_checkbox("demo joints", True)
        show_bones = srv.gui.add_checkbox("skeleton", True)
        show_obj = srv.gui.add_checkbox("objects", True)
    with srv.gui.add_folder("Grounding"):
        apply_ground = srv.gui.add_checkbox("apply grounding", True)
        foot_pct = srv.gui.add_slider("foot offset pct", 0, 100, 1, 50)   # percentile of lower foot z
        show_floor = srv.gui.add_checkbox("floor plane z=0", True)
        show_low = srv.gui.add_checkbox("lowest-point markers", True)
    info = srv.gui.add_markdown("")

    floor_h = srv.scene.add_box("/floor", color=(170, 170, 178), dimensions=(4.0, 4.0, 0.004),
                                position=(0.0, 0.0, 0.0))
    hj = srv.scene.add_point_cloud("/joints", demo_j[0], np.tile([[40, 200, 60]], (n_demo, 1)).astype(np.uint8),
                                   point_size=0.025)
    hobj = [srv.scene.add_mesh_simple(f"/obj{k}", vl, fl, color=(255, 140, 0), side="double")
            for k, (vl, fl, _) in enumerate(objs)]
    # marqueurs du point le plus bas : rouge = semelle humaine, jaune = chaque objet — les deux choses sur lesquelles l'ancrage doit
    # reposer sur z=0. Les regarder séparément est exactement comment distinguer un décalage partagé d'un scindé.
    low_h = srv.scene.add_point_cloud("/low_human", human_low[:1], np.array([[255, 40, 40]], np.uint8),
                                      point_size=0.05)
    low_o = [srv.scene.add_point_cloud(f"/low_obj{k}", obj_low[k][:1], np.array([[255, 210, 0]], np.uint8),
                                       point_size=0.05) for k in range(len(objs))]

    def render(_=None):
        f = int(sld.value)
        on = apply_ground.value                              # ancrage = fait tomber chaque entité à z=0
        gh = float(np.percentile(lower_foot, foot_pct.value)) if on else 0.0   # décalage z humain foot-pct
        go = [object_offset if on else 0.0 for _ in range(len(objs))]       # décalage z objet partagé
        dzh = np.array([0.0, 0.0, gh], np.float32)
        if body is not None and show_mesh.value:
            srv.scene.add_mesh_simple("/body", verts[f] - dzh, faces, color=(200, 200, 210),
                                      opacity=0.55, side="double")
        else:
            srv.scene.add_mesh_simple("/body", np.zeros((3, 3), np.float32), np.array([[0, 1, 2]]), opacity=0.0)
        hj.points = demo_j[f] - dzh
        hj.visible = show_joints.value
        if body is not None and show_bones.value and bone_pairs:
            seg = np.stack([np.stack([bones[f, a], bones[f, b]]) for a, b in bone_pairs]).astype(np.float32) - dzh
            srv.scene.add_line_segments("/skeleton", seg,
                                        np.tile([[[0, 120, 255]]], (len(bone_pairs), 2, 1)).astype(np.uint8),
                                        line_width=3.0)
        else:
            srv.scene.add_line_segments("/skeleton", np.zeros((1, 2, 3), np.float32),
                                        np.zeros((1, 2, 3), np.uint8), line_width=0.1)
        for k, (_, _, poses) in enumerate(objs):
            h = hobj[k]
            h.position = poses[f][:3] - np.array([0.0, 0.0, go[k]], np.float32)
            h.wxyz = poses[f][3:]
            h.visible = show_obj.value
        floor_h.visible = show_floor.value
        low_h.points = (human_low[f] - dzh)[None]
        low_h.visible = show_low.value
        for k in range(len(objs)):
            low_o[k].points = (obj_low[k][f] - np.array([0.0, 0.0, go[k]], np.float32))[None]
            low_o[k].visible = show_low.value

        hz = human_minz[f] - gh
        oz = [obj_minz[k][f] - go[k] for k in range(len(objs))]
        info.content = (
            f"**frame {frames[f]}** ({f + 1}/{F}) · grounding **{'ON' if on else 'OFF'}**\n\n"
            f"human offset = **foot-joint p{int(foot_pct.value)} = {gh:+.4f} m**  "
            f"(sole median for contrast: {sole_med:+.4f})\n\n"
            f"lowest z (this frame) — human sole **{hz:+.4f}**" +
            "".join(f", obj{k} **{z:+.4f}**" for k, z in enumerate(oz)) + " m\n\n"
            f"object offset (shared): {object_offset:+.4f} m" + ("" if objs else " (no objects)"))

    for h in (sld, show_mesh, show_joints, show_bones, show_obj, apply_ground, foot_pct,
              show_floor, show_low):
        h.on_update(render)
    render()

    def loop():
        while True:
            if play.value:
                sld.value = (int(sld.value) + 1) % F
                render()
            time.sleep(1.0 / float(fps.value))
    threading.Thread(target=loop, daemon=True).start()
    print(f"viser ready -> http://localhost:{port}")
    while True:
        time.sleep(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    add_scene_args(ap)
    a = ap.parse_args()
    spec = scene_from_args(a)
    view_scene(spec, port=a.port, frame_step=a.frame_step, max_frames=a.max_frames)


if __name__ == "__main__":
    main()
