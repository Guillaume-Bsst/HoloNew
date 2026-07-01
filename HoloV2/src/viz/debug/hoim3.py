"""Vue multi-personnes de débogage pour HOI-M3 — valide le chargement de bout en bout (réécrit sur
``viz/core``).

Les scènes HOI-M3 ont plusieurs personnes, chacune manipulant différents objets ; le loader à un seul
humain garde une personne + tous les objets, ce qui semble incohérent (les autres objets sont pilotés
par des personnes que nous ne montrons pas). Cette vue rend TOUTES les personnes + tous les objets
ensemble (comme la boîte à outils officielle le fait), donc le chargement par-entité peut être vérifié
par rapport à une scène cohérente. Viewer de débogage : réutilise le loader (objets + une personne) ET
pilote délibérément l'interne de l'étage ``load`` ``build_person_params`` (les autres personnes) + le
modèle de corps SMPL-X — exception debug-viewer de l'ARCHITECTURE.md. Conserve son PROPRE parser (hoim3
construit sa SceneSpec autrement que les flags partagés). viser confiné à ``core/viser_ops`` + ce module ;
lecture/keep-alive depuis ``core.player.play_loop``.

Exécution :
    python -m src.viz.debug.hoim3 --motion-path <..._human.npz> --model-dir <smplx_models>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ... import paths
from ...prepare.contracts import SceneSpec
from ...prepare.load import load
from ...prepare.load.datasets.hoim3 import build_person_params
from ..core.player import play_loop
from ..core.viser_ops import hide
from ._args import _g1_robot

_PALETTE = [(70, 130, 220), (220, 90, 90), (90, 200, 120), (210, 170, 60), (170, 110, 210)]


def view(spec: SceneSpec, *, port: int = 8080, frame_step: int = 30, max_frames: int = 150) -> None:
    """Lance le visualiseur multi-personnes interactif pour une scène HOI-M3.

    Charge les objets via le loader standard, pose TOUTES les personnes présentes frame par frame
    via ``build_person_params``, puis ouvre un serveur viser. Bloque jusqu'à interruption
    (keep-alive via ``play_loop``).

    Args:
        spec: Spécification de scène HOI-M3 (chemin motion + modèles SMPL-X).
        port: Port viser (défaut 8080).
        frame_step: Pas d'échantillonnage des frames (défaut 30).
        max_frames: Nombre maximal de frames affichées (défaut 150).
    """
    import trimesh
    import viser

    raw = load(spec)                                          # objets (+ une personne) gratuitement
    hd = np.load(str(spec.motion_path), allow_pickle=True)
    smpl_params = hd["smpl_params"]
    gender = str(hd["gender"])
    ids = [int(np.asarray(p["id"])) for p in smpl_params[0]]  # personnes présentes à la frame 0
    frames = list(range(0, raw.n_frames, frame_step))[:max_frames]
    F = len(frames)
    print(f"HOI-M3 multi-person: {len(ids)} people {ids}, {len(raw.object_poses_raw)} objects, "
          f"showing {F} frames")

    # Mailles SMPL-X posées par-personne pour les frames affichées (avance réelle).
    persons = []  # liste de (verts (F,V,3), faces, couleur)
    for k, pid in enumerate(ids):
        params, body = build_person_params(smpl_params, pid, gender, Path(spec.smpl_model_dir),
                                           spec.smplh_dir, spec.smpl2smplx_pkl)
        verts = np.stack([body.posed_vertices(params, t) for t in frames]).astype(np.float32)
        persons.append((verts, body.faces, _PALETTE[k % len(_PALETTE)]))
        print(f"  person {pid}: posed {F} frames")

    # Objets rigides : maille locale centrée + pose monde par-frame (pos_xyz + quat_wxyz), exactement
    # comme le loader les a produits.
    objs = []  # liste de (verts_local (V,3), faces (F,3), poses_frames (F,7) pos_xyz+quat_wxyz)
    for kk in range(len(raw.object_poses_raw)):
        m = trimesh.load(str(raw.object_mesh_paths[kk]), force="mesh", process=False, skip_materials=True)
        objs.append((np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32),
                     np.asarray(raw.object_poses_raw[kk], np.float32)[frames]))

    print("done. starting viser ...")
    srv = viser.ViserServer(port=port)
    srv.scene.add_grid("/grid", width=6.0, height=6.0)
    with srv.gui.add_folder("Display"):
        show_people = srv.gui.add_checkbox("people", True)
        show_obj = srv.gui.add_checkbox("objects", True)
    info = srv.gui.add_markdown("")

    # Maillages objets (orange) : maille locale créée une fois, transform mis à jour par-frame.
    oh = [srv.scene.add_mesh_simple(f"/obj{k}", vl, fl, color=(255, 140, 0), side="double")
          for k, (vl, fl, _) in enumerate(objs)]
    # Dernières poignées par-personne (ré-ajoutées par-frame car les sommets changent).
    person_h: list = [None] * len(persons)

    # Frame courante mémorisée pour re-déclencher render depuis les callbacks GUI sans le slider.
    _cur: list[int] = [0]

    def render(f: int) -> None:
        """Rafraîchit toutes les poignées de scène pour la frame ``f``.

        Appelé par ``play_loop`` à chaque pas du slider (et au rendu initial) avec l'index de frame.
        """
        _cur[0] = f
        # Maillages personnes : ré-ajoutés à chaque frame (sommets changent) ou masqués via hide().
        for k, (verts, faces, color) in enumerate(persons):
            if show_people.value:
                person_h[k] = srv.scene.add_mesh_simple(f"/person{k}", verts[f], faces,
                                                         color=color, side="double")
            elif person_h[k] is not None:
                hide(person_h[k])
        # Objets : transform monde mis à jour, maille locale inchangée.
        for k, (_, _, poses) in enumerate(objs):
            oh[k].position = poses[f][:3]
            oh[k].wxyz = poses[f][3:]
            oh[k].visible = show_obj.value
        info.content = f"**frame {frames[f]}** ({f + 1}/{F}) — {len(persons)} people"

    def _rerender(_=None) -> None:
        """Redéclenche render au frame courant quand un toggle GUI change (indépendamment du slider)."""
        render(_cur[0])

    # Câble les toggles Display sur _rerender (le slider Playback est câblé par play_loop).
    for h in (show_people, show_obj):
        h.on_update(_rerender)

    print(f"viser ready -> http://localhost:{port}")
    # play_loop gère le dossier Playback (slider/play/fps), le rendu initial et le keep-alive.
    # Contrairement à Player(source, layers), play_loop est prévu pour les viewers debug sans Source/VizFrame.
    play_loop(srv, n_frames=F, render=render, fps_default=20)


def main() -> None:
    """Point d'entrée CLI : parse les arguments et lance ``view``."""
    ap = argparse.ArgumentParser(description="Visualiseur multi-personnes HOI-M3 — débogage chargement")
    ap.add_argument("--motion-path", required=True, type=Path,
                    help="absolu, ou relatif à [datasets.hoim3].motion dans paths.toml")
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="répertoire modèle SMPL-X ; défaut : paths.toml [models].smplx")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--frame-step", type=int, default=30)
    ap.add_argument("--max-frames", type=int, default=150)
    a = ap.parse_args()
    try:
        cfg = paths.load_paths()
    except FileNotFoundError:
        # paths.toml FORTEMENT requis uniquement pour un défaut : model-dir manquant ou mouvement relatif.
        if a.model_dir is None or not Path(a.motion_path).is_absolute():
            raise
        cfg = {}
    model_dir = a.model_dir if a.model_dir is not None else paths.smplx_dir(cfg)
    motion = paths.resolve_motion("hoim3", a.motion_path, cfg)
    spec = SceneSpec(dataset="hoim3", motion_path=motion, robot=_g1_robot(), smpl_model_dir=model_dir,
                     smplh_dir=paths.smplh_dir(cfg), smpl2smplx_pkl=paths.smpl2smplx_pkl(cfg))
    view(spec, port=a.port, frame_step=a.frame_step, max_frames=a.max_frames)


if __name__ == "__main__":
    main()
