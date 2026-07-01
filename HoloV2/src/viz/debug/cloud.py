"""Visualiseur de nuage de points — débogage visuel de la cuisson ``point_cloud`` (réécrit sur ``viz/core``).

Construit le nuage humain creux éparpillé-skinned du sujet (réutilisant l'échantillonnage de la
correspondance) et le nuage rigide de chaque objet, puis les pose par-frame avec la seule op
``pose_cloud`` — le chemin d'exécution sans-maille, sans-torch. Les points humains sont coloriés
par leur erreur de parité contre la VRAIE surface SMPL posée (avance complète) via ``core.colors.parity``,
donc on peut SEE le LBS-on-cloud tracker du corps ; les points des objets (rigide K=1) s'assoient sur
la surface objet. Viewer de débogage : il pilote la cuisson ET importe DÉLIBÉRÉMENT l'op AVAL
``targets.interaction.pose_cloud`` pour exercer le chemin runtime exact que le solveur utilise.
viser reste confiné à ``core/viser_ops`` + ce module.

Exécution :
    python -m src.viz.debug.cloud --motion-path <smplx.npz> --model-dir <smplx_models> [--dataset hodome]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ... import paths                               # pour _DEFAULT_CORR (HOLOV2_ROOT centralisé)
from ...prepare.config import CloudConfig
from ...prepare.contracts import SceneSpec
from ...prepare.load import load
from ...prepare.load.mesh import load_mesh
from ...prepare.load.smpl import build_body_model
from ...prepare.point_cloud import build_human_cloud, build_object_cloud
from ...prepare.point_cloud.correspondence import load_correspondence
from ...targets.interaction import pose_cloud       # op AVAL (délibéré) : chemin runtime cloud du solveur
from ..core.colors import parity                    # colormap parité : bleu 0 → rouge ≥ vmax
from ..core.player import play_loop
from ..core.viser_ops import add_point_cloud, hide, quat_wxyz_to_R
from ._args import add_scene_args, scene_from_args
from ._geometry import parity_error, surface_points

_DEFAULT_CORR = paths.HOLOV2_ROOT / "cache" / "correspondence" / "corr_neutral.npz"


def _object_world(cloud, pose7: np.ndarray) -> np.ndarray:
    """(P, 3) nuage d'objet posé par une ``[x, y, z, qw, qx, qy, qz]`` pose monde via le ``pose_cloud`` partagé."""
    # quat_wxyz_to_R attend un batch (L, 4) ; on passe (1, 4) et on extrait [0] pour avoir (3, 3).
    R = quat_wxyz_to_R(np.asarray(pose7, np.float64)[3:7][None])[0]   # (3, 3), wxyz -> R
    return pose_cloud(cloud, R[None], np.asarray(pose7, np.float64)[:3][None])


def view_cloud(spec: SceneSpec, corr_path: Path, *, port: int = 8080, frame_step: int = 2,
               max_frames: int = 150, vmax: float = 0.02) -> None:
    """Lance le visualiseur de nuage de points interactif pour un ``SceneSpec`` donné.

    Charge la correspondance, précalcule les nuages posés + erreurs de parité sur les frames
    sélectionnées, puis ouvre un serveur viser. Bloque jusqu'à interruption (keep-alive via
    ``play_loop``).

    Args:
        spec: Spécification de scène (dataset, chemin motion, répertoire modèle SMPL-X…).
        corr_path: Chemin vers la cache de correspondance (.npz).
        port: Port viser (défaut 8080).
        frame_step: Pas d'échantillonnage des frames (défaut 2).
        max_frames: Nombre maximal de frames affichées (défaut 150).
        vmax: Seuil de saturation de la colormap parité, en mètres (défaut 0.02 m = 20 mm).
    """
    import viser

    raw = load(spec)
    if not raw.is_parametric:
        raise ValueError("le nuage humain nécessite un corps paramétrique (params SMPL) ; cette source n'en a pas")
    params = raw.smpl_params
    body = build_body_model(params, Path(spec.smpl_model_dir))
    _, sampling = load_correspondence(corr_path)
    human = build_human_cloud(body, sampling, CloudConfig())
    obj_clouds = [build_object_cloud(*load_mesh(p), CloudConfig()) for p in raw.object_mesh_paths]

    frames = list(range(0, raw.n_frames, frame_step))[:max_frames]
    F, N = len(frames), human.n_points
    V = body.rest_vertices(params).shape[0]
    tri_v = body.faces[sampling.tri_idx]                            # (N, 3) triangles de référence de surface
    posed = np.empty((F, N, 3), np.float32)
    verts = np.empty((F, V, 3), np.float32)
    colors = np.empty((F, N, 3), np.uint8)
    obj_posed = [np.empty((F, c.n_points, 3), np.float32) for c in obj_clouds]
    print(f"human cloud: {N} pts, K={human.n_influences}; {len(obj_clouds)} object cloud(s) "
          f"[{', '.join(str(c.n_points) for c in obj_clouds) or '-'}]; precomputing {F} frames ...")
    med = np.empty(F)
    p95 = np.empty(F)
    for i, t in enumerate(frames):
        v = body.posed_vertices(params, t)                          # (V, 3) avance SMPL complète (réf parité)
        ref = surface_points(v, tri_v, sampling.bary.astype(np.float64))
        pc = pose_cloud(human, *body.bone_transforms(params, t))    # (N, 3) chemin d'exécution sans-maille
        err = parity_error(pc, ref)
        verts[i], posed[i], colors[i] = v, pc, parity(err, vmax)
        med[i], p95[i] = np.median(err), np.percentile(err, 95)
        for k, c in enumerate(obj_clouds):
            obj_posed[k][i] = _object_world(c, raw.object_poses_raw[k][t])
    print(f"parity over clip: median {med.mean()*1000:.1f}mm, p95 {p95.mean()*1000:.1f}mm")

    srv = viser.ViserServer(port=port)
    srv.scene.add_grid("/grid", width=4.0, height=4.0)
    with srv.gui.add_folder("Display"):
        show_human = srv.gui.add_checkbox("human cloud", True)
        show_objs = srv.gui.add_checkbox("object clouds", True)
        show_mesh = srv.gui.add_checkbox("SMPL surface (ghost)", False)
        size = srv.gui.add_number("point size", 0.012, min=0.002, max=0.05, step=0.002)
    info = srv.gui.add_markdown("")

    hum_h = add_point_cloud(srv, "/human", posed[0], colors[0], point_size=float(size.value))
    obj_h = [add_point_cloud(srv, f"/obj{k}", op[0],
                             np.tile([[255, 140, 0]], (op.shape[1], 1)).astype(np.uint8),
                             point_size=float(size.value)) for k, op in enumerate(obj_posed)]
    ghost_h: list = [None]                                          # dernière poignée /ghost (ré-ajoutée par frame)

    # Frame courante mémorisée pour re-déclencher render depuis les callbacks GUI sans le slider.
    _cur: list[int] = [0]

    def render(f: int) -> None:
        """Rafraîchit toutes les poignées de scène pour la frame ``f``.

        Appelé par ``play_loop`` à chaque pas du slider (et au rendu initial) avec l'index de frame.
        """
        _cur[0] = f
        hum_h.points = posed[f]
        hum_h.colors = colors[f]
        hum_h.point_size = float(size.value)
        hum_h.visible = show_human.value
        for k, h in enumerate(obj_h):
            h.points = obj_posed[k][f]
            h.point_size = float(size.value)
            h.visible = show_objs.value
        # Ghost SMPL : ré-ajouté par frame car les sommets changent ; masqué via hide() si désactivé.
        if show_mesh.value:
            ghost_h[0] = srv.scene.add_mesh_simple("/ghost", verts[f], body.faces,
                                                   color=(200, 200, 210), opacity=0.4, side="double")
        elif ghost_h[0] is not None:
            hide(ghost_h[0])
        info.content = (f"**frame {frames[f]}** ({f + 1}/{F})\n\n"
                        f"err parité humain — médiane **{med[f]*1000:.1f}mm**, p95 **{p95[f]*1000:.1f}mm**\n\n"
                        f"couleur humain : bleu 0 → rouge ≥ {vmax*1000:.0f}mm · objets : orange (rigide K=1)")

    def _rerender(_=None) -> None:
        """Redéclenche render au frame courant quand un toggle GUI change (indépendamment du slider)."""
        render(_cur[0])

    # Câble les contrôles Display sur _rerender (le slider Playback est câblé par play_loop)
    for h in (show_human, show_objs, show_mesh, size):
        h.on_update(_rerender)

    print(f"viser ready -> http://localhost:{port}")
    # play_loop gère le dossier Playback (slider/play/fps), le rendu initial et le keep-alive.
    # Contrairement à Player(source, layers), play_loop est prévu pour les viewers debug sans Source/VizFrame.
    play_loop(srv, n_frames=F, render=render, fps_default=20)


def main() -> None:
    """Point d'entrée CLI : parse les arguments et lance ``view_cloud``."""
    ap = argparse.ArgumentParser(description="Visualiseur de nuage de points — débogage cuisson point_cloud")
    add_scene_args(ap)
    ap.add_argument("--corr", type=Path, default=_DEFAULT_CORR, help="cache de correspondance (.npz)")
    a = ap.parse_args()
    spec = scene_from_args(a)
    view_cloud(spec, a.corr, port=a.port, frame_step=a.frame_step, max_frames=a.max_frames)


if __name__ == "__main__":
    main()
