"""Visualiseur de nuage de points — débogage visuel de la cuisson ``point_cloud`` (humain + objets).

Construit le nuage humain creux éparpillé-skinned du sujet (réutilisant l'échantillonnage de la correspondance) et le
nuage rigide de chaque objet, puis les pose par-frame avec la seule opération ``pose_cloud`` — le chemin d'exécution
sans-maille, sans-torch. Les points humains sont coloriés par leur erreur de parité contre la VRAIE surface SMPL posée
(avance complète), donc on peut SEE le LBS-on-cloud tracker du corps et ferme les creuses articulaires ;
les points des objets (rigide K=1) sont posés par leur pose monde par-frame et s'assoient sur la surface objet.
Consommateur pur (pilote la cuisson pour récupérer des artefacts ; pas de hooks de calcul).

Exécution :
    python -m src.viz.cloud --motion-path <smplx.npz> --model-dir <smplx_models> [--dataset hodome]
"""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as _Rot

from ..prepare.contracts import SceneSpec
from ..prepare.config import CloudConfig
from ..prepare.load import load
from ..prepare.load.mesh import load_mesh
from ..prepare.load.smpl import build_body_model
from ..prepare.point_cloud import build_human_cloud, build_object_cloud
from ..prepare.point_cloud.correspondence import load_correspondence
from ..targets.interaction import pose_cloud
from .debug._args import add_scene_args, scene_from_args

_DEFAULT_CORR = Path(__file__).resolve().parent.parent.parent / "cache" / "correspondence" / "corr_neutral.npz"


def _heat(err: np.ndarray, vmax: float) -> np.ndarray:
    """(N,) erreur → (N,3) uint8, bleu (0) → rouge (>= vmax). vmax en mètres."""
    t = np.clip(np.asarray(err) / vmax, 0.0, 1.0)[:, None]
    return (np.concatenate([t, np.zeros_like(t), 1.0 - t], axis=1) * 255).astype(np.uint8)


def _object_world(cloud, pose7: np.ndarray) -> np.ndarray:
    """(P,3) nuage d'objet posé par une ``[x,y,z,qw,qx,qy,qz]`` pose monde via le ``pose_cloud`` partagé."""
    rot = _Rot.from_quat(np.asarray(pose7, np.float64)[[4, 5, 6, 3]]).as_matrix()   # wxyz → xyzw
    return pose_cloud(cloud, rot[None], np.asarray(pose7, np.float64)[:3][None])


def view_cloud(spec: SceneSpec, corr_path: Path, *, port: int = 8080, frame_step: int = 2,
               max_frames: int = 150, vmax: float = 0.02) -> None:
    import viser

    raw = load(spec)
    if not raw.is_parametric:
        raise ValueError("the human cloud needs a parametric body (SMPL params); this source has none")
    params = raw.smpl_params
    body = build_body_model(params, Path(spec.smpl_model_dir))
    _, sampling = load_correspondence(corr_path)
    human = build_human_cloud(body, sampling, CloudConfig())
    obj_clouds = [build_object_cloud(*load_mesh(p), CloudConfig()) for p in raw.object_mesh_paths]

    frames = list(range(0, raw.n_frames, frame_step))[:max_frames]
    F, N = len(frames), human.n_points
    V = body.rest_vertices(params).shape[0]
    tri_v = body.faces[sampling.tri_idx]                            # (N,3) pour la référence de surface
    posed = np.empty((F, N, 3), np.float32)
    verts = np.empty((F, V, 3), np.float32)
    colors = np.empty((F, N, 3), np.uint8)
    obj_posed = [np.empty((F, c.n_points, 3), np.float32) for c in obj_clouds]
    print(f"human cloud: {N} pts, K={human.n_influences}; {len(obj_clouds)} object cloud(s) "
          f"[{', '.join(str(c.n_points) for c in obj_clouds) or '-'}]; precomputing {F} frames ...")
    med = np.empty(F); p95 = np.empty(F)
    for i, t in enumerate(frames):
        v = body.posed_vertices(params, t)                          # (V,3) avance SMPL complète (réf parité)
        ref = np.einsum("nij,ni->nj", v[tri_v], sampling.bary.astype(np.float64))
        pc = pose_cloud(human, *body.bone_transforms(params, t))    # (N,3) le chemin d'exécution sans-maille
        err = np.linalg.norm(pc - ref, axis=1)
        verts[i], posed[i], colors[i] = v, pc, _heat(err, vmax)
        med[i], p95[i] = np.median(err), np.percentile(err, 95)
        for k, c in enumerate(obj_clouds):
            obj_posed[k][i] = _object_world(c, raw.object_poses_raw[k][t])
    print(f"parity over clip: median {med.mean()*1000:.1f}mm, p95 {p95.mean()*1000:.1f}mm")

    srv = viser.ViserServer(port=port)
    srv.scene.add_grid("/grid", width=4.0, height=4.0)
    with srv.gui.add_folder("Playback"):
        sld = srv.gui.add_slider("frame", 0, F - 1, 1, 0)
        play = srv.gui.add_checkbox("play", False)
        fps = srv.gui.add_number("fps", 20, min=1, max=120, step=1)
    with srv.gui.add_folder("Display"):
        show_human = srv.gui.add_checkbox("human cloud", True)
        show_objs = srv.gui.add_checkbox("object clouds", True)
        show_mesh = srv.gui.add_checkbox("SMPL surface (ghost)", False)
        size = srv.gui.add_number("point size", 0.012, min=0.002, max=0.05, step=0.002)
    info = srv.gui.add_markdown("")

    hum_h = srv.scene.add_point_cloud("/human", posed[0], colors[0], point_size=float(size.value))
    obj_h = [srv.scene.add_point_cloud(f"/obj{k}", op[0], np.tile([[255, 140, 0]], (op.shape[1], 1)).astype(np.uint8),
                                       point_size=float(size.value)) for k, op in enumerate(obj_posed)]

    def render(_=None):
        f = int(sld.value)
        hum_h.points, hum_h.colors, hum_h.point_size = posed[f], colors[f], float(size.value)
        hum_h.visible = show_human.value
        for k, h in enumerate(obj_h):
            h.points, h.point_size, h.visible = obj_posed[k][f], float(size.value), show_objs.value
        if show_mesh.value:
            srv.scene.add_mesh_simple("/ghost", verts[f], body.faces, color=(200, 200, 210),
                                      opacity=0.4, side="double")
        else:
            srv.scene.add_mesh_simple("/ghost", np.zeros((3, 3), np.float32), np.array([[0, 1, 2]]), opacity=0.0)
        info.content = (f"**frame {frames[f]}** ({f + 1}/{F})\n\n"
                        f"err parité humain — médiane **{med[f]*1000:.1f}mm**, p95 **{p95[f]*1000:.1f}mm**\n\n"
                        f"couleur humain : bleu 0 → rouge ≥ {vmax*1000:.0f}mm · objets : orange (rigide K=1)")

    for h in (sld, show_human, show_objs, show_mesh, size):
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
    ap.add_argument("--corr", type=Path, default=_DEFAULT_CORR, help="correspondence cache (.npz)")
    a = ap.parse_args()
    spec = scene_from_args(a)
    view_cloud(spec, a.corr, port=a.port, frame_step=a.frame_step, max_frames=a.max_frames)


if __name__ == "__main__":
    main()
