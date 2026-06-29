"""Multi-person debug view for HOI-M3 — validate the loading end-to-end.

HOI-M3 scenes have several people, each manipulating different objects; the single-human loader
keeps one person + all objects, which looks incoherent (the other objects are driven by people we
don't show). This view renders ALL people + all objects together (as the official toolbox does), so
the per-entity loading can be checked against a coherent scene. Pure consumer: it reuses the loader
(objects + one person) and ``build_person_params`` (the other people) + the SMPL-X body model.

Run:
    python -m src.viz.hoim3_multiperson --motion-path <..._human.npz> --model-dir <smplx_models>
"""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np

from ..prepare.contracts import RobotSpec, SceneSpec
from .. import paths
from ..prepare.load import load
from ..prepare.load.datasets.hoim3 import build_person_params

_PALETTE = [(70, 130, 220), (220, 90, 90), (90, 200, 120), (210, 170, 60), (170, 110, 210)]


def view(spec: SceneSpec, *, port: int = 8080, frame_step: int = 30, max_frames: int = 150) -> None:
    import trimesh
    import viser

    raw = load(spec)                                          # objects (+ one person) for free
    hd = np.load(str(spec.motion_path), allow_pickle=True)
    smpl_params = hd["smpl_params"]
    gender = str(hd["gender"])
    ids = [int(np.asarray(p["id"])) for p in smpl_params[0]]  # people present at frame 0
    frames = list(range(0, raw.n_frames, frame_step))[:max_frames]
    F = len(frames)
    print(f"HOI-M3 multi-person: {len(ids)} people {ids}, {len(raw.object_poses_raw)} objects, "
          f"showing {F} frames")

    # Per-person posed SMPL-X meshes for the shown frames (real forward).
    persons = []  # (verts (F,V,3), faces, color)
    for k, pid in enumerate(ids):
        params, body = build_person_params(smpl_params, pid, gender, Path(spec.smpl_model_dir))
        verts = np.stack([body.posed_vertices(params, t) for t in frames]).astype(np.float32)
        persons.append((verts, body.faces, _PALETTE[k % len(_PALETTE)]))
        print(f"  person {pid}: posed {F} frames")

    # Objects: centred local mesh + per-frame Z-up pose, exactly as the loader produced them.
    objs = []  # (verts_local, faces, poses (F,7))
    for kk in range(len(raw.object_poses_raw)):
        m = trimesh.load(str(raw.object_mesh_paths[kk]), force="mesh", process=False, skip_materials=True)
        objs.append((np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32),
                     np.asarray(raw.object_poses_raw[kk], np.float32)[frames]))

    print("starting viser ...")
    srv = viser.ViserServer(port=port)
    srv.scene.add_grid("/grid", width=6.0, height=6.0)
    with srv.gui.add_folder("Playback"):
        sld = srv.gui.add_slider("frame", 0, F - 1, 1, 0)
        play = srv.gui.add_checkbox("play", False)
        fps = srv.gui.add_number("fps", 20, min=1, max=120, step=1)
    with srv.gui.add_folder("Display"):
        show_people = srv.gui.add_checkbox("people", True)
        show_obj = srv.gui.add_checkbox("objects", True)
    info = srv.gui.add_markdown("")

    oh = [srv.scene.add_mesh_simple(f"/obj{k}", vl, fl, color=(255, 140, 0), side="double")
          for k, (vl, fl, _) in enumerate(objs)]

    def render(_=None):
        f = int(sld.value)
        for k, (verts, faces, color) in enumerate(persons):
            if show_people.value:
                srv.scene.add_mesh_simple(f"/person{k}", verts[f], faces, color=color, side="double")
            else:
                srv.scene.add_mesh_simple(f"/person{k}", np.zeros((3, 3), np.float32),
                                          np.array([[0, 1, 2]]), opacity=0.0)
        for k, (_, _, poses) in enumerate(objs):
            oh[k].position = poses[f][:3]
            oh[k].wxyz = poses[f][3:]
            oh[k].visible = show_obj.value
        info.content = f"**frame {frames[f]}** ({f + 1}/{F}) — {len(persons)} people"

    for h in (sld, show_people, show_obj):
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
    ap.add_argument("--motion-path", required=True, type=Path)
    ap.add_argument("--model-dir", type=Path, default=None, help="SMPL-X model dir; default: paths.toml 'smplx'")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--frame-step", type=int, default=30)
    ap.add_argument("--max-frames", type=int, default=150)
    a = ap.parse_args()
    model_dir = a.model_dir if a.model_dir is not None else paths.smplx_dir()
    robot = RobotSpec(name="g1", urdf_path=paths.HOLOV2_ROOT / "models" / "g1" / "g1_29dof.urdf",
                      link_names=("pelvis",), dof=29, height=1.3)
    spec = SceneSpec(dataset="hoim3", motion_path=a.motion_path, robot=robot, smpl_model_dir=model_dir)
    view(spec, port=a.port, frame_step=a.frame_step, max_frames=a.max_frames)


if __name__ == "__main__":
    main()
