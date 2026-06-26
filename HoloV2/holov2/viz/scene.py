"""Scene preview viewer — visual debug of the ``load`` stage.

Given a ``SceneSpec`` it loads the ``RawMotion`` + builds the ``BodyModel`` and shows, per frame,
the posed SMPL-X mesh, the skeleton (FK bones + demo joints), the object(s) posed by their world
poses, and the ground. Pure consumer (drives ``load`` to get artifacts; no compute hooks). The
full ``FrameTrace`` viewer (``viewer.py``) comes later, once ``targets`` exists.

Run:
    python -m holov2.viz.scene --motion-path <smplx.npz> --model-dir <smplx_models> [--dataset hodome]
"""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..contracts import RobotSpec, SceneSpec
from ..prepare.load import load
from ..prepare.load.smpl import build_body_model


def _pose_object(verts_local: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Rigid-transform object verts by a (7,) world pose [x,y,z,qw,qx,qy,qz]."""
    pos, quat_wxyz = pose[:3], pose[3:]
    rot = R.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_matrix()
    return verts_local @ rot.T + pos


def view_scene(spec: SceneSpec, *, port: int = 8080, frame_step: int = 2, max_frames: int = 200) -> None:
    import trimesh
    import viser

    raw = load(spec)
    frames = list(range(0, raw.n_frames, frame_step))[:max_frames]
    F = len(frames)
    print(f"loaded {raw.source_format}: T={raw.n_frames}, showing {F} frames, "
          f"{len(raw.object_poses_raw)} object(s), parametric={raw.is_parametric}")

    # --- precompute per shown frame (bounded) ---
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

    # object meshes (local) + posed world verts per shown frame
    obj_meshes = []  # (faces, world_verts (F, Vo, 3))
    for k in range(len(raw.object_poses_raw)):
        m = trimesh.load(str(raw.object_mesh_paths[k]), force="mesh", process=False)
        vl, fo = np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int64)
        wv = np.stack([_pose_object(vl, raw.object_poses_raw[k][t]) for t in frames]).astype(np.float32)
        obj_meshes.append((fo, wv))

    # skeleton segments (parent -> child)
    bone_pairs = [(int(parents[j]), j) for j in range(body.n_bones) if parents[j] >= 0] if body else []

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
    info = srv.gui.add_markdown("")

    hj = srv.scene.add_point_cloud("/joints", demo_j[0], np.tile([[40, 200, 60]], (n_demo, 1)).astype(np.uint8),
                                   point_size=0.025)

    def render(_=None):
        f = int(sld.value)
        if body is not None and show_mesh.value:
            srv.scene.add_mesh_simple("/body", verts[f], faces, color=(200, 200, 210), opacity=0.55)
        else:
            srv.scene.add_mesh_simple("/body", np.zeros((3, 3), np.float32), np.array([[0, 1, 2]]), opacity=0.0)
        hj.points = demo_j[f]
        hj.visible = show_joints.value
        if body is not None and show_bones.value and bone_pairs:
            seg = np.stack([np.stack([bones[f, a], bones[f, b]]) for a, b in bone_pairs]).astype(np.float32)
            srv.scene.add_line_segments("/skeleton", seg,
                                        np.tile([[[0, 120, 255]]], (len(bone_pairs), 2, 1)).astype(np.uint8),
                                        line_width=3.0)
        else:
            srv.scene.add_line_segments("/skeleton", np.zeros((1, 2, 3), np.float32),
                                        np.zeros((1, 2, 3), np.uint8), line_width=0.1)
        for k, (fo, wv) in enumerate(obj_meshes):
            if show_obj.value:
                srv.scene.add_mesh_simple(f"/obj{k}", wv[f], fo, color=(255, 140, 0), opacity=0.8)
            else:
                srv.scene.add_mesh_simple(f"/obj{k}", np.zeros((3, 3), np.float32),
                                          np.array([[0, 1, 2]]), opacity=0.0)
        info.content = f"**frame {frames[f]}** ({f + 1}/{F})"

    for h in (sld, show_mesh, show_joints, show_bones, show_obj):
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
    ap.add_argument("--dataset", default="hodome")
    ap.add_argument("--motion-path", required=True, type=Path)
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--frame-step", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=200)
    a = ap.parse_args()
    robot = RobotSpec(name="g1", urdf_path=Path("g1.urdf"), link_names=("pelvis",), dof=29, height=1.3)
    spec = SceneSpec(dataset=a.dataset, motion_path=a.motion_path, robot=robot, smpl_model_dir=a.model_dir)
    view_scene(spec, port=a.port, frame_step=a.frame_step, max_frames=a.max_frames)


if __name__ == "__main__":
    main()
