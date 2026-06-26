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
from scipy.spatial.transform import Rotation as _Rot

from ..contracts import CalibrationConfig, RobotSpec, SceneSpec
from ..prepare.calibration import build_calibration
from ..prepare.load import load
from ..prepare.load.smpl import build_body_model


def _object_world_lowz(vl: np.ndarray, poses: np.ndarray, cap: int = 8000):
    """Per-frame lowest WORLD point of a rigid object posed by ``poses`` (F,7) pos-first wxyz.

    Returns ``(min_z (F,), low_point (F,3))``. The vertex set ``vl`` (V,3 local) is subsampled to
    ``cap`` to bound cost on dense scans — a near-exact lowest point, enough for a debug marker."""
    v = vl
    if vl.shape[0] > cap:
        v = vl[np.random.default_rng(0).choice(vl.shape[0], cap, replace=False)]
    rot = _Rot.from_quat(poses[:, [4, 5, 6, 3]]).as_matrix()          # wxyz -> xyzw
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

    # objects are rigid: keep local mesh + per-frame world pose, and update the transform per frame.
    objs = []  # (verts_local, faces, poses_frames (F, 7))
    for k in range(len(raw.object_poses_raw)):
        m = trimesh.load(str(raw.object_mesh_paths[k]), force="mesh", process=False, skip_materials=True)
        vl = np.asarray(m.vertices, np.float32)
        fl = np.asarray(m.faces, np.int32)
        poses = np.asarray(raw.object_poses_raw[k], np.float32)[frames]
        objs.append((vl, fl, poses))

    # skeleton segments (parent -> child)
    bone_pairs = [(int(parents[j]), j) for j in range(body.n_bones) if parents[j] >= 0] if body else []

    # --- grounding debug: the calibration + per-frame floor clearances (RAW, pre-grounding) ---
    # The whole scene is grounded by a SINGLE z-shift = calib.floor_offset (driven by the human
    # sole). These clearances let us SEE whether the human soles AND each object share that floor:
    # if the per-clip medians of human_minz and obj_minz disagree, that scene needs a DIFFERENT
    # offset for the object than for the human (independent placement) rather than one shared shift.
    calib = build_calibration(raw, CalibrationConfig(), body=body)
    floor_offset = float(calib.floor_offset)
    # Human lowest world z + lowest point per frame (surface if parametric, else demo joints).
    src = verts if verts is not None else demo_j
    human_minz = src[:, :, 2].min(axis=1)                               # (F,)
    human_low = src[np.arange(F), src[:, :, 2].argmin(axis=1)]          # (F, 3)
    obj_minz, obj_low = [], []                                          # per object: (F,), (F, 3)
    for (vl, _fl, poses) in objs:
        mz, lp = _object_world_lowz(vl, poses)
        obj_minz.append(mz); obj_low.append(lp)
    hz_med = float(np.median(human_minz))
    oz_med = [float(np.median(z)) for z in obj_minz]
    print(f"calibration: floor_offset={floor_offset:+.4f} m, human_stature={calib.human_stature:.3f} m")
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
        show_floor = srv.gui.add_checkbox("floor plane z=0", True)
        show_low = srv.gui.add_checkbox("lowest-point markers", True)
    info = srv.gui.add_markdown("")

    floor_h = srv.scene.add_box("/floor", color=(170, 170, 178), dimensions=(4.0, 4.0, 0.004),
                                position=(0.0, 0.0, 0.0))
    hj = srv.scene.add_point_cloud("/joints", demo_j[0], np.tile([[40, 200, 60]], (n_demo, 1)).astype(np.uint8),
                                   point_size=0.025)
    hobj = [srv.scene.add_mesh_simple(f"/obj{k}", vl, fl, color=(255, 140, 0), side="double")
            for k, (vl, fl, _) in enumerate(objs)]
    # lowest-point markers: red = human sole, yellow = each object — the two things grounding must
    # rest on z=0. Watching them separately is exactly how we tell a shared offset from a split one.
    low_h = srv.scene.add_point_cloud("/low_human", human_low[:1], np.array([[255, 40, 40]], np.uint8),
                                      point_size=0.05)
    low_o = [srv.scene.add_point_cloud(f"/low_obj{k}", obj_low[k][:1], np.array([[255, 210, 0]], np.uint8),
                                       point_size=0.05) for k in range(len(objs))]

    def render(_=None):
        f = int(sld.value)
        g = floor_offset if apply_ground.value else 0.0      # grounding = shift the scene down to z=0
        dz = np.array([0.0, 0.0, g], np.float32)
        if body is not None and show_mesh.value:
            srv.scene.add_mesh_simple("/body", verts[f] - dz, faces, color=(200, 200, 210),
                                      opacity=0.55, side="double")
        else:
            srv.scene.add_mesh_simple("/body", np.zeros((3, 3), np.float32), np.array([[0, 1, 2]]), opacity=0.0)
        hj.points = demo_j[f] - dz
        hj.visible = show_joints.value
        if body is not None and show_bones.value and bone_pairs:
            seg = np.stack([np.stack([bones[f, a], bones[f, b]]) for a, b in bone_pairs]).astype(np.float32) - dz
            srv.scene.add_line_segments("/skeleton", seg,
                                        np.tile([[[0, 120, 255]]], (len(bone_pairs), 2, 1)).astype(np.uint8),
                                        line_width=3.0)
        else:
            srv.scene.add_line_segments("/skeleton", np.zeros((1, 2, 3), np.float32),
                                        np.zeros((1, 2, 3), np.uint8), line_width=0.1)
        for k, (_, _, poses) in enumerate(objs):
            h = hobj[k]
            h.position = poses[f][:3] - dz
            h.wxyz = poses[f][3:]
            h.visible = show_obj.value
        floor_h.visible = show_floor.value
        low_h.points = (human_low[f] - dz)[None]
        low_h.visible = show_low.value
        for k in range(len(objs)):
            low_o[k].points = (obj_low[k][f] - dz)[None]
            low_o[k].visible = show_low.value

        hz = human_minz[f] - g
        oz = [obj_minz[k][f] - g for k in range(len(objs))]
        info.content = (
            f"**frame {frames[f]}** ({f + 1}/{F}) · grounding **{'ON' if g else 'OFF'}** "
            f"(floor_offset = {floor_offset:+.4f} m)\n\n"
            f"lowest z (this frame) — human **{hz:+.4f}**" +
            "".join(f", obj{k} **{z:+.4f}**" for k, z in enumerate(oz)) + " m  (want ≈ 0)\n\n"
            f"lowest z (clip median) — human **{hz_med - g:+.4f}**" +
            "".join(f", obj{k} **{m - g:+.4f}**" for k, m in enumerate(oz_med)) +
            " m  ← human vs obj differ ⇒ they need separate offsets")

    for h in (sld, show_mesh, show_joints, show_bones, show_obj, apply_ground, show_floor, show_low):
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
    ap.add_argument("--dataset-root", type=Path, default=None,
                    help="release root for auxiliary metadata (OMOMO betas/scales + meshes)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--frame-step", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=200)
    a = ap.parse_args()
    robot = RobotSpec(name="g1", urdf_path=Path("g1.urdf"), link_names=("pelvis",), dof=29, height=1.3)
    spec = SceneSpec(dataset=a.dataset, motion_path=a.motion_path, robot=robot,
                     smpl_model_dir=a.model_dir, dataset_root=a.dataset_root)
    view_scene(spec, port=a.port, frame_step=a.frame_step, max_frames=a.max_frames)


if __name__ == "__main__":
    main()
