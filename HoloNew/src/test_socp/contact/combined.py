# src/contact/combined.py
"""Per-frame four-channel contact field driver."""
from __future__ import annotations

import numpy as np

from HoloNew.src.test_socp.correspondence.human_body import HumanBody
from .backends.coal import build_bvh, refit_bvh, surface_field_kdtree
from .backends.floor import floor_field
from .backends.sdf import sdf_surface_field
from .contact_field import ContactField, _stack, _to_object_frame

# Per-process frame context. Set once by _init_frame_ctx and reused across every frame,
# so the static object BVH, the floor/object KD-trees and the human body model are built
# only once. This single-init-then-stream shape is what a future real-time path would use:
# init the context, then call _frame_fields(...) as each new frame arrives.
_FRAME_CTX: dict = {}


def _init_frame_ctx(
    human_faces, object_mesh, object_grid_local, floor_grid, margin, fn,
    human_body_params: dict | None = None,
    human_pc_cache=None,
    object_sdf=None,
):
    from scipy.spatial import cKDTree
    ctx = {
        "human_faces": human_faces, "object_mesh": object_mesh,
        "object_grid_local": object_grid_local, "floor_grid": floor_grid,
        "margin": margin, "fn": fn,
        # SDF mode samples the object SDF instead, so the object BVH (a Coal build) is skipped.
        "obj_bvh": (build_bvh(np.asarray(object_mesh.vertices), np.asarray(object_mesh.faces))
                    if (object_mesh is not None and object_sdf is None) else None),
        "floor_tree": cKDTree(floor_grid) if floor_grid is not None else None,
        "obj_grid_tree": cKDTree(object_grid_local) if object_grid_local is not None else None,
        "pc_cache": human_pc_cache,
        "object_sdf": object_sdf,
    }
    if human_body_params is not None:
        ctx["human_body"] = HumanBody(**human_body_params)

    _FRAME_CTX.update(ctx)


def _frame_fields(payload):
    """Compute one frame's channel fields against the shared _FRAME_CTX;
    returns (t, human_floor, floor_human, human_object|None, object_human|None).

    In SDF mode (an object_sdf is supplied) only the human-side channels are produced —
    human_floor (analytic) and human_object (sampled from the SDF) — and floor_human /
    object_human are None. Those reverse channels target the deforming human mesh, so they
    can only be answered by Coal; dropping them keeps the per-frame path entirely Coal-free."""
    import trimesh as _trimesh
    # payload: (t, quat, pelvis, obj_pose, [hverts, [hprobes]])
    t, quat, pelvis, obj_pose = payload[:4]
    ctx = _FRAME_CTX
    faces, margin, fn = ctx["human_faces"], ctx["margin"], ctx["fn"]

    # Compute human vertices and probes locally in the worker to parallelize SMPL-X.
    body = ctx.get("human_body")
    if body is not None:
        hverts = body.placed_verts(quat, pelvis, frame_idx=t)
        hprobes = body.placed_points(quat, pelvis, ctx["pc_cache"], frame_idx=t)
    else:
        # Fallback for tests: vertices and probes provided in the payload.
        hverts = payload[4]
        hprobes = payload[5] if len(payload) > 5 else payload[4]

    hum_flr = floor_field(hprobes, margin)  # analytic, no Coal

    sdf = ctx.get("object_sdf")
    if sdf is not None:
        # SDF mode: human-side channels only. Sample the precomputed object SDF in the
        # object-local frame; no human BVH and no reverse channels, so no Coal at all.
        hprobes_local = _to_object_frame(hprobes, obj_pose)
        hum_obj = sdf_surface_field(hprobes_local, sdf, margin)
        return t, hum_flr, None, hum_obj, None

    # Coal mode: full four channels. World-frame human: reuse/refit the BVH and mesh.
    hbvh = ctx.get("hbvh")
    hmesh = ctx.get("hmesh")
    if hbvh is None or hbvh.num_vertices != len(hverts):
        hbvh = build_bvh(hverts, faces)
        hmesh = _trimesh.Trimesh(vertices=hverts, faces=faces, process=False)
        ctx["hbvh"], ctx["hmesh"] = hbvh, hmesh
    else:
        refit_bvh(hbvh, hverts)
        hmesh.vertices = hverts

    # Channel: floor probes vs human. floor_grid is static -> use cached floor_tree.
    flr_hum = fn(ctx["floor_grid"], hbvh, hmesh, margin, tree=ctx["floor_tree"])

    hum_obj = obj_hum = None
    if ctx["obj_bvh"] is not None:
        # Query both object-channel directions in the OBJECT-LOCAL frame.
        hverts_local = _to_object_frame(hverts, obj_pose)

        # Object-frame human: reuse/refit the local representation.
        hbvh_local = ctx.get("hbvh_local")
        hmesh_local = ctx.get("hmesh_local")
        if hbvh_local is None or hbvh_local.num_vertices != len(hverts_local):
            hbvh_local = build_bvh(hverts_local, faces)
            hmesh_local = _trimesh.Trimesh(vertices=hverts_local, faces=faces, process=False)
            ctx["hbvh_local"], ctx["hmesh_local"] = hbvh_local, hmesh_local
        else:
            refit_bvh(hbvh_local, hverts_local)
            hmesh_local.vertices = hverts_local

        hprobes_local = _to_object_frame(hprobes, obj_pose)
        # Channel: human probes vs object. hprobes_local changes -> no cached tree.
        hum_obj = fn(hprobes_local, ctx["obj_bvh"], ctx["object_mesh"], margin)
        # Channel: object probes vs human. object_grid_local is static -> use cached tree.
        obj_hum = fn(ctx["object_grid_local"], hbvh_local, hmesh_local, margin, tree=ctx["obj_grid_tree"])

    return t, hum_flr, flr_hum, hum_obj, obj_hum


def compute_contact_fields(
    T: int,
    quats: np.ndarray,      # (T, 52, 4) wxyz global orientations
    pelvises: np.ndarray,   # (T, 3) world pelvis positions
    human_faces: np.ndarray,
    human_body_params: dict | None,        # params to instantiate HumanBody once in the frame ctx
    human_pc_cache,         # PointCloudCache for stable surface sampling
    object_mesh,            # trimesh in object-local frame (or None)
    object_grid_local: np.ndarray | None,  # (N, 3) object-frame probes (or None)
    obj_poses: np.ndarray,  # (T, 7) [qw,qx,qy,qz,x,y,z]
    floor_grid: np.ndarray, # (N, 3) world-frame floor probes
    margin: float,
    fn=surface_field_kdtree,
    hverts: np.ndarray | None = None,  # (T, V, 3) optional precomputed vertices (for tests)
    hprobes: np.ndarray | None = None, # (T, N_h, 3) optional precomputed probes (for tests)
    object_sdf=None,                   # precomputed ObjectSDF; routes the object channel off Coal
    progress: bool = False,            # show a per-frame tqdm bar with the live throughput (fps)
) -> dict[str, ContactField]:
    """Stack the channel fields over T frames. Object channels are skipped (omitted from
    the dict) when no object_mesh is provided.

    Two modes:
    - Coal (default): all four channels — human_floor, floor_human, human_object,
      object_human.
    - SDF (object_sdf supplied): only the human-side channels human_floor (analytic) and
      human_object (sampled from the SDF). The reverse channels target the deforming human
      mesh and need Coal, so they are dropped to keep the per-frame path Coal-free.

    Both the human mesh deformation (SMPL-X skinning) and the geometric queries
    (BVH builds + Coal distance) run in a single linear pass over the frames, t=0..T-1.
    No parallelism: the context is built once, then each frame is solved in order, which
    keeps the path real-time-friendly (the same _frame_fields call can later be driven by
    a live frame stream).
    """
    if T < 1:
        raise ValueError(f"compute_contact_fields needs T >= 1, got {T}")

    # Build the shared frame context once (object BVH, KD-trees, HumanBody, SDF).
    _init_frame_ctx(
        human_faces, object_mesh, object_grid_local, floor_grid, margin, fn,
        human_body_params, human_pc_cache, object_sdf,
    )

    results: list = [None] * T
    # Linear pass, frame by frame from t=0 to T-1. The optional tqdm bar reports the live
    # throughput as frame/s — i.e. the real-time fps we currently sustain on this pass.
    frame_iter = range(T)
    if progress:
        from tqdm import tqdm
        frame_iter = tqdm(frame_iter, desc="  Contact field", unit="frame", leave=False)
    for t in frame_iter:
        payload = [t, quats[t], pelvises[t], np.asarray(obj_poses[t])]
        if hverts is not None:
            payload.append(hverts[t])
        if hprobes is not None:
            payload.append(hprobes[t])
        _, *fields = _frame_fields(tuple(payload))
        results[t] = fields

    out = {"human_floor": _stack([results[t][0] for t in range(T)])}
    if object_sdf is not None:
        # SDF mode: only the human-side channels (no Coal reverse channels).
        out["human_object"] = _stack([results[t][2] for t in range(T)])
    else:
        out["floor_human"] = _stack([results[t][1] for t in range(T)])
        if object_mesh is not None:
            out["human_object"] = _stack([results[t][2] for t in range(T)])
            out["object_human"] = _stack([results[t][3] for t in range(T)])
    return out
