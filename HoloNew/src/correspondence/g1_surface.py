"""Sample the G1 surface, per link, in a manual rest configuration.

Each sampled point is stored both in world (for the OT solve) and in the frame of
the link it belongs to (offset_local), so re-posing the G1 by FK relocates it for
free. The rest config is a T-pose (arms straight out to the sides) to match the
SMPL-X rest pose; the per-segment OT alignment is orientation-invariant, so exact
pose matching is not required, but a clean T-pose keeps the limb clouds well
separated. Tune via viz.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .segments import SEGMENTS, g1_link_to_segment

_SEG_IDX = {s: i for i, s in enumerate(SEGMENTS)}

# Actuated-joint angles (radians) defining the rest pose, by URDF joint name; unset
# joints = 0. T-pose: shoulder_roll abducts the arms to horizontal (+-pi/2), and
# elbow ~1.55 straightens the forearm (the G1 elbow zero is bent ~90 deg forward).
G1_REST_ANGLES: dict[str, float] = {
    "left_shoulder_roll_joint": 1.5708, "right_shoulder_roll_joint": -1.5708,
    "left_elbow_joint": 1.55, "right_elbow_joint": 1.55,
}


@dataclass(frozen=True)
class G1Surface:
    points_world: np.ndarray  # (M, 3) sampled points, rest pose, world frame
    link_idx: np.ndarray      # (M,)   index into link_names
    offset_local: np.ndarray  # (M, 3) point in its link's frame
    seg: np.ndarray           # (M,)   segment index (into segments.SEGMENTS)
    link_names: list[str]     # (L,)   link name per index


def build_rest_cfg(urdf: object) -> np.ndarray:
    """Actuated-joint vector (n,) in urdf.actuated_joint_names order for the rest pose."""
    return np.array(
        [G1_REST_ANGLES.get(name, 0.0) for name in urdf.actuated_joint_names],
        dtype=np.float64,
    )


def _outer_shell(mesh):
    """Watertight outer shell of a (possibly triangle-soup) mesh via voxelise -> fill
    -> marching cubes. The G1 visual meshes are non-watertight and carry internal
    geometry (the torso/knees/hands have 10-45% of their faces inside the outer
    surface), so plain surface sampling lands many points inside the body. The shell
    keeps only the true outside and closes holes. Pitch adapts to the thinnest extent
    so small links survive; falls back to the input mesh if the remesh is degenerate.
    """
    ext = mesh.bounds[1] - mesh.bounds[0]
    pitch = float(np.clip(ext.min() / 6.0, 0.003, 0.008))
    try:
        vox = mesh.voxelized(pitch=pitch).fill()
        shell = vox.marching_cubes
        if len(shell.vertices) == 0:
            return mesh
        shell.apply_transform(vox.transform)        # voxel index frame -> mesh frame
        if not shell.is_watertight or shell.area <= 0:
            return mesh
    except Exception:
        return mesh
    return shell


def sample_g1_surface(urdf: object, density: float) -> G1Surface:
    """Surface-sample every G1 link mesh at `density` pts/m², posed in the rest config.

    Each mesh is first reduced to its watertight outer shell so samples land only on
    the true outside (the raw G1 meshes carry internal geometry); see _outer_shell.
    """
    import trimesh

    urdf.update_cfg(build_rest_cfg(urdf))
    scene = urdf.scene

    link_names: list[str] = []
    link_id: dict[str, int] = {}
    pw_chunks, li_chunks, ol_chunks, seg_chunks = [], [], [], []

    print(f"  G1 correspondence: sampling {len(scene.graph.nodes_geometry)} meshes...")
    sampled_links = set()

    for node in scene.graph.nodes_geometry:
        T_geom, geom_name = scene.graph.get(node)
        mesh = scene.geometry[geom_name]

        # Find the true owning URDF link.
        link = node
        while link not in urdf.link_map and link in scene.graph.transforms.parents:
            link = scene.graph.transforms.parents[link]

        if link not in urdf.link_map:
            for ln in urdf.link_map:
                if ln in node:
                    link = ln
                    break

        if link not in urdf.link_map:
            link = "pelvis" if "pelvis" in urdf.link_map else list(urdf.link_map.keys())[0]

        # Sample the watertight outer shell (excludes internal geometry), then snap
        # each sample back onto the real mesh so points hug the true visual surface:
        # the shell is dilated ~5mm by voxelisation, so sampling it directly would
        # float the contact field off the mesh (visible dimples when re-posed).
        shell = _outer_shell(mesh)
        n = max(1, int(shell.area * density))
        shell_pts, _ = trimesh.sample.sample_surface(shell, n)
        if len(shell_pts) == 0:
            continue
        pts_file, _dist, face_idx = trimesh.proximity.closest_point(mesh, shell_pts)
        if len(pts_file) == 0:
            continue

        sampled_links.add(link)

        # Push points slightly outwards (e.g. 1mm).
        normals = mesh.face_normals[face_idx]
        pts_file += normals * 0.001

        T_geom = np.asarray(T_geom)
        pts_world = pts_file @ T_geom[:3, :3].T + T_geom[:3, 3]

        Tl = np.asarray(urdf.get_transform(link))
        offset = (pts_world - Tl[:3, 3]) @ Tl[:3, :3]

        if link not in link_id:
            link_id[link] = len(link_names)
            link_names.append(link)
        li = link_id[link]
        s_name = g1_link_to_segment(link)
        seg = _SEG_IDX[s_name]
        # print(f"    {link} -> {s_name}") # Debug noise

        pw_chunks.append(pts_world.astype(np.float32))
        li_chunks.append(np.full(len(pts_world), li, dtype=np.int64))
        ol_chunks.append(offset.astype(np.float32))
        seg_chunks.append(np.full(len(pts_world), seg, dtype=np.int64))

    print(f"  G1 correspondence: sampled {len(sampled_links)} links: {sorted(list(sampled_links))}")

    return G1Surface(
        points_world=np.concatenate(pw_chunks),
        link_idx=np.concatenate(li_chunks),
        offset_local=np.concatenate(ol_chunks),
        seg=np.concatenate(seg_chunks),
        link_names=link_names,
    )
