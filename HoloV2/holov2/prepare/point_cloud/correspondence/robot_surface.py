"""Sample a robot's surface, per link, in a given rest configuration (offline, for the OT build).

Robot-agnostic: it takes a URDF path and the rest-pose joint angles as inputs (the robot-specific
angles live in ``prepare/load/robot.py``, keyed by robot name). Each sampled point is stored in the
frame of the link it belongs to (``offset_local``), so re-posing the robot by FK relocates it for
free. A T-pose-like rest (limbs spread) keeps the limb clouds well separated for the per-segment OT.

Each link mesh is first reduced to its watertight OUTER SHELL so samples land only on the true
outside: raw visual meshes are non-watertight and carry internal geometry, so plain surface sampling
would land many points inside the body.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .segments import link_to_segment, seg_index

# Robot surface sampling density (pts / m^2) — a build-time constant of the correspondence, not a
# per-run knob, so it stays here rather than in the config.
SURFACE_DENSITY = 3000.0


@dataclass(frozen=True)
class RobotSurface:
    points_world: np.ndarray   # (M, 3) sampled points, rest pose, world frame
    link_idx: np.ndarray       # (M,)   index into link_names
    offset_local: np.ndarray   # (M, 3) point in its link's frame
    seg: np.ndarray            # (M,)   segment index (into segments.SEGMENTS)
    link_names: tuple[str, ...]


def _rest_cfg(actuated_joint_names, rest_angles: dict[str, float]) -> np.ndarray:
    """Actuated-joint vector (n,) in URDF order; joints absent from ``rest_angles`` default to 0."""
    return np.array([rest_angles.get(name, 0.0) for name in actuated_joint_names], dtype=np.float64)


def _outer_shell(mesh):
    """Watertight outer shell of a (possibly triangle-soup) mesh via voxelise -> fill -> marching
    cubes, keeping only the true outside and closing holes. Pitch adapts to the thinnest extent so
    small links survive; falls back to the input mesh if the remesh is degenerate."""
    ext = mesh.bounds[1] - mesh.bounds[0]
    pitch = float(np.clip(ext.min() / 6.0, 0.003, 0.008))
    try:
        vox = mesh.voxelized(pitch=pitch).fill()
        shell = vox.marching_cubes
        if len(shell.vertices) == 0:
            return mesh
        shell.apply_transform(vox.transform)            # voxel index frame -> mesh frame
        if not shell.is_watertight or shell.area <= 0:
            return mesh
    except Exception:
        return mesh
    return shell


def sample_robot_surface(urdf_path: Path, rest_angles: dict[str, float],
                         density: float = SURFACE_DENSITY) -> RobotSurface:
    """Surface-sample every link mesh at ``density`` pts/m^2, posed in the rest config.

    Each mesh is reduced to its watertight outer shell (samples land only on the true outside), then
    each sample is snapped back onto the real mesh (the shell is dilated ~5 mm by voxelisation, so
    sampling it directly would float the points off the surface) and pushed out 1 mm."""
    import trimesh
    import yourdfpy

    urdf = yourdfpy.URDF.load(str(urdf_path), load_meshes=True, build_scene_graph=True)
    urdf.update_cfg(_rest_cfg(urdf.actuated_joint_names, rest_angles))
    scene = urdf.scene

    link_names: list[str] = []
    link_id: dict[str, int] = {}
    pw, li, ol, sg = [], [], [], []
    rng = np.random.default_rng(0)                       # deterministic sampling

    for node in scene.graph.nodes_geometry:
        t_geom, geom_name = scene.graph.get(node)
        mesh = scene.geometry[geom_name]

        link = node                                      # find the true owning URDF link
        while link not in urdf.link_map and link in scene.graph.transforms.parents:
            link = scene.graph.transforms.parents[link]
        if link not in urdf.link_map:
            link = next((ln for ln in urdf.link_map if ln in node), None) or (
                "pelvis" if "pelvis" in urdf.link_map else next(iter(urdf.link_map)))

        shell = _outer_shell(mesh)
        n = max(1, int(shell.area * density))
        shell_pts, _ = trimesh.sample.sample_surface(shell, n, seed=int(rng.integers(1 << 31)))
        if len(shell_pts) == 0:
            continue
        pts_file, _d, face_idx = trimesh.proximity.closest_point(mesh, shell_pts)
        if len(pts_file) == 0:
            continue
        pts_file = pts_file + mesh.face_normals[face_idx] * 0.001     # push out 1 mm

        t_geom = np.asarray(t_geom)
        pts_world = pts_file @ t_geom[:3, :3].T + t_geom[:3, 3]
        t_link = np.asarray(urdf.get_transform(link))
        offset = (pts_world - t_link[:3, 3]) @ t_link[:3, :3]         # into the link frame

        if link not in link_id:
            link_id[link] = len(link_names)
            link_names.append(link)
        idx = link_id[link]
        s = seg_index(link_to_segment(link))
        pw.append(pts_world.astype(np.float32))
        li.append(np.full(len(pts_world), idx, dtype=np.int64))
        ol.append(offset.astype(np.float32))
        sg.append(np.full(len(pts_world), s, dtype=np.int64))

    return RobotSurface(points_world=np.concatenate(pw), link_idx=np.concatenate(li),
                        offset_local=np.concatenate(ol), seg=np.concatenate(sg),
                        link_names=tuple(link_names))
