"""Sample a robot's surface, per link, in a given rest configuration (offline, for the OT build).

Robot-agnostic: it takes a URDF path and the rest-pose joint angles as inputs (the robot-specific
angles live in ``prepare/load/robot.py``, keyed by robot name). Each sampled point is stored in the
frame of the link it belongs to (``offset_local``), so re-posing the robot by FK relocates it for
free. A T-pose-like rest (limbs spread) keeps the limb clouds well separated for the per-segment OT.

The robot geometry + kinematics come from **pinocchio** (the single robot model of the project — the
same engine as ``prepare/load/robot.PinRobot``): the visual ``GeometryModel`` gives, per geometry, the
owning link (``parentFrame``, authoritative — no scene-graph heuristic), the rest-pose world placement
(``updateGeometryPlacements``), and the mesh file (or a primitive shape). Only the mesh sampling stays
on ``trimesh``.

Each link mesh is first reduced to its watertight OUTER SHELL so samples land only on the true
outside: raw visual meshes are non-watertight and carry internal geometry, so plain surface sampling
would land many points inside the body.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .segments import link_to_segment, seg_index


@dataclass(frozen=True)
class RobotSurface:
    points_world: np.ndarray   # (M, 3) sampled points, rest pose, world frame
    link_idx: np.ndarray       # (M,)   index into link_names
    offset_local: np.ndarray   # (M, 3) point in its link's frame
    seg: np.ndarray            # (M,)   segment index (into segments.SEGMENTS)
    link_names: tuple[str, ...]


def _rest_q(model, rest_angles: dict[str, float]) -> np.ndarray:
    """Free-flyer neutral config with the named actuated joints set to ``rest_angles`` (absent -> 0)."""
    q = np.asarray(__import__("pinocchio").neutral(model), np.float64)
    qadr = {model.names[j]: model.joints[j].idx_q for j in range(2, model.njoints)}
    for name, a in rest_angles.items():
        if name in qadr:
            q[qadr[name]] = float(a)
    return q


def _geometry_mesh(geom_obj):
    """A ``trimesh.Trimesh`` for one ``GeometryObject``: its mesh file, or a primitive (sphere / box /
    cylinder / capsule) rebuilt from the coal shape. Returns ``None`` for an unsupported shape."""
    import trimesh
    mp = geom_obj.meshPath or ""
    if mp and Path(mp).suffix.lower() in (".obj", ".stl", ".dae", ".ply"):
        mesh = trimesh.load(mp, force="mesh", process=False)
        s = np.asarray(geom_obj.meshScale, np.float64)
        if not np.allclose(s, 1.0):
            mesh.apply_scale(s)
        return mesh
    g = geom_obj.geometry                                            # coal primitive
    name = type(g).__name__
    if name == "Sphere":
        return trimesh.creation.icosphere(subdivisions=2, radius=float(g.radius))
    if name == "Box":
        return trimesh.creation.box(extents=2.0 * np.asarray(g.halfSide, np.float64))
    if name in ("Cylinder", "Capsule"):
        return trimesh.creation.cylinder(radius=float(g.radius), height=2.0 * float(g.halfLength))
    return None


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
                         density: float) -> RobotSurface:
    """Surface-sample every link mesh at ``density`` pts/m^2, posed in the rest config.

    Each mesh is reduced to its watertight outer shell (samples land only on the true outside), then
    each sample is snapped back onto the real mesh (the shell is dilated ~5 mm by voxelisation, so
    sampling it directly would float the points off the surface) and pushed out 1 mm. Geometry +
    placements come from pinocchio's visual ``GeometryModel`` (mesh sampling stays trimesh)."""
    import trimesh
    import pinocchio as pin

    urdf_path = Path(urdf_path)
    model = pin.buildModelFromUrdf(str(urdf_path), pin.JointModelFreeFlyer())
    geom = pin.buildGeomFromUrdf(model, str(urdf_path), pin.GeometryType.VISUAL,
                                 package_dirs=[str(urdf_path.parent)])
    data, gdata = model.createData(), geom.createData()
    q = _rest_q(model, rest_angles)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    pin.updateGeometryPlacements(model, data, geom, gdata)

    link_names: list[str] = []
    link_id: dict[str, int] = {}
    pw, li, ol, sg = [], [], [], []
    rng = np.random.default_rng(0)                       # deterministic sampling

    for i, go in enumerate(geom.geometryObjects):
        mesh = _geometry_mesh(go)
        if mesh is None or len(mesh.vertices) == 0:
            continue
        link = model.frames[go.parentFrame].name        # authoritative owning link (no heuristic)

        shell = _outer_shell(mesh)
        n = max(1, int(shell.area * density))
        shell_pts, _ = trimesh.sample.sample_surface(shell, n, seed=int(rng.integers(1 << 31)))
        if len(shell_pts) == 0:
            continue
        pts_file, _d, face_idx = trimesh.proximity.closest_point(mesh, shell_pts)
        if len(pts_file) == 0:
            continue
        pts_file = pts_file + mesh.face_normals[face_idx] * 0.001     # push out 1 mm

        oMg = gdata.oMg[i]                                            # geometry world placement
        pts_world = pts_file @ np.asarray(oMg.rotation).T + np.asarray(oMg.translation)
        oMf = data.oMf[go.parentFrame]                               # link world placement
        offset = (pts_world - np.asarray(oMf.translation)) @ np.asarray(oMf.rotation)

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
