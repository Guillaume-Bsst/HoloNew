"""Échantillonne la surface d'un robot, par lien, dans une configuration au repos donnée (hors ligne,
pour la construction OT).

Indépendant du robot : il prend un chemin URDF et les angles articulaires au repos comme entrées
(les angles spécifiques au robot vivent dans ``prepare/load/robot.py``, indexés par le nom du
robot). Chaque point échantillonné est stocké dans le repère du lien auquel il appartient
(``offset_local``), donc re-poser le robot par FK le relocalise gratuitement. Un repos de type
T-pose (membres écartés) garde les nuages de membres bien séparés pour l'OT par segment.

La géométrie du robot + cinématique proviennent de **pinocchio** (le modèle de robot unique du
projet — le même moteur que ``prepare/load/robot.PinRobot``) : le ``GeometryModel`` visuel donne,
par géométrie, le lien propriétaire (``parentFrame``, faisant autorité — pas d'heuristique de
graphe de scène), le placement mondial au repos (``updateGeometryPlacements``), et le fichier de
maillage (ou une forme primitive). Seul l'échantillonnage du maillage reste sur ``trimesh``.

Chaque maillage de lien est d'abord réduit à sa COQUE EXTERNE étanche pour que les échantillons
ne touchent que le véritable extérieur : les maillages visuels bruts ne sont pas étanches et portent
de la géométrie interne, donc l'échantillonnage de surface ordinaire atterrirait beaucoup de points
à l'intérieur du corps.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .segments import link_to_segment, seg_index


@dataclass(frozen=True)
class RobotSurface:
    points_world: np.ndarray   # (M, 3) sampled points, rest pose, world frame
    link_idx: np.ndarray       # (M,)   index dans link_names
    offset_local: np.ndarray   # (M, 3) point in its link's frame
    seg: np.ndarray            # (M,)   index de segment (dans segments.SEGMENTS)
    link_names: tuple[str, ...]


def _rest_q(model, rest_angles: dict[str, float]) -> np.ndarray:
    """Config neutre free-flyer avec les articulations commandées nommées définies à ``rest_angles`` (absent → 0)."""
    q = np.asarray(__import__("pinocchio").neutral(model), np.float64)
    qadr = {model.names[j]: model.joints[j].idx_q for j in range(2, model.njoints)}
    for name, a in rest_angles.items():
        if name in qadr:
            q[qadr[name]] = float(a)
    return q


def _geometry_mesh(geom_obj):
    """Un ``trimesh.Trimesh`` pour un ``GeometryObject`` : son fichier de maillage, ou une primitive
    (sphère / boîte / cylindre / capsule) reconstruite à partir de la forme coal. Retourne ``None``
    pour une forme non supportée."""
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
    """Coque extérieure étanche d'un maillage (possiblement soupe de triangles) via voxélisation →
    remplissage → marching cubes, gardant uniquement le vrai extérieur et fermant les trous. Le pas
    s'adapte à l'étendue la plus fine pour que les petits liens survivent ; revient au maillage
    d'entrée si le remaillage est dégénéré."""
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
    """Échantillonne la surface de chaque maillage de lien à ``density`` pts/m^2, posé dans la
    configuration au repos.

    Chaque maillage est réduit à sa coque extérieure étanche (les échantillons ne touchent que le
    vrai extérieur), puis chaque échantillon est attaché au vrai maillage (la coque est dilatée
    ~5 mm par voxélisation, donc l'échantillonner directement ferait flotter les points de la
    surface) et poussé de 1 mm. La géométrie + placements proviennent du ``GeometryModel`` visuel
    de pinocchio (l'échantillonnage du maillage reste trimesh)."""
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
    rng = np.random.default_rng(0)                       # échantillonnage déterministe

    for i, go in enumerate(geom.geometryObjects):
        mesh = _geometry_mesh(go)
        if mesh is None or len(mesh.vertices) == 0:
            continue
        link = model.frames[go.parentFrame].name        # lien propriétaire faisant autorité (pas d'heuristique)

        shell = _outer_shell(mesh)
        n = max(1, int(shell.area * density))
        shell_pts, _ = trimesh.sample.sample_surface(shell, n, seed=int(rng.integers(1 << 31)))
        if len(shell_pts) == 0:
            continue
        pts_file, _d, face_idx = trimesh.proximity.closest_point(mesh, shell_pts)
        if len(pts_file) == 0:
            continue
        pts_file = pts_file + mesh.face_normals[face_idx] * 0.001     # push out 1 mm

        oMg = gdata.oMg[i]                                            # placement mondial de la géométrie
        pts_world = pts_file @ np.asarray(oMg.rotation).T + np.asarray(oMg.translation)
        oMf = data.oMf[go.parentFrame]                               # placement mondial du lien
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
