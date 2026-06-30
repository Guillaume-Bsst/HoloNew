"""Construit la correspondance humain<->robot hors ligne par transport optimal par segment.

Indépendant du robot : les éléments spécifiques au robot sont l'URDF (``RobotSpec.urdf_path``) et
les angles au repos (``load/robot.correspondence_rest_angles``) ; tout le reste est générique.
L'actif est la paire ``(CorrespondenceTable, SurfaceSampling)`` — la table mappe chaque point de
surface du robot à un échantillon humain, et l'échantillonnage est le ``(tri_idx, bary)`` canonique
que le nuage humain (du sujet) doit réutiliser pour que son ordre de points corresponde à
``smpl_idx``. Sauvegardé au format ``.npz`` que ``correspondence/cache.py`` lit.

Le côté humain est un corps de modèle NEUTRE (zéro betas) : l'OT par segment normalise le centre et
l'échelle, donc l'appairage est neutre en forme et réutilisable sur les sujets ; seul le skinning du
nuage humain du sujet est recalculé (l'identité d'échantillonnage est partagée).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ...contracts import CorrespondenceTable, RobotSpec
from ...config import PrepareConfig
from ...load.robot import correspondence_rest_angles
from ...load.smpl import SmplBody, rest_body_model
from ..sampling import SurfaceSampling, sampling_id
from .cache import load_correspondence, save_correspondence
from .ot_couple import couple
from .robot_surface import sample_robot_surface
from .segments import point_segments

# SMPL-X au repos est natif Y-up face à +Z ; les mondes URDF humanoides sont Z-up face à +X.
# L'OT apparie l'humain contre le robot dans le repère mondial du robot, donc le nuage humain est
# pivoté dedans (appliqué comme points @ R.T). Sans cela, les ancres des axes par segment ne sont
# pas d'accord et les segments centraux cartographient miroir.
SMPLX_TO_ROBOT_FRAME = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _sample_surface(rest_verts: np.ndarray, faces: np.ndarray, density: float,
                    seed: int) -> SurfaceSampling:
    """Échantillonne uniformément le maillage au repos → l'``SurfaceSampling`` canonique (tri_idx, bary)."""
    import trimesh
    mesh = trimesh.Trimesh(vertices=np.asarray(rest_verts, np.float64), faces=np.asarray(faces),
                           process=False)
    n = max(1, int(mesh.area * density))
    pts, tri_idx = trimesh.sample.sample_surface_even(mesh, n, seed=seed)
    bary = trimesh.triangles.points_to_barycentric(mesh.triangles[tri_idx], pts)
    tri_idx, bary = np.asarray(tri_idx, np.int64), np.asarray(bary, np.float32)
    return SurfaceSampling(tri_idx=tri_idx, bary=bary, sampling_id=sampling_id(tri_idx, bary))


def _human_source(body: SmplBody, sampling: SurfaceSampling) -> tuple[np.ndarray, np.ndarray]:
    """Échantillons humains neutres au repos dans le repère mondial du robot (M,3) + leur index de segment (M,)."""
    rest = np.asarray(body.rest_vertices(None), np.float64)              # (V,3) native Y-up
    tri_v = body.faces[sampling.tri_idx]                                 # (N,3) ids de vertex
    pts = np.einsum("nij,ni->nj", rest[tri_v], sampling.bary.astype(np.float64))
    pts_robot = pts @ SMPLX_TO_ROBOT_FRAME.T                             # native -> robot world frame
    seg = point_segments(body.lbs_weights, body.faces, sampling.tri_idx, sampling.bary)
    return pts_robot, seg


def build_correspondence(config: PrepareConfig, neutral_body: SmplBody,
                         spec: RobotSpec) -> tuple[CorrespondenceTable, SurfaceSampling]:
    """Lance le pipeline OT complet → ``(CorrespondenceTable, SurfaceSampling)``."""
    sampling = _sample_surface(neutral_body.rest_vertices(None), neutral_body.faces,
                               config.cloud.human_density, config.cloud.seed)
    human_pts, human_seg = _human_source(neutral_body, sampling)
    robot = sample_robot_surface(spec.urdf_path, correspondence_rest_angles(spec.name),
                                 config.correspondence.robot_density)
    smpl_idx = couple(human_pts, human_seg, robot.points_world, robot.seg, config.correspondence.ot_reg)
    table = CorrespondenceTable(smpl_idx=smpl_idx, link_idx=robot.link_idx,
                                offset_local=robot.offset_local, link_names=robot.link_names,
                                smpl_sampling_id=sampling.sampling_id)
    return table, sampling


class CorrespondenceBuilder:
    """``AssetBuilder`` produisant le ``(CorrespondenceTable, SurfaceSampling)`` pour un (robot,
    modèle). Limité par (robot, modèle, config d'échantillonnage/OT) ; neutre en forme, donc réutilisé
    sur les sujets. ``save``/``load`` délèguent à ``correspondence/cache.py`` (le format partagé)."""

    def cache_key(self, config: PrepareConfig, spec: RobotSpec) -> str:
        h = hashlib.sha1()
        c, cc = config.cloud, config.correspondence
        h.update(f"{c.human_density}|{c.seed}|{cc.ot_reg}|{cc.robot_density}|{spec.name}".encode())
        h.update(str(spec.urdf_path).encode())
        h.update(repr(correspondence_rest_angles(spec.name)).encode())  # éditer les angles au repos g1 invalide le cache
        return h.hexdigest()

    def build(self, config: PrepareConfig, neutral_body: SmplBody,
              spec: RobotSpec) -> tuple[CorrespondenceTable, SurfaceSampling]:
        return build_correspondence(config, neutral_body, spec)

    def save(self, asset: tuple[CorrespondenceTable, SurfaceSampling], path: Path) -> None:
        save_correspondence(asset, path)

    def load(self, path: Path) -> tuple[CorrespondenceTable, SurfaceSampling]:
        return load_correspondence(path)


def regenerate(model_dir: Path, spec: RobotSpec, out_path: Path, config: PrepareConfig | None = None) -> None:
    """Reconstruit la correspondance pour ``spec`` à partir d'un corps de modèle NEUTRE et écrit le ``.npz``."""
    config = config or PrepareConfig()
    body = rest_body_model(np.zeros(10, np.float32), "neutral", model_dir)
    builder = CorrespondenceBuilder()
    asset = builder.build(config, body, spec)
    builder.save(asset, out_path)
    table, _ = asset
    print(f"correspondence: {table.n_points} robot points over {len(table.link_names)} links -> {out_path}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Rebuild the human<->robot OT correspondence.")
    ap.add_argument("--model-dir", required=True, type=Path, help="SMPL-X models dir")
    ap.add_argument("--urdf", required=True, type=Path, help="robot URDF")
    ap.add_argument("--robot-name", default="g1", help="RobotSpec.name (selects the rest pose)")
    ap.add_argument("--out", required=True, type=Path, help="output .npz")
    a = ap.parse_args()
    spec = RobotSpec(name=a.robot_name, urdf_path=a.urdf, link_names=(), dof=0, height=1.3)
    regenerate(a.model_dir, spec, a.out)


if __name__ == "__main__":
    main()
