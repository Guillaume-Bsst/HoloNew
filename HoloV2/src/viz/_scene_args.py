"""Glue CLI partagé pour les points d'entrée viz : déclare les flags de scène communs et assemble une
``SceneSpec`` entièrement résolue (complète les défauts à partir du paths.toml local machine).

EDGE-uniquement : importé par les fonctions ``main()`` viz, jamais par le pipeline pur. Évite aux quatre
viewers de dupliquer chacun la construction RobotSpec/SceneSpec.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .. import paths
from ..prepare.contracts import RobotSpec, SceneSpec


def add_scene_args(ap: argparse.ArgumentParser) -> None:
    """Ajoute les flags de sélection de scène partagés par les CLIs viz."""
    ap.add_argument("--dataset", default="hodome")
    ap.add_argument("--motion-path", required=True, type=Path,
                    help="absolute, or relative to [datasets.<dataset>].motion in paths.toml")
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="SMPL-X model dir; default: paths.toml [models].smplx")
    ap.add_argument("--dataset-root", type=Path, default=None,
                    help="release root for object/betas metadata; default: paths.toml [datasets.<dataset>].meta")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--frame-step", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--person-id", type=int, default=None, help="multi-person: which person to retarget")
    ap.add_argument("--object-names", default=None, help="comma-separated subset of objects to load")


def _g1_robot() -> RobotSpec:
    """RobotSpec G1 défaut pour les points d'entrée viz (source unique URDF/DOF/stature)."""
    return RobotSpec(name="g1", urdf_path=paths.HOLOV2_ROOT / "models" / "g1" / "g1_29dof.urdf",
                     link_names=("pelvis",), dof=29, height=1.3)


def scene_from_args(a: argparse.Namespace, *, paths_file: Path | None = None) -> SceneSpec:
    """Construit une SceneSpec entièrement résolue, complète les chemins manquants à partir de paths.toml.

    Les args CLI explicites toujours gagnent ; paths.toml n'est lu que si un défaut est requis (ainsi
    les invocations entièrement explicites et absolues fonctionnent même sans paths.toml).
    """
    # paths.toml n'est FORTEMENT requis que si un défaut doit en provenir : un model-dir manquant
    # ou un chemin de mouvement relatif. Un --dataset-root manquant se dégrade à None (dataset_meta_root
    # retourne None), il ne doit donc PAS forcer le fichier — les invocations absolues fonctionnent sans paths.toml.
    hard_need = (a.model_dir is None) or (not Path(a.motion_path).is_absolute())
    try:
        cfg = paths.load_paths(paths_file)
    except FileNotFoundError:
        if hard_need:
            raise
        cfg = {}

    model_dir = a.model_dir if a.model_dir is not None else paths.smplx_dir(cfg)
    motion = paths.resolve_motion(a.dataset, a.motion_path, cfg)
    droot = a.dataset_root if a.dataset_root is not None else paths.dataset_meta_root(a.dataset, cfg)

    objs = tuple(a.object_names.split(",")) if a.object_names else None
    return SceneSpec(dataset=a.dataset, motion_path=motion, robot=_g1_robot(),
                     smpl_model_dir=model_dir, dataset_root=droot,
                     person_id=a.person_id, object_names=objs,
                     smplh_dir=paths.smplh_dir(cfg), smpl2smplx_pkl=paths.smpl2smplx_pkl(cfg))
