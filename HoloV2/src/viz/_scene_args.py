"""Shared CLI glue for the viz entry points: declare the common scene flags and assemble a
fully-resolved ``SceneSpec`` from them (filling defaults from the machine-local paths.toml).

EDGE-only: imported by viz ``main()`` functions, never by the pure pipeline. Keeps the four
viewers from each duplicating the RobotSpec/SceneSpec construction.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .. import paths
from ..prepare.contracts import RobotSpec, SceneSpec


def add_scene_args(ap: argparse.ArgumentParser) -> None:
    """Add the scene-selection flags shared by the viz CLIs."""
    ap.add_argument("--dataset", default="hodome")
    ap.add_argument("--motion-path", required=True, type=Path,
                    help="absolute, or relative to the dataset's [roots] entry in paths.toml")
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="SMPL-X model dir; default: paths.toml 'smplx'")
    ap.add_argument("--dataset-root", type=Path, default=None,
                    help="release root for object/betas metadata; default: paths.toml roots[dataset]")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--frame-step", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--person-id", type=int, default=None, help="multi-person: which person to retarget")
    ap.add_argument("--object-names", default=None, help="comma-separated subset of objects to load")


def scene_from_args(a: argparse.Namespace, *, paths_file: Path | None = None) -> SceneSpec:
    """Build a fully-resolved SceneSpec, filling missing paths from paths.toml.

    Explicit CLI args always win; paths.toml is read only when a default is needed (so fully
    explicit, absolute invocations work even without a paths.toml).
    """
    need_cfg = (a.model_dir is None) or (a.dataset_root is None) or (not Path(a.motion_path).is_absolute())
    cfg = paths.load_paths(paths_file) if need_cfg else {}

    model_dir = a.model_dir if a.model_dir is not None else paths.smplx_dir(cfg)
    motion = paths.resolve_motion(a.dataset, a.motion_path, cfg)
    droot = a.dataset_root
    if droot is None:
        try:
            droot = paths.dataset_root(a.dataset, cfg)
        except ValueError:
            droot = None   # datasets without a configured root (e.g. hoim3) keep dataset_root unset

    objs = tuple(a.object_names.split(",")) if a.object_names else None
    robot = RobotSpec(name="g1", urdf_path=paths.HOLOV2_ROOT / "models" / "g1" / "g1_29dof.urdf",
                      link_names=("pelvis",), dof=29, height=1.3)
    return SceneSpec(dataset=a.dataset, motion_path=motion, robot=robot,
                     smpl_model_dir=model_dir, dataset_root=droot,
                     person_id=a.person_id, object_names=objs)
