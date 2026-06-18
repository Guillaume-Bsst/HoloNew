"""Façade normalization: map the --dataset + 3-path CLI onto the legacy
RetargetingConfig fields (data_path / task_name / data_format / omomo_dir) so the
existing loaders, TEST-SOCP / GMR builders, and view_stages direct reads all work
unchanged.

- omomo: motion is the InterMimic .pt (data_format smplh); betas live in the non-new
  OMOMO pickle (model_path) whose root (two levels up) is the omomo_dir the SMPL-X
  mesh / probe read.
- hodome: the raw HODome SMPL-X .npz is prepped once into the processed smplx npz the
  smplx retargeting path consumes, cached on disk; data_format becomes smplx.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from HoloNew.src.data_loaders.base import DATASET_TO_FORMAT
from HoloNew.src.data_loaders.hodome import prep_hodome_processed
from HoloNew.src.paths import get_path

# Disk cache for prepped HODome processed npz (one per sequence stem).
_HODOME_CACHE_DIR = Path(tempfile.gettempdir()) / "holonew_hodome_processed"


def resolve_paths_by_name(dataset: str, name: str):
    """Resolve (model_path, motion_path, obj_path, smpl_model_dir) from the dataset
    roots in path.yaml given only the sequence name. obj_path is None when absent/not
    applicable. Supported for omomo and hodome (the multi-file datasets)."""
    if dataset == "omomo":
        motion = get_path("omomo_new") / f"{name}.pt"
        omomo = get_path("omomo")
        model = omomo / "data" / "train_diffusion_manip_seq_joints24.p"
        smpl_model_dir = get_path("smplh_models")
        # Object name is the 2nd token (sub3_largebox_003 -> largebox); meshes are
        # named <obj>_cleaned_simplified.obj.
        obj = None
        parts = name.split("_")
        if len(parts) >= 2:
            matches = sorted((omomo / "data" / "captured_objects").glob(
                f"{parts[1]}_cleaned_simplified.obj"))
            obj = matches[0] if matches else None
        return model, motion, obj, smpl_model_dir

    if dataset == "hodome":
        root = get_path("hodome")
        motion = root / "smplx" / f"{name}.npz"
        obj = root / "object" / f"{name}.npz"
        model = get_path("smplx_models") / "smplx"
        return model, motion, (obj if obj.exists() else None), None

    raise ValueError(
        f"--motion-name is not supported for dataset {dataset!r}; pass explicit "
        f"--model-path/--motion-path instead.")


def resolve_motion_name_into_cfg(cfg) -> None:
    """Fill cfg.model_path/motion_path/obj_path/smpl_model_dir from cfg.motion_name via
    the global dataset roots. Explicit paths already set are kept (explicit wins).
    No-op if motion_name is None or both core paths are already provided."""
    # Dataset keys (registry, DATASET_TO_FORMAT) are lowercase; accept any input case
    # so --dataset OMOMO resolves like --dataset omomo.
    if cfg.dataset is not None:
        cfg.dataset = cfg.dataset.lower()
    if cfg.motion_name is None or (cfg.model_path is not None and cfg.motion_path is not None):
        return
    model, motion, obj, smpl_model_dir = resolve_paths_by_name(cfg.dataset, cfg.motion_name)
    cfg.model_path = cfg.model_path or model
    cfg.motion_path = cfg.motion_path or motion
    cfg.obj_path = cfg.obj_path or obj
    cfg.smpl_model_dir = cfg.smpl_model_dir or smpl_model_dir


def normalize_dataset_cfg(cfg) -> None:
    """When cfg.dataset is set, fill the legacy fields in place. No-op otherwise."""
    if cfg.dataset is None:
        return

    # --motion-name: resolve model/motion/obj/smpl-model from the global dataset roots.
    resolve_motion_name_into_cfg(cfg)

    if cfg.model_path is None or cfg.motion_path is None:
        raise ValueError(
            f"--dataset {cfg.dataset} requires --motion-name, or --model-path and --motion-path.")

    dataset = cfg.dataset
    cfg.data_format = DATASET_TO_FORMAT[dataset]

    # The smplx solve path is robot-only: objects are a viewer overlay (resolved from the
    # dataset's own files), not wired into the solver's InterMimic .pt object channel. So
    # object_interaction has no .pt to load and would build a wrong object SDF — force
    # robot_only for smplx datasets (the object overlay still shows, gated on the dataset).
    if cfg.data_format == "smplx" and cfg.task_type == "object_interaction":
        import logging
        logging.getLogger(__name__).info(
            "Dataset %s is smplx (object is a viewer overlay); using task_type=robot_only "
            "for the solve instead of object_interaction.", dataset)
        cfg.task_type = "robot_only"

    if dataset == "omomo":
        motion = Path(cfg.motion_path)
        cfg.data_path = motion.parent
        cfg.task_name = motion.stem
        # OMOMO release root holding data/{train,test}_*.p is two levels up from the
        # pickle file passed as model_path (…/OMOMO/data/<file>.p -> …/OMOMO).
        cfg.omomo_dir = Path(cfg.model_path).parent.parent

    elif dataset == "hodome":
        stem = Path(cfg.motion_path).stem
        _HODOME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        processed = _HODOME_CACHE_DIR / f"{stem}.npz"
        if not processed.exists():
            data = prep_hodome_processed(Path(cfg.motion_path), Path(cfg.model_path))
            np.savez(processed, **data)
        cfg.data_path = _HODOME_CACHE_DIR
        cfg.task_name = stem

    else:
        # lafan / sfu / climbing: motion_path locates the file; reuse its parent/stem.
        motion = Path(cfg.motion_path)
        cfg.data_path = motion.parent
        cfg.task_name = motion.stem
