"""Façade normalization: map the --dataset + 3-path CLI onto the legacy
RetargetingConfig fields (data_path / task_name / data_format / omomo_dir) so the
existing loaders, TEST-SOCP / GMR builders, and view_stages direct reads all work
unchanged.

- omomo: motion is the InterMimic .pt (data_format smplh); betas live in the non-new
  OMOMO pickle (model_path) whose root (two levels up) is the omomo_dir the SMPL-X
  mesh / probe read.
- hoim3: the raw HODome SMPL-X .npz is prepped once into the processed smplx npz the
  smplx retargeting path consumes, cached on disk; data_format becomes smplx.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from HoloNew.src.data_loaders.base import DATASET_TO_FORMAT
from HoloNew.src.data_loaders.hoim3 import prep_hoim3_processed

# Disk cache for prepped HOI-M3 processed npz (one per sequence stem).
_HOIM3_CACHE_DIR = Path(tempfile.gettempdir()) / "holonew_hoim3_processed"


def normalize_dataset_cfg(cfg) -> None:
    """When cfg.dataset is set, fill the legacy fields in place. No-op otherwise."""
    if cfg.dataset is None:
        return

    if cfg.model_path is None or cfg.motion_path is None:
        raise ValueError(
            f"--dataset {cfg.dataset} requires --model-path and --motion-path.")

    dataset = cfg.dataset
    cfg.data_format = DATASET_TO_FORMAT[dataset]

    if dataset == "omomo":
        motion = Path(cfg.motion_path)
        cfg.data_path = motion.parent
        cfg.task_name = motion.stem
        # OMOMO release root holding data/{train,test}_*.p is two levels up from the
        # pickle file passed as model_path (…/OMOMO/data/<file>.p -> …/OMOMO).
        cfg.omomo_dir = Path(cfg.model_path).parent.parent

    elif dataset == "hoim3":
        stem = Path(cfg.motion_path).stem
        _HOIM3_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        processed = _HOIM3_CACHE_DIR / f"{stem}.npz"
        if not processed.exists():
            data = prep_hoim3_processed(Path(cfg.motion_path), Path(cfg.model_path))
            np.savez(processed, **data)
        cfg.data_path = _HOIM3_CACHE_DIR
        cfg.task_name = stem

    else:
        # lafan / sfu / climbing: motion_path locates the file; reuse its parent/stem.
        motion = Path(cfg.motion_path)
        cfg.data_path = motion.parent
        cfg.task_name = motion.stem
