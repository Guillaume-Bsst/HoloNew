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
from dataclasses import replace
from pathlib import Path

import numpy as np

from HoloNew.data_utils.prep_amass_smplx_for_rt import prep_amass_processed
from HoloNew.src.data_loaders.base import DATASET_TO_FORMAT
from HoloNew.src.data_loaders.hodome import PREP_FORMAT_VERSION, prep_hodome_processed
from HoloNew.src.paths import get_path

# Disk caches for prepped processed npz (one per sequence stem), per source dataset.
_HODOME_CACHE_DIR = Path(tempfile.gettempdir()) / "holonew_hodome_processed"
_SFU_CACHE_DIR = Path(tempfile.gettempdir()) / "holonew_sfu_processed"


def _processed_cache_valid(path: Path) -> bool:
    """True iff the cached processed npz exists and matches the current prep format
    version. A pre-versioning cache (no ``prep_version`` key, e.g. the old 22-joint
    orientations) or a stale version is treated as invalid so it gets rebuilt. Shared by
    the HODome and SFU caches (both write the same processed format + ``prep_version``)."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        with np.load(path) as d:
            return "prep_version" in d.files and int(d["prep_version"]) == PREP_FORMAT_VERSION
    except Exception:  # noqa: BLE001 - corrupt/unreadable cache -> rebuild
        return False


def ensure_hodome_processed(motion_path, model_path, cache_dir=None) -> Path:
    """Path to the processed HODome npz, (re)built when missing or stale (prep format
    version mismatch) and cached on disk keyed by sequence stem. Writes the current
    ``prep_version`` so future format bumps invalidate older caches automatically."""
    cache_dir = Path(cache_dir) if cache_dir is not None else _HODOME_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    processed = cache_dir / f"{Path(motion_path).stem}.npz"
    if not _processed_cache_valid(processed):
        data = prep_hodome_processed(Path(motion_path), Path(model_path))
        np.savez(processed, prep_version=PREP_FORMAT_VERSION, **data)
    return processed


def ensure_sfu_processed(motion_path, model_dir, cache_dir=None) -> Path:
    """Path to the processed SFU npz, (re)built when missing or stale and cached on disk
    keyed by sequence stem (mirrors HODome). ``motion_path`` is the raw AMASS SMPL-X .npz;
    ``model_dir`` is the SMPL-X body-model dir (…/models/smplx) whose PARENT holds the
    SMPLX_*.npz the AMASS prep loads."""
    cache_dir = Path(cache_dir) if cache_dir is not None else _SFU_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    processed = cache_dir / f"{Path(motion_path).stem}.npz"
    if not _processed_cache_valid(processed):
        data = prep_amass_processed(Path(motion_path), Path(model_dir).parent)
        np.savez(processed, prep_version=PREP_FORMAT_VERSION, **data)
    return processed


def resolve_paths_by_name(dataset: str, name: str):
    """Resolve (model_path, motion_path, obj_path, smpl_model_dir) from the dataset
    roots in path.yaml given only the sequence name. obj_path is None when absent/not
    applicable. Supported for omomo, hodome, sfu and lafan (sfu/lafan are object-less)."""
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

    if dataset == "sfu":
        # SFU motions are nested per subject: <root>/<subject>/<name>.npz, subject = the
        # token before the first "_" (0008_ChaCha001_stageii -> 0008). No object; reuse the
        # shared SMPL-X body model (same as hodome).
        root = get_path("sfu")
        subject = name.split("_", 1)[0]
        motion = root / subject / f"{name}.npz"
        model = get_path("smplx_models") / "smplx"
        return model, motion, None, None

    if dataset == "lafan":
        # LAFAN loads the preprocessed .npy (extract_global_positions output), flat under
        # the lafan root. No object and no separate body model; model_path is a non-None
        # placeholder (the root) so the normalize gate passes — the lafan loader ignores it.
        root = get_path("lafan")
        motion = root / f"{name}.npy"
        return root, motion, None, None

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


def _has_object_source(cfg) -> bool:
    """True when the dataset loader yields an object for this sequence."""
    if cfg.obj_path is None:
        return False
    from HoloNew.src.data_loaders.base import resolve_loader
    srcs = resolve_loader(cfg.dataset).object_source(
        motion_path=cfg.motion_path, obj_path=cfg.obj_path, model_path=cfg.model_path,
        task_type=cfg.task_type, constants=None, motion_data_config=cfg.motion_data_config,
        smpl_model_dir=getattr(cfg, "smpl_model_dir", None))
    return len(srcs) > 0


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

    # An object-less dataset is robot-only: there is no object to wire into the solver, so
    # object_interaction would build a wrong/empty object SDF — force robot_only. smplx may
    # carry an object (HODome with an object .npz), so it only downgrades when the loader
    # yields none; lafan/sfu never have an object channel here. smplh (OMOMO) keeps its .pt
    # object, so it is left untouched.
    if (cfg.data_format in ("smplx", "lafan") and cfg.task_type == "object_interaction"
            and not _has_object_source(cfg)):
        import logging
        logging.getLogger(__name__).info(
            "Dataset %s (%s) has no object source; using task_type=robot_only "
            "for the solve instead of object_interaction.", dataset, cfg.data_format)
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
        # Version-checked disk cache: a pre-#5 cache (22-joint orientations, no version
        # key) is detected as stale and rebuilt with 55-joint (hand-posing) orientations.
        ensure_hodome_processed(cfg.motion_path, cfg.model_path)
        cfg.data_path = _HODOME_CACHE_DIR
        cfg.task_name = stem
        # Object name = token (2nd "_"-segment), so create_task_constants propagates
        # OBJECT_NAME=token to the builder gate AND the scene-swap path (_w_<token>.xml).
        if cfg.task_type == "object_interaction" and _has_object_source(cfg):
            token = stem.split("_", 1)[1] if "_" in stem else stem
            cfg.task_config = replace(cfg.task_config, object_name=token)

    elif dataset == "sfu":
        # SFU raw is AMASS SMPL-X (poses/betas/trans), not the processed format the smplx
        # path reads. FK-prep it into that format, cached on disk by stem (mirrors hodome),
        # then point the legacy data_path/task_name at the cache. model_path is the SMPL-X
        # body-model dir; the prep uses its parent (…/models) to load SMPLX_*.npz.
        stem = Path(cfg.motion_path).stem
        ensure_sfu_processed(cfg.motion_path, cfg.model_path)
        cfg.data_path = _SFU_CACHE_DIR
        cfg.task_name = stem

    else:
        # lafan / climbing: motion_path locates the file; reuse its parent/stem.
        motion = Path(cfg.motion_path)
        cfg.data_path = motion.parent
        cfg.task_name = motion.stem
