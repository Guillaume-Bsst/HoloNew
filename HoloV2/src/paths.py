"""Machine-local path registry for HoloV2 (SMPL model assets + dataset roots).

EDGE concern (effets de bord aux extrémités): read ONLY by CLI/entry points, never by the
pure pipeline (prepare/targets). Source of truth = HoloV2/paths.toml (gitignored,
machine-local; copy paths.example.toml). Parsed with the stdlib tomllib — no third-party dep.
These are environment paths, NOT algorithmic knobs, so they live here and never in config.py.

Schema (see paths.example.toml):
    [models]          smplx (required), smplh (optional), smpl2smplx (optional .pkl file)
    [datasets.<name>] motion (base for a relative --motion-path), meta (optional; default motion)
"""
from __future__ import annotations

import tomllib
from pathlib import Path

HOLOV2_ROOT = Path(__file__).resolve().parents[1]   # .../HoloV2 (where paths.toml lives)
PATHS_TOML = HOLOV2_ROOT / "paths.toml"
PATHS_EXAMPLE = HOLOV2_ROOT / "paths.example.toml"


def load_paths(path: Path | None = None) -> dict:
    """Parse paths.toml -> dict ({"models": {...}, "datasets": {name: {...}}}).

    Raises FileNotFoundError (pointing at the example template) when the file is absent.
    """
    p = Path(path) if path is not None else PATHS_TOML
    if not p.exists():
        raise FileNotFoundError(
            f"paths config not found: {p}. Copy the template: "
            f"`cp {PATHS_EXAMPLE.name} {p.name}` then edit your machine paths.")
    with open(p, "rb") as fh:
        return tomllib.load(fh)


def _models(cfg: dict) -> dict:
    return cfg.get("models") or {}


def _dataset(cfg: dict, name: str) -> dict:
    return (cfg.get("datasets") or {}).get(name) or {}


def smplx_dir(cfg: dict | None = None, *, path: Path | None = None) -> Path:
    """SMPL-X model dir (folder with SMPLX_{NEUTRAL,MALE,FEMALE}.npz). Required: ValueError if unset."""
    cfg = cfg if cfg is not None else load_paths(path)
    val = _models(cfg).get("smplx")
    if not val:
        raise ValueError(f"paths.toml is missing [models].smplx (the SMPL-X model dir). Set it in {PATHS_TOML}.")
    return Path(val)


def smplh_dir(cfg: dict | None = None, *, path: Path | None = None) -> Path | None:
    """SMPL-H model dir (holds <gender>/model.npz), or None if unset. Optional (HOI-M3 only)."""
    cfg = cfg if cfg is not None else load_paths(path)
    val = _models(cfg).get("smplh")
    return Path(val) if val else None


def smpl2smplx_pkl(cfg: dict | None = None, *, path: Path | None = None) -> Path | None:
    """SMPL->SMPL-X deformation-transfer .pkl file, or None if unset. Optional (HOI-M3 only)."""
    cfg = cfg if cfg is not None else load_paths(path)
    val = _models(cfg).get("smpl2smplx")
    return Path(val) if val else None


def dataset_motion_root(name: str, cfg: dict | None = None, *, path: Path | None = None) -> Path:
    """Base dir for a relative --motion-path of `name`. Required: ValueError if unset."""
    cfg = cfg if cfg is not None else load_paths(path)
    val = _dataset(cfg, name).get("motion")
    if not val:
        raise ValueError(
            f"paths.toml is missing [datasets.{name}].motion. Add it in {PATHS_TOML} "
            f"(or pass an absolute --motion-path).")
    return Path(val)


def dataset_meta_root(name: str, cfg: dict | None = None, *, path: Path | None = None) -> Path | None:
    """Release root for `name`'s betas/scales/object meshes (fills SceneSpec.dataset_root).

    Defaults to the dataset's `motion` root when `meta` is unset; None if the dataset is absent.
    """
    cfg = cfg if cfg is not None else load_paths(path)
    d = _dataset(cfg, name)
    val = d.get("meta") or d.get("motion")
    return Path(val) if val else None


def resolve_motion(name: str, motion: str | Path, cfg: dict | None = None,
                   *, path: Path | None = None) -> Path:
    """Resolve a motion path: absolute -> as-is; relative -> dataset_motion_root(name)/motion."""
    m = Path(motion)
    if m.is_absolute():
        return m
    cfg = cfg if cfg is not None else load_paths(path)
    return dataset_motion_root(name, cfg) / m
