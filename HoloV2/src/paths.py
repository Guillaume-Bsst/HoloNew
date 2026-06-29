"""Machine-local path registry for HoloV2 (dataset roots + SMPL-X model dir).

EDGE concern (effets de bord aux extrémités): read ONLY by CLI/entry points, never by the
pure pipeline (prepare/targets). Source of truth = HoloV2/paths.toml (gitignored,
machine-local; copy paths.example.toml). Parsed with the stdlib tomllib — no third-party dep.
These are environment paths, NOT algorithmic knobs, so they live here and never in config.py.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

HOLOV2_ROOT = Path(__file__).resolve().parents[1]   # .../HoloV2 (where paths.toml lives)
PATHS_TOML = HOLOV2_ROOT / "paths.toml"
PATHS_EXAMPLE = HOLOV2_ROOT / "paths.example.toml"


def load_paths(path: Path | None = None) -> dict:
    """Parse paths.toml -> dict, e.g. {"smplx": str, "roots": {dataset: str}}.

    Raises FileNotFoundError (pointing at the example template) when the file is absent.
    """
    p = Path(path) if path is not None else PATHS_TOML
    if not p.exists():
        raise FileNotFoundError(
            f"paths config not found: {p}. Copy the template: "
            f"`cp {PATHS_EXAMPLE.name} {p.name}` then edit your machine paths.")
    with open(p, "rb") as fh:
        return tomllib.load(fh)


def smplx_dir(cfg: dict | None = None, *, path: Path | None = None) -> Path:
    """SMPL-X model dir (folder holding SMPLX_NEUTRAL.npz). ValueError if the key is unset."""
    cfg = cfg if cfg is not None else load_paths(path)
    val = cfg.get("smplx")
    if not val:
        raise ValueError(f"paths.toml is missing 'smplx' (the SMPL-X model dir). Set it in {PATHS_TOML}.")
    return Path(val)


def dataset_root(dataset: str, cfg: dict | None = None, *, path: Path | None = None) -> Path:
    """Release root for `dataset` from the [roots] table. ValueError if the key is unset."""
    cfg = cfg if cfg is not None else load_paths(path)
    val = (cfg.get("roots") or {}).get(dataset)
    if not val:
        raise ValueError(
            f"paths.toml is missing roots.{dataset!r}. Add it under [roots] in {PATHS_TOML} "
            f"(or pass an absolute --motion-path / --dataset-root).")
    return Path(val)


def resolve_motion(dataset: str, motion: str | Path, cfg: dict | None = None,
                   *, path: Path | None = None) -> Path:
    """Resolve a motion path: absolute -> as-is; relative -> dataset_root(dataset)/motion."""
    m = Path(motion)
    if m.is_absolute():
        return m
    cfg = cfg if cfg is not None else load_paths(path)
    return dataset_root(dataset, cfg) / m
