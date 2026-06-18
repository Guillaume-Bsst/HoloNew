"""Central dataset/model path registry, read from path.yaml.

Replaces the WBT_* environment variables: edit path.yaml (next to COMMAND.md) for your
own layout. Relative entries resolve against the wbt_rl repo root; absolute entries are
used as-is.
"""
from __future__ import annotations

from pathlib import Path

# paths.py is at modules/01_retargeting/HoloNew/HoloNew/src/paths.py
_PKG_ROOT = Path(__file__).resolve().parents[1]   # .../HoloNew/HoloNew (where path.yaml lives)
REPO_ROOT = Path(__file__).resolve().parents[5]    # wbt_rl repo root
PATHS_YAML = _PKG_ROOT / "path.yaml"


def load_paths(yaml_path: Path | None = None) -> dict:
    """Parse path.yaml into a flat {key: value} dict.

    Minimal flat-YAML parser (the holonew env has no PyYAML, and this config is a flat
    key/value map): one `key: value` per line, `#` comments and blank lines ignored.
    """
    p = Path(yaml_path) if yaml_path is not None else PATHS_YAML
    if not p.exists():
        raise FileNotFoundError(f"path config not found: {p}")
    cfg: dict[str, str] = {}
    for raw in p.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()        # drop inline comments
        if not line or ":" not in line:
            continue
        key, val = line.split(":", 1)
        cfg[key.strip()] = val.strip()
    return cfg


def get_path(key: str, yaml_path: Path | None = None) -> Path:
    """Resolved path for `key`: relative -> repo root, absolute -> as-is.

    Raises ValueError if the key is missing/empty. Does NOT check existence (callers
    guard with .exists()/.is_dir() where a missing dataset should degrade rather than
    error).
    """
    cfg = load_paths(yaml_path)
    val = cfg.get(key)
    if not val:
        raise ValueError(
            f"path.yaml is missing '{key}'. Add it to {PATHS_YAML} (or pass an absolute path).")
    p = Path(val)
    return p if p.is_absolute() else (REPO_ROOT / p)
