"""Registre de chemins locaux au système pour HoloV2 (assets modèles SMPL + racines dataset).

Préoccupation EDGE (effets de bord aux extrémités) : lire UNIQUEMENT par CLI/points d'entrée, jamais par
le pipeline pur (prepare/targets). Source de vérité = HoloV2/paths.toml (gitignored, local machine ;
copier paths.example.toml). Analysé avec tomllib stdlib — aucune dépendance tierce.
Ce sont des chemins d'environnement, PAS des knobs algorithmiques, donc ils vivent ici et jamais dans config.py.

Schéma (voir paths.example.toml):
    [models]          smplx (requis), smplh (optionnel), smpl2smplx (optionnel fichier .pkl)
    [datasets.<name>] motion (base pour --motion-path relatif), meta (optionnel ; défaut motion)
"""
from __future__ import annotations

import tomllib
from pathlib import Path

HOLOV2_ROOT = Path(__file__).resolve().parents[1]   # .../HoloV2 (où vit paths.toml)
PATHS_TOML = HOLOV2_ROOT / "paths.toml"
PATHS_EXAMPLE = HOLOV2_ROOT / "paths.example.toml"


def load_paths(path: Path | None = None) -> dict:
    """Analyser paths.toml → dict ({"models": {...}, "datasets": {name: {...}}}).

    Lève FileNotFoundError (pointant le modèle d'exemple) quand le fichier est absent.
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
    """Répertoire modèle SMPL-X (dossier avec SMPLX_{NEUTRAL,MALE,FEMALE}.npz). Requis : ValueError si non défini."""
    cfg = cfg if cfg is not None else load_paths(path)
    val = _models(cfg).get("smplx")
    if not val:
        raise ValueError(f"paths.toml is missing [models].smplx (the SMPL-X model dir). Set it in {PATHS_TOML}.")
    return Path(val)


def smplh_dir(cfg: dict | None = None, *, path: Path | None = None) -> Path | None:
    """Répertoire modèle SMPL-H (contient <gender>/model.npz), ou None si non défini. Optionnel (HOI-M3 uniquement)."""
    cfg = cfg if cfg is not None else load_paths(path)
    val = _models(cfg).get("smplh")
    return Path(val) if val else None


def smpl2smplx_pkl(cfg: dict | None = None, *, path: Path | None = None) -> Path | None:
    """Fichier .pkl de transfert de déformation SMPL→SMPL-X, ou None si non défini. Optionnel (HOI-M3 uniquement)."""
    cfg = cfg if cfg is not None else load_paths(path)
    val = _models(cfg).get("smpl2smplx")
    return Path(val) if val else None


def dataset_motion_root(name: str, cfg: dict | None = None, *, path: Path | None = None) -> Path:
    """Répertoire de base pour --motion-path relatif de `name`. Requis : ValueError si non défini."""
    cfg = cfg if cfg is not None else load_paths(path)
    val = _dataset(cfg, name).get("motion")
    if not val:
        raise ValueError(
            f"paths.toml is missing [datasets.{name}].motion. Add it in {PATHS_TOML} "
            f"(or pass an absolute --motion-path).")
    return Path(val)


def dataset_meta_root(name: str, cfg: dict | None = None, *, path: Path | None = None) -> Path | None:
    """Racine de mise à jour pour les betas/scales/meshes objets de `name` (remplit SceneSpec.dataset_root).

    Défaut vers la racine `motion` du dataset quand `meta` est non défini ; None si le dataset est absent.
    """
    cfg = cfg if cfg is not None else load_paths(path)
    d = _dataset(cfg, name)
    val = d.get("meta") or d.get("motion")
    return Path(val) if val else None


def resolve_motion(name: str, motion: str | Path, cfg: dict | None = None,
                   *, path: Path | None = None) -> Path:
    """Résoudre un chemin de mouvement : absolu → tel quel ; relatif → dataset_motion_root(name)/motion."""
    m = Path(motion)
    if m.is_absolute():
        return m
    cfg = cfg if cfg is not None else load_paths(path)
    return dataset_motion_root(name, cfg) / m
