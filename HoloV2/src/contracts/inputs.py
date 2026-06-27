"""Pipeline ENTRY inputs (data identity) — distinct from the step config.

``RobotSpec`` (target robot identity) + ``SceneSpec`` (WHAT to run). The HOW (algorithm knobs)
is the step config, kept apart at the top of the repo in ``config_types`` / ``config_values``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RobotSpec:
    """Identity of the target robot (drives loading, FK, cache keys)."""

    name: str                      # "g1", "h1", "t1"
    urdf_path: Path
    link_names: tuple[str, ...]
    dof: int
    height: float                  # nominal robot height (m); consumed DOWNSTREAM by the
                                   # correspondence/transport layer as scale = robot_height /
                                   # human_stature — NOT by the (robot-free) calibration


@dataclass(frozen=True)
class SceneSpec:
    """WHAT to run. The loader turns this into ``RawMotion``; cache keys mix this identity
    with the relevant step-config subset (the schemas in ``config_types``)."""

    dataset: str                              # loader key (omomo/hodome/sfu/hoim3)
    motion_path: Path                         # the sequence
    robot: RobotSpec
    smpl_model_dir: Path | None = None        # parametric body model dir (None ok => style-only)
    object_mesh_paths: tuple[Path, ...] = ()  # optional override; else resolved by the loader
    ground_mesh_path: Path | None = None      # None => flat ground (plane SDF); else terrain mesh -> SDF
    cache_dir: Path | None = None             # default: HoloV2/cache/
    dataset_root: Path | None = None          # release root for auxiliary metadata kept apart from
                                              # motion_path (OMOMO betas/scales + captured meshes)
    person_id: int | None = None             # multi-person datasets: which person to retarget
                                              # (None => the first present); ignored if single-person
    object_names: tuple[str, ...] | None = None  # named-object datasets: subset to load
                                              # (None => all); ignored if objects are unnamed
