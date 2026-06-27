"""Config TYPES for the ``prepare`` step — the dataclass SCHEMAS (what knobs exist).

The defaults are sensible inline values; named presets live in ``config_values.prepare``,
and the data artifacts that flow through the pipeline live in ``src.contracts`` (config is
kept apart from data on purpose). The sub-configs are grouped by deliverable; ``PrepareConfig``
composes them and is what ``runner.prepare`` receives. Each builder reads (and hashes into its cache
key) ONLY its relevant sub-config, so a knob change invalidates only the affected asset.

Knob vs constant: a field here is something a user legitimately tunes. Fixed FACTS (frame
conventions, SMPL joint orders, segment taxonomy, dataset formats, robot rest poses) and internal
perf caps are NOT knobs — they stay as constants local to the module that owns them.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CalibrationConfig:
    """Scene grounding + subject characterisation (``prepare/calibration``)."""

    foot_percentile: float = 50.0    # human floor = this percentile of the lower foot-joint height
                                     # over the clip (50 = median; targets the resting foot, robust to
                                     # the SMPL toe penetration a min/low-percentile would chase)
    object_floor_pct: float = 1.0    # object floor = this (low) percentile of the lowest posed object
                                     # point: ~the "lowest reach" of the floor-touching object, robust
                                     # to a stray low vertex/frame
    fallback_stature: float = 1.78   # human stature (m) used for a non-parametric source (no betas to FK)


@dataclass(frozen=True)
class SdfConfig:
    """Signed-distance grids for objects / terrain (``prepare/sdf``)."""

    spacing: float = 0.01            # isotropic voxel size (m) for object/terrain grids
    margin: float = 0.05             # band beyond the surface that is stored
    # (the flat ground is a coarse but EXACT plane SDF — built analytically, no mesh; see build_plane_sdf)


@dataclass(frozen=True)
class CloudConfig:
    """Surface point clouds — human + objects (``prepare/point_cloud``)."""

    human_density: float = 2000.0    # points / m^2 on the SMPL surface
    object_density: float = 2000.0   # points / m^2 on each object
    seed: int = 0                    # deterministic sampling — KEY component (shared by the human
                                     # cloud AND the correspondence; they MUST agree)
    k_influences: int = 4            # K of the sparse LBS-on-cloud skinning


@dataclass(frozen=True)
class CorrespondenceConfig:
    """Human<->robot surface correspondence by optimal transport (``prepare/point_cloud/correspondence``)."""

    rest_pose: str = "tpose"         # robot rest config used for the OT alignment
    ot_reg: float = 0.05             # OT entropic regularisation
    robot_density: float = 3000.0    # points / m^2 sampled on the robot surface for the OT build


@dataclass(frozen=True)
class PrepareConfig:
    """All knobs of the ``prepare`` step, composed — the single object ``runner.prepare`` receives;
    each builder reads only its sub-config. ``cloud`` feeds BOTH the human cloud and the
    correspondence (their dependency chain)."""

    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    sdf: SdfConfig = field(default_factory=SdfConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    correspondence: CorrespondenceConfig = field(default_factory=CorrespondenceConfig)
