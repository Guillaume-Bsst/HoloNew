"""Config of the ``prepare`` stage — the dataclass SCHEMAS (the knobs), kept apart from the data.

The defaults are sensible inline values, so ``PrepareConfig()`` IS the default config. The data
artifacts that flow through the pipeline live in ``prepare/contracts.py`` (config is the HOW, kept
apart from the WHAT). The sub-configs are grouped by deliverable; ``PrepareConfig`` composes them and
is what ``runner.prepare`` receives. Each builder reads (and hashes into its cache key) ONLY its
relevant sub-config, so a knob change invalidates only the affected asset.

Knob vs constant: a field here is something a user legitimately tunes. Fixed FACTS (frame
conventions, SMPL joint orders, segment taxonomy, dataset formats, robot rest poses) and internal
perf caps are NOT knobs — they stay as constants local to the module that owns them.

Each sub-config validates its own ranges in ``__post_init__`` (frozen, so the checks only raise —
they never mutate), catching a nonsensical knob at construction rather than deep in a builder.

Changing a run's config: ``PrepareConfig()`` for the default, or override a sub-config inline, e.g.
``PrepareConfig(sdf=SdfConfig(spacing=0.005))``. A tyro CLI front-end attaches here with the run
entry point.
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
    # (no stature knob: grounding is body-free; the subject stature lives on the BodyModel, betas-FK)

    def __post_init__(self) -> None:
        if not 0.0 <= self.foot_percentile <= 100.0:
            raise ValueError(f"foot_percentile must be in [0, 100], got {self.foot_percentile}")
        if not 0.0 <= self.object_floor_pct <= 100.0:
            raise ValueError(f"object_floor_pct must be in [0, 100], got {self.object_floor_pct}")


@dataclass(frozen=True)
class SdfConfig:
    """Signed-distance grids for objects / terrain (``prepare/sdf``)."""

    spacing: float = 0.01            # isotropic voxel size (m) for object/terrain grids
    margin: float = 0.05             # band beyond the surface that is stored
    # (the flat ground is a coarse but EXACT plane SDF — built analytically, no mesh; see build_plane_sdf)

    def __post_init__(self) -> None:
        if self.spacing <= 0.0:
            raise ValueError(f"spacing must be > 0, got {self.spacing}")
        if self.margin <= 0.0:
            raise ValueError(f"margin must be > 0, got {self.margin}")


@dataclass(frozen=True)
class CloudConfig:
    """Surface point clouds — human + objects (``prepare/point_cloud``)."""

    human_density: float = 2000.0    # points / m^2 on the SMPL surface
    object_density: float = 2000.0   # points / m^2 on each object
    seed: int = 0                    # deterministic sampling — KEY component (shared by the human
                                     # cloud AND the correspondence; they MUST agree)
    k_influences: int = 4            # K of the sparse LBS-on-cloud skinning

    def __post_init__(self) -> None:
        if self.human_density <= 0.0:
            raise ValueError(f"human_density must be > 0, got {self.human_density}")
        if self.object_density <= 0.0:
            raise ValueError(f"object_density must be > 0, got {self.object_density}")
        if self.k_influences < 1:
            raise ValueError(f"k_influences must be >= 1, got {self.k_influences}")


@dataclass(frozen=True)
class CorrespondenceConfig:
    """Human<->robot surface correspondence by optimal transport (``prepare/point_cloud/correspondence``)."""

    ot_reg: float = 0.05             # OT entropic regularisation
    robot_density: float = 3000.0    # points / m^2 sampled on the robot surface for the OT build

    def __post_init__(self) -> None:
        if self.ot_reg <= 0.0:
            raise ValueError(f"ot_reg must be > 0, got {self.ot_reg}")
        if self.robot_density <= 0.0:
            raise ValueError(f"robot_density must be > 0, got {self.robot_density}")


@dataclass(frozen=True)
class GeodesicConfig:
    """Graphe géodésique sur le cloud objet (``prepare/geodesic``). La DENSITÉ/seed ne sont PAS ici :
    la géodésique réutilise l'échantillonnage du ``object_cloud`` (``CloudConfig``) — un seul sampling
    canonique partagé par le cloud ET la table géodésique."""

    k_neighbors: int = 8       # k du graphe k-NN de surface (Dijkstra all-pairs scipy)
    normal_gate: float = -0.5  # arête i--j seulement si dot(n_i, n_j) > normal_gate, dans [-1, 1].
                               # DÉFAUT -0.5 : garde les faces perpendiculaires adjacentes (dot=0, ex.
                               # arêtes d'un cube) ET coupe les arêtes quasi-opposées (dot≈-1, ex.
                               # traversée d'une plaque fine). 0.0 scinderait un cube en 6 faces ;
                               # -1.0 ≈ aucun gating.
    max_points: int = 6000     # garde-fou : ValueError si P dépasse (stockage 4*P^2 octets) —
                               # baisser object_density ou relever ce knob en conscience

    def __post_init__(self) -> None:
        if self.k_neighbors < 1:
            raise ValueError(f"k_neighbors must be >= 1, got {self.k_neighbors}")
        if not -1.0 <= self.normal_gate <= 1.0:
            raise ValueError(f"normal_gate must be in [-1, 1], got {self.normal_gate}")
        if self.max_points < 1:
            raise ValueError(f"max_points must be >= 1, got {self.max_points}")


@dataclass(frozen=True)
class PrepareConfig:
    """All knobs of the ``prepare`` step, composed — the single object ``runner.prepare`` receives;
    each builder reads only its sub-config. ``cloud`` feeds the human cloud, the correspondence,
    and the geodesic table (one canonical sampling shared by all three)."""

    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    sdf: SdfConfig = field(default_factory=SdfConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    correspondence: CorrespondenceConfig = field(default_factory=CorrespondenceConfig)
    geodesic: GeodesicConfig = field(default_factory=GeodesicConfig)
