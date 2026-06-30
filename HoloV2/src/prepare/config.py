"""Config de l'étape ``prepare`` — les SCHÉMAS dataclass (les knobs), séparés des données.

Les défauts sont des valeurs inline sensées, donc ``PrepareConfig()`` EST la config par défaut. Les
artefacts de données qui circulent dans le pipeline vivent dans ``prepare/contracts.py`` (config est
le HOW, séparé du WHAT). Les sous-configs sont groupés par livrable ; ``PrepareConfig`` les compose
et c'est ce que ``runner.prepare`` reçoit. Chaque builder lit (et hash dans sa clé cache) UNIQUEMENT
sa sous-config pertinente, donc un changement de knob invalide uniquement l'asset affecté.

Knob vs constante : un champ ici est quelque chose qu'un utilisateur règle légitimement. Les FAITS
fixes (conventions frame, ordres joints SMPL, taxonomie segment, formats dataset, poses repos robot)
et les caps perf internes NE sont PAS des knobs — ils restent comme constantes locales au module qui
les possède.

Chaque sous-config valide ses propres plages dans ``__post_init__`` (gelé, donc les vérifications
lèvent seulement — ne mutent jamais), attrapant un knob insensé à la construction plutôt que profond
dans un builder.

Changer la config d'une run : ``PrepareConfig()`` pour le défaut, ou override une sous-config inline,
p. ex. ``PrepareConfig(sdf=SdfConfig(spacing=0.005))``. Un frontend CLI tyro s'attache ici avec le
point d'entrée de run.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CalibrationConfig:
    """Ancrage scène + caractérisation sujet (``prepare/calibration``)."""

    foot_percentile: float = 50.0    # sol humain = ce percentile de la hauteur joint pied inférieur
                                     # sur le clip (50 = médiane ; cible le pied au repos, robuste à
                                     # la pénétration orteil SMPL qu'un min/low-percentile chercherait)
    object_floor_pct: float = 1.0    # sol objet = ce percentile (bas) du point objet posé le plus bas :
                                     # ~la « portée la plus basse » de l'objet qui touche le sol, robuste
                                     # à un sommet/frame bas errant
    # (pas de knob stature : l'ancrage est body-free ; la stature sujet vit sur BodyModel, betas-FK)

    def __post_init__(self) -> None:
        if not 0.0 <= self.foot_percentile <= 100.0:
            raise ValueError(f"foot_percentile must be in [0, 100], got {self.foot_percentile}")
        if not 0.0 <= self.object_floor_pct <= 100.0:
            raise ValueError(f"object_floor_pct must be in [0, 100], got {self.object_floor_pct}")


@dataclass(frozen=True)
class SdfConfig:
    """Grilles distance-signée pour objets / terrain (``prepare/sdf``)."""

    spacing: float = 0.01            # taille voxel isotrope (m) pour grilles objet/terrain
    margin: float = 0.05             # bande au-delà de la surface qui est stockée
    # (le sol plat est un SDF plan grossier mais EXACT — construit analytiquement, pas de mesh ; voir build_plane_sdf)

    def __post_init__(self) -> None:
        if self.spacing <= 0.0:
            raise ValueError(f"spacing must be > 0, got {self.spacing}")
        if self.margin <= 0.0:
            raise ValueError(f"margin must be > 0, got {self.margin}")


@dataclass(frozen=True)
class CloudConfig:
    """Nuages de points de surface — humain + objets (``prepare/point_cloud``)."""

    human_density: float = 2000.0    # points / m^2 sur la surface SMPL
    object_density: float = 2000.0   # points / m^2 sur chaque objet
    seed: int = 0                    # sampling déterministe — composante CLÉ (partagée par le nuage
                                     # humain ET la correspondance ; ils DOIVENT être d'accord)
    k_influences: int = 4            # K du skinning LBS-on-cloud creux

    def __post_init__(self) -> None:
        if self.human_density <= 0.0:
            raise ValueError(f"human_density must be > 0, got {self.human_density}")
        if self.object_density <= 0.0:
            raise ValueError(f"object_density must be > 0, got {self.object_density}")
        if self.k_influences < 1:
            raise ValueError(f"k_influences must be >= 1, got {self.k_influences}")


@dataclass(frozen=True)
class CorrespondenceConfig:
    """Correspondance surface humain↔robot par transport optimal (``prepare/point_cloud/correspondence``)."""

    ot_reg: float = 0.05             # régularisation entropique OT
    robot_density: float = 3000.0    # points / m^2 échantillonnés sur la surface robot pour le build OT

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
    """Tous les knobs de l'étape ``prepare``, composés — l'objet unique que ``runner.prepare`` reçoit ;
    chaque builder lit seulement sa sous-config. ``cloud`` alimente le nuage humain, la correspondance,
    et la table géodésique (un unique sampling canonique partagé par tous les trois)."""

    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    sdf: SdfConfig = field(default_factory=SdfConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    correspondence: CorrespondenceConfig = field(default_factory=CorrespondenceConfig)
    geodesic: GeodesicConfig = field(default_factory=GeodesicConfig)
