"""Contrats de données de l'étape ``prepare`` — sa surface publique de TYPES.

Chaque artefact que ``prepare`` produit ou consomme en franchissant une limite de module est une
dataclass GELÉE de tableaux numpy (Structure-of-Arrays) ou un PROTOCOLE d'interface. DONNÉES +
PROTOCOLES uniquement — pas de logique, pas d'I/O, pas de dépendances lourdes (numpy-only), donc
ce module est importable partout.

C'est le contrat de l'étape : les étapes aval importent leurs entrées D'ICI (par ex. ``targets`` fait
``from ..prepare.contracts import GroundedScene, InteractionContext``), jamais des sous-modules
internes de ``prepare``. Les knobs (le HOW) vivent à part dans ``prepare/config.py`` ; celui-ci ne
tient que les données (le WHAT) qui circulent. Le pipeline est linéaire (prepare → targets → solve),
chaque étage possède ses propres contrats et ne dépend que des types publics de l'étage amont — le
graphe de dépendances reste acyclique.

Conventions
-----------
- Per-frame est l'unité canonique. La cible actuelle est OFFLINE replay (``process_frame`` indexe une
  ``GroundedScene`` chargée au frame ``f``) ; la téléopération en direct est une variante future des
  mêmes opérations.
- Deux ensembles de joints, maintenus distincts : ``J_demo`` (les joints du dataset, utilisés par
  ``style``) et ``J_bones`` (le squelette SMPL, utilisé pour poser les nuages). Ne jamais les
  confondre.
- Les quaternions sont wxyz. Les poses rigides sont ``(x, y, z, qw, qx, qy, qz)``.
- Les tableaux sont read-only par convention (``frozen`` gèle la liaison, pas le buffer).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


# =============================================================================
# Protocoles (interfaces — impls concrètes dans prepare/load/*)
# =============================================================================
@runtime_checkable
class BodyModel(Protocol):
    """Corps humain paramétrique (famille SMPL). Impl concrète dans ``prepare/load/smpl.py``.
    Pose le corps à partir de paramètres per-frame ; ``bone_transforms`` donne les transformations
    monde per-os utilisées pour poser le nuage humain (mesh-free, via le skinning creux)."""

    faces: np.ndarray  # (F, 3) int — topologie, frame-invariant
    n_bones: int       # J_bones (52 SMPL-H / 55 SMPL-X)
    stature: float     # stature au repos du sujet (m), betas-FK — propriété pure rest-mesh (pas de mouvement).
                       # Vit sur le corps (son propriétaire naturel), PAS sur la calibration ; alimente
                       # le ratio du style : ``ratio = stature / StyleConfig.human_height_assumption``,
                       # appliqué dans ``targets`` via ``targets.config.SceneScaleConfig``.

    def posed_vertices(self, params: "SmplParams", t: int) -> np.ndarray:
        """(V, 3) sommets mesh monde au frame ``t`` (usage offline : sampling, viz)."""

    def bone_transforms(self, params: "SmplParams", t: int) -> tuple[np.ndarray, np.ndarray]:
        """(J_bones,3,3) rotations monde et (J_bones,3) origines monde au frame ``t`` (FK)."""

    def rest_vertices(self, params: "SmplParams") -> np.ndarray:
        """(V, 3) sommets en pose au repos pour le sujet (sampling du nuage une seule fois)."""


@runtime_checkable
class RobotModel(Protocol):
    """Cinématique robot. Les transformations au repos (q-indépendantes) sont utilisées par ``prepare``
    pour échantillonner la surface et construire la correspondance ; le FK complet + Jacobiens
    (q-dépendants) sont utilisés par ``solve`` via l'évaluateur ``targets``. La configuration ``q``
    est un vecteur free-flyer pinocchio ``[pelvis(7: pos + quat xyzw), joints]`` de longueur ``nq`` ;
    la tangente ``v`` a pour dimension ``nv = 6 + n_joints``. Impl concrète dans ``prepare/load/robot.py``
    (pinocchio)."""

    link_names: tuple[str, ...]
    dof: int          # joints actionnés (= nv - 6)
    nq: int           # dim configuration (free-flyer)
    nv: int           # dim tangente (= 6 + dof)

    def link_transforms(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(L,3,3) rotations, (L,3) positions : transformation MONDE de chaque lien pour ``q`` (free-flyer)."""

    def rest_transforms(self) -> tuple[np.ndarray, np.ndarray]:
        """Transformations des liens à la configuration neutre (base identité, joints 0)."""

    def neutral(self) -> np.ndarray:
        """Configuration neutre ``(nq,)`` (base identité avec quaternion unitaire, joints 0)."""

    def integrate(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Pas variétal ``q ⊕ v`` ``(nq,)`` (préserve le quaternion base unitaire)."""

    def link_jacobians(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Pour ``q`` : (rot (L,3,3), pos (L,3), jac_lin (L,3,nv), jac_ang (L,3,nv)) dans le repère
        MONDE, alignés avec ``link_names``. ``jac_lin``/``jac_ang`` sont les Jacobiens de repère
        translationnel/angulaire LOCAL_WORLD_ALIGNED : ``dp_world = jac_lin @ v``, ``omega_world = jac_ang @ v``."""

    def joint_pos_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Limites de position des joints actionnés ``(lower (dof,), upper (dof,))`` (rad), alignées avec
        les DOFs des joints ``v[6:6+dof]`` / ``q[7:7+dof]``. Utilisé par ``solve`` pour boxer le pas joint."""


class AssetBuilder(Protocol):
    """FORME commune des builders d'artefacts offline (``prepare/``) : calibration, sdf,
    point_cloud. Un guide NOMINAL (cache_key / build / load / save), PAS une interface
    polymorphique stricte : chaque builder concret prend sa propre sous-config (un schéma de
    ``prepare/config.py``) plus ses propres entrées spécifiques, donc les vraies signatures
    diffèrent. ``config``/inputs sont typés ``Any`` ici pour cette raison, et ``@runtime_checkable``
    est intentionnellement omis — une vérification ``isinstance`` sur les signatures ``Any`` serait
    une fausse garantie. Chaque builder hash UNIQUEMENT son sous-ensemble de config pertinent
    (+ inputs + clés amont), donc un changement de param invalide seulement les éléments affectés."""

    def cache_key(self, config: Any, *inputs: Any) -> str:
        """Clé stable du sous-ensemble de config pertinent + inputs (hash géométrie/sujet)."""

    def build(self, config: Any, *inputs: Any) -> Any:
        """Le calcul offline lourd → l'artefact."""

    def load(self, path: Path) -> Any: ...
    def save(self, asset: Any, path: Path) -> None: ...


# =============================================================================
# Entrée : quoi exécuter (identité données) — distinct de la config d'étape
# =============================================================================
@dataclass(frozen=True)
class RobotSpec:
    """Identité du robot cible (détermine le chargement, FK, clés cache)."""

    name: str                      # "g1", "h1", "t1"
    urdf_path: Path
    link_names: tuple[str, ...]
    dof: int
    height: float                  # hauteur robot nominale (m) ; NON utilisée par la calibration
                                   # (robot-free). L'échelle de scène des refs n'utilise PAS
                                   # ``robot_height/stature`` — elle utilise
                                   # ``ratio = stature / StyleConfig.human_height_assumption``
                                   # (ratio du style), appliqué dans ``targets.config.SceneScaleConfig``.


@dataclass(frozen=True)
class SceneSpec:
    """QUOI exécuter. Le loader transforme ceci en ``RawMotion`` ; les clés cache mélangent cette
    identité avec le sous-ensemble pertinent de config d'étape (les schémas dans ``prepare/config.py``)."""

    dataset: str                              # clé loader (omomo/hodome/sfu/hoim3)
    motion_path: Path                         # la séquence
    robot: RobotSpec
    smpl_model_dir: Path | None = None        # répertoire modèle corps paramétrique (None ok => style-only)
    object_mesh_paths: tuple[Path, ...] = ()  # override optionnel ; sinon résolu par le loader
    ground_mesh_path: Path | None = None      # None => sol plat (SDF plan) ; sinon terrain mesh → SDF
    cache_dir: Path | None = None             # défaut : HoloV2/cache/
    dataset_root: Path | None = None          # racine release pour métadonnées auxiliaires séparées de
                                              # motion_path (OMOMO betas/scales + meshes capturés)
    person_id: int | None = None             # datasets multi-personne : quelle personne retargeter
                                              # (None => la première présente) ; ignoré si mono-personne
    object_names: tuple[str, ...] | None = None  # datasets objets nommés : sous-ensemble à charger
                                              # (None => tous) ; ignoré si objets sans nom
    smplh_dir: Path | None = None             # HOI-M3 uniquement : répertoire modèle SMPL-H
                                              # (contient <gender>/model.npz) ; None => dériver par
                                              # convention de smpl_model_dir
    smpl2smplx_pkl: Path | None = None        # HOI-M3 uniquement : transfer deformation .pkl SMPL→SMPL-X ;
                                              # None => dériver par convention de smpl_model_dir


# =============================================================================
# load/ — mouvement brut et paramètres corps paramétrique
# =============================================================================
@dataclass(frozen=True)
class SmplParams:
    """Paramètres per-frame d'un corps paramétrique (SMPL-H / SMPL-X). Inclut les MAINS
    (nécessaires pour la préhension) ; les paramètres de face SMPL-X sont optionnels."""

    betas: np.ndarray            # (B,)        forme sujet (time-invariant)
    global_orient: np.ndarray    # (T, 3)      orientation racine, axis-angle
    body_pose: np.ndarray        # (T, 21*3)   rotations joints corps, axis-angle
    left_hand_pose: np.ndarray   # (T, 15*3)
    right_hand_pose: np.ndarray  # (T, 15*3)
    transl: np.ndarray           # (T, 3)      translation racine
    gender: str                  # "neutral" | "male" | "female"
    model_type: str              # "smplh" | "smplx"
    jaw_pose: np.ndarray | None = None     # (T, 3)   SMPL-X uniquement
    leye_pose: np.ndarray | None = None    # (T, 3)
    reye_pose: np.ndarray | None = None    # (T, 3)
    expression: np.ndarray | None = None   # (T, E)

    @property
    def n_frames(self) -> int:
        return self.transl.shape[0]


@dataclass(frozen=True)
class RawMotion:
    """Sortie d'un dataset loader ``prepare/load/`` — uniforme entre formats, AVANT calibration.
    Chaque loader courant est PARAMÉTRIQUE (remplit ``smpl_params``) ; le ``| None`` est une
    provision structurelle pour une future source positions-only, PAS un chemin actif. Quand
    ``smpl_params is None`` il n'y a pas de corps à poser, et le pipeline ``targets`` basé os
    (style + interaction ont tous les deux besoin du FK du corps) ne s'exécute pas — voir
    ``is_parametric``."""

    joint_pos: np.ndarray                 # (T, J_demo, 3) positions joints monde (toujours présent)
    joint_names: tuple[str, ...]          # (J_demo,)
    fps: float
    source_format: str
    object_poses_raw: tuple[np.ndarray, ...]  # un (T, 7) par objet
    object_mesh_paths: tuple[Path, ...]       # un par objet, aligné avec poses
    smpl_params: SmplParams | None = None

    @property
    def is_parametric(self) -> bool:
        return self.smpl_params is not None

    @property
    def n_frames(self) -> int:
        return self.joint_pos.shape[0]


# =============================================================================
# calibration — géométrie scène et ancrage
# =============================================================================
@dataclass(frozen=True)
class ObjectMesh:
    """Un objet rigide : géométrie dans son repère local + pose monde per-frame. Construit à la
    demande par ``prepare/load/mesh.py`` (offline uniquement — n'atteint jamais le runtime/solve)."""

    vertices: np.ndarray  # (V, 3) repère local objet
    faces: np.ndarray     # (F, 3) int
    poses: np.ndarray     # (T, 7) pose monde par frame [x,y,z,qw,qx,qy,qz]
    name: str
    static: bool = False  # pose constante sur T → eval peut ignorer la transformation per-frame


@dataclass(frozen=True)
class Calibration:
    """ANCRAGE per-(sujet, prise). ROBOT-FREE *ET* BODY-FREE : construit à partir des joints démo mocap
    (sol humain) et des meshes/poses objets (sol objet) seuls — pas de betas/body nécessaires — donc
    cache par prise indépendamment du robot cible. La ``stature`` du sujet vit sur ``BodyModel`` (son
    propriétaire naturel rest-mesh), et l'échelle human→robot n'y appartient PAS — l'échelle de scène
    des refs utilise ``ratio = stature / StyleConfig.human_height_assumption`` (le ratio du style),
    appliquée dans ``targets`` via ``targets.config.SceneScaleConfig``.

    Mono-humain, multi-objet : l'humain et les objets s'ancrent chacun par leur PROPRE décalage z
    (l'humain peut flotter tandis que les objets reposent déjà au sol, donc un décalage scène partagé
    les pousserait à travers). ``human_offset`` ancre l'humain (ses pieds) ; ``object_offset`` est UN
    SEUL décalage partagé par TOUS les objets (ancre l'objet qui touche le sol juste au-dessus du sol,
    gardant la géométrie inter-objets). Asset offline, PAS un cache géométrie : (sujet, prise).

    TODO : une calibration par-objet/inter-objets plus fine pourrait ancrer chaque objet et optimiser
    conjointement les contacts objet↔objet & objet↔sol (alors ``object_offset`` → décalages par-objet)."""

    human_offset: float                  # décalage z ancrant l'humain (pieds → sol)
    object_offset: float                 # décalage z partagé par TOUS les objets (objet qui touche → ~sol)
    root_frame: np.ndarray               # (4, 4) transformation monde encadrant la racine


@dataclass(frozen=True)
class GroundedScene:
    """Sortie de ``prepare`` (mouvement chargé avec calibration appliquée). L'unique entrée des deux
    traitements (style, interaction). L'ancrage ``Calibration`` voyage dedans (provenance/viz), donc
    ``prepare`` retourne juste ``(GroundedScene, InteractionContext)``.

    Porte le ``body`` du sujet (le moteur de pose live) : per frame, ``interaction`` pose le nuage
    humain via ``body.bone_transforms(smpl_params, f)``. Le body est typé par le PROTOCOLE numpy-only
    ``BodyModel``, donc ``targets`` l'appelle tout en restant torch-free à l'import (torch est caché
    à l'intérieur de l'instance, construit une fois dans ``prepare``). Les meshes objets restent des
    CHEMINS mesh, PAS de géométrie live — l'asymétrie est justifiée : l'humain DEFORME (a besoin de FK
    per-frame), les objets sont RIGIDES (une pose7 + le nuage objet pré-échantillonné suffisent).
    ``style`` lit aussi le body (il suit les OS SMPL, pas les joints démo) ; ``solve`` ne voit jamais de
    ``GroundedScene`` (consomme ``FrameTargets``) — donc aucun objet lourd ne l'atteint. ``body is None``
    ⇔ source positions-only (pas de params SMPL) : un placeholder STRUCTUREL, pas un chemin câblé — le
    pipeline ``targets`` est basé os (style + cloud posing ont tous deux besoin du FK du body) et lève sur
    ``body is None``."""

    joint_pos: np.ndarray                  # (T, J_demo, 3) joints démo ancrés — style
    joint_names: tuple[str, ...]           # (J_demo,)
    object_poses: tuple[np.ndarray, ...]   # pose monde ancrée (T, 7) par objet
    object_mesh_paths: tuple[Path, ...]    # géométrie tirée à la demande par prepare
    calibration: Calibration
    fps: float
    smpl_params: SmplParams | None = None  # params ancrés → consommés par ``body.bone_transforms``
    body: BodyModel | None = None          # moteur de pose live du sujet (None => positions-only)

    @property
    def n_frames(self) -> int:
        return self.joint_pos.shape[0]

    @property
    def n_objects(self) -> int:
        return len(self.object_poses)

    @property
    def is_parametric(self) -> bool:
        return self.smpl_params is not None


# =============================================================================
# sdf / point_cloud — assets géométrie build-once (les entrées interaction)
# =============================================================================
@dataclass(frozen=True)
class SDF:
    """Grille de distance signée d'une surface rigide, dans son repère local — pour objets, terrain
    ET sol plat (un plan est un champ affine, donc l'interpolation trilinéaire le reproduit EXACTEMENT
    sur une petite grille ; c'est aussi un SDF ordinaire, gardant chaque canal homogène — voir
    ``build_plane_sdf``).

    Porte une grille WITNESS (point de surface le plus proche par nœud) à côté de la distance : l'eval
    reconstruit la direction de contact comme ``normalize(probe - witness)`` à partir du witness
    interpolé trilinéairement, qui reste un vrai vecteur unitaire près des arêtes/coins de box aigus —
    où le gradient différence finie de la grille de distance est instable. Échantillonné par interpolation
    trilinéaire dans l'eval (``targets/interaction/eval.py``) ; pure données ici (pas de méthode)."""

    grid: np.ndarray     # (Nx, Ny, Nz) distance signée (négatif = intérieur)
    witness: np.ndarray  # (Nx, Ny, Nz, 3) point de surface le plus proche par nœud, repère local
    origin: np.ndarray   # (3,) coords locales du nœud (0, 0, 0)
    spacing: float       # taille voxel isotrope (m)
    name: str            # nom du canal, p. ex. "obj0" / "ground"

    def __post_init__(self) -> None:
        if self.witness.shape != self.grid.shape + (3,):
            raise ValueError(
                f"witness shape {self.witness.shape} != grid shape {self.grid.shape} + (3,)")


@dataclass(frozen=True)
class GeodesicTable:
    """All-pairs géodésique (distance de graphe k-NN) sur les points de surface d'un mesh rigide, en
    frame locale. AUTO-CONTENU : porte SES points + normales, donc consommable sans le ``object_cloud``.
    La ligne ``geo[j]`` EST le champ géodésique mono-source depuis le point ``j`` (lookup O(1), ligne
    contiguë) — c'est ce qu'on lit à un ``witness(q)`` continu pour le résidu witness (côté solve).
    Géométrie rigide ⇒ pose-invariant (une translation/rotation préserve les géodésiques)."""

    points: np.ndarray    # (P, 3) f32  échantillons de surface (= sampling object_cloud), frame locale
    normals: np.ndarray   # (P, 3) f32  normale unitaire par point (gating snap/interp thin/concave)
    geo: np.ndarray       # (P, P) f32  geo[i, j] = géodésique de graphe i->j (symétrique)
    name: str             # nom de canal ("obj0"/"terrain") — provenance, aligné SDF/cloud
    sampling_id: str = "" # identité du sampling (densité/seed/topo) — provenance

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    def __post_init__(self) -> None:
        p = self.points.shape[0]
        if self.geo.shape != (p, p):
            raise ValueError(f"geo shape {self.geo.shape} != (P, P) with P={p}")
        if self.normals.shape != self.points.shape:
            raise ValueError(
                f"normals shape {self.normals.shape} != points shape {self.points.shape}")


@dataclass(frozen=True)
class Channel:
    """Un canal d'évaluation = une source distance-signée + sa liaison pose per-frame. Rend
    l'alignement sol/objet EXPLICITE (pas de décalage implicite N vs N+1). CHAQUE canal porte un
    ``sdf`` pour que l'eval ait UN SEUL chemin trilinéaire (homogène, pas de cas spécial sol-plat) ;
    ``object_idx`` ne fait que fixer la liaison pose :

    - ``object_idx is None`` => le SOL statique dans le repère monde. Son ``sdf`` est une grille de plan
      par défaut (un plan est affine, donc une petite grille reproduit ``z`` EXACTEMENT) ou une grille
      TERRAIN (escaliers/pente/escalade).
    - ``object_idx`` défini  => objet ``object_idx``, son ``sdf`` posé par ``object_poses[object_idx][f]``."""

    name: str
    object_idx: int | None        # None = sol statique (monde) ; sinon index dans object_poses/clouds
    sdf: SDF                       # grille distance-signée (plan sol / terrain / objet)
    geodesic: "GeodesicTable | None" = None   # None = sol PLAN (le coût retombe sur l'euclidien
                                              # analytique, qui EST la géodésique exacte d'un plan) ;
                                              # sinon objet/terrain. Seule entorse au "jamais de None"
                                              # du sdf, assumée : le plan est le seul cas à forme close.


@dataclass(frozen=True)
class PointCloud:
    """Échantillons de surface portant leur propre SKINNING CREUX, posés à partir des transformations
    partie seules (mesh-free, torch-free), uniformément pour chaque sorte de partie :
      - objet : K=1, poids 1, partie = le corps rigide.
      - robot : K=1, poids 1, partie = le lien (posé par FK).
      - humain : K~4, blend LBS-on-cloud sur les os SMPL dominants (ferme les plis articulaires).

    Pose un frame, donnée la transformation monde de chaque partie ``T[j] = (R_j, t_j)`` :
        p_world[i] = sum_k weights[i,k] * (R[parts[i,k]] @ offsets[i,k] + t[parts[i,k]])
    ``offsets`` sont dans le repère REST-local de chaque partie (skinning cuit une fois offline)."""

    parts: np.ndarray     # (P, K) int    index partie/os par influence
    weights: np.ndarray   # (P, K) float  lignes somment à 1 (K=1 => rigide)
    offsets: np.ndarray   # (P, K, 3)     point dans repère rest-local de partie k
    sampling_id: str = "" # identité du sampling (densité/seed/topologie) — se lie à la
                          # correspondance construite contre lui (voir CorrespondenceTable)

    @property
    def n_points(self) -> int:
        return self.parts.shape[0]

    @property
    def n_influences(self) -> int:
        return self.parts.shape[1]


@dataclass(frozen=True)
class CorrespondenceTable:
    """Correspondance fixe SMPL ↔ surface robot (construite une fois par transport optimal, OT).

    Appaire M points : côté humain (``smpl_idx`` dans le nuage SMPL) et côté robot (``link_idx`` +
    ``offset_local`` dans le repère de ce lien). Le transport copie le champ humain à ``smpl_idx[m]``
    sur le point robot m. VALIDE UNIQUEMENT pour le nuage SMPL dont ``sampling_id == smpl_sampling_id``
    (assert à l'assemblage)."""

    smpl_idx: np.ndarray         # (M,) index dans l'ordre de points du PointCloud SMPL
    link_idx: np.ndarray         # (M,) index lien robot (dans link_names)
    offset_local: np.ndarray     # (M, 3) point robot dans le repère de ce lien
    link_names: tuple[str, ...]  # (L,)
    smpl_sampling_id: str = ""   # le sampling nuage humain contre lequel celui-ci a été construit

    @property
    def n_points(self) -> int:
        return self.smpl_idx.shape[0]


@dataclass(frozen=True)
class InteractionContext:
    """Tous les assets build-once pour le traitement interaction, passés explicitement (pas de globals).

    Invariants (vérifiés à l'assemblage) :
    - ``channels[0]`` est le SOL (statique ; SDF plan par défaut, ou SDF terrain) ;
      le reste sont des canaux objets avec ``object_idx`` alignés avec ``object_clouds`` et
      l'ordre objets de la scène.
    - ``human_cloud.sampling_id == correspondence.smpl_sampling_id``.
    - ``robot_cloud.n_points == correspondence.n_points`` (mêmes M points)."""

    channels: tuple[Channel, ...]          # sol (statique) + un par objet
    human_cloud: PointCloud                # sur la surface SMPL
    object_clouds: tuple[PointCloud, ...]  # un par objet (object_clouds[i] ↔ channel object_idx=i)
    correspondence: CorrespondenceTable    # SMPL → robot (liaison STATIQUE)
    margin: float                          # marge activation champ (m)
    robot_cloud: PointCloud                # les M points robot de correspondance comme nuage K=1, parties
                                           # dans l'ordre lien FK robot — solve les pose à q (re-eval online)
    robot: RobotModel                      # moteur cinématique q-dépendant (FK pour poser robot_cloud) ;
                                           # miroir de GroundedScene.body, dépendances lourdes cachées

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.channels)
