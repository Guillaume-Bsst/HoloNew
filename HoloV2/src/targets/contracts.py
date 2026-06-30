"""Contrats de données de l'étage ``targets`` — sa surface de types PUBLIQUE.

Les résultats d'évaluation de champ par frame et les artefacts de cibles (le contrat ``targets`` ->
``solve``), plus l'état de pose par frame partagé et la trace viz. Dataclasses FROZEN de tableaux numpy,
numpy-only (pas de logique, pas d'I/O), donc ce module est importable partout.

``targets`` consomme les contrats amont ``prepare`` (``from ..prepare.contracts import ...``) et les
expose comme ses propres types publics ; ``solve`` et ``viz`` importent leurs entrées d'ici. Le
pipeline est linéaire (prepare -> targets -> solve), donc chaque étage possède ses contrats et ne dépend
que des types publics de l'étage amont — le graphe de dépendances reste acyclique.

Convention canal-first : les tableaux ``ContactField`` / ``MultiChannelField`` sont ``(C, P)`` = C canaux
sur P points (ops par canal contiguës). C = sol + N objets. ``J_bones`` (squelette SMPL, dans
``FramePose``) est distinct de ``J_demo`` (les joints du dataset) — ne jamais les confondre. Une séquence
est ``list[FrameTargets]``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# =============================================================================
# interaction/ — résultats d'évaluation de champ
# =============================================================================
@dataclass(frozen=True)
class ContactField:
    """Un nuage vs UN canal, UNE frame. Probes inactives : distance=+margin, le reste 0.
    ``direction``/``witness`` sont dans le frame du CANAL (voir ``MultiChannelField``)."""

    distance: np.ndarray   # (P,)    distance signée
    direction: np.ndarray  # (P, 3)  normale de contact (surface -> point)
    witness: np.ndarray    # (P, 3)  point de surface le plus proche
    active: np.ndarray     # (P,)    bool, dans le margin

    def __post_init__(self) -> None:
        for name in ("distance", "direction", "witness", "active"):
            getattr(self, name).flags.writeable = False


@dataclass(frozen=True)
class MultiChannelField:
    """Un nuage vs TOUS les canaux, UNE frame. Canal-first, homogène (C = sol + N obj).

    Frame NATUREL par canal (prêt pour l'objet-comme-variable) : le canal SOL est dans le frame MONDE ;
    chaque canal OBJET est dans le frame LOCAL de CET objet — la probe est mappée dans le frame objet,
    le champ y est lu, et ``direction``/``witness`` y sont GARDÉS (pas d'aller-retour monde).
    ``distance`` est une longueur (invariante au frame). C'est exactement le frame dans lequel les termes
    objet du solve sont construits (la Jacobienne de mouvement rigide de l'objet se couple en objet-local),
    donc le canal objet ne nécessite AUCUNE réécriture quand l'objet devient une variable de décision."""

    distance: np.ndarray         # (C, P)
    direction: np.ndarray        # (C, P, 3)
    witness: np.ndarray          # (C, P, 3)
    active: np.ndarray           # (C, P) bool
    channels: tuple[str, ...]    # (C,) noms de canaux

    def __post_init__(self) -> None:
        C = len(self.channels)
        for name in ("distance", "direction", "witness", "active"):
            got = getattr(self, name).shape[0]
            if got != C:
                raise ValueError(f"{name} has {got} channels, expected {C}")
        if self.distance.ndim != 2:
            raise ValueError(f"distance must be 2-D (C, P), got shape {self.distance.shape}")
        P = self.distance.shape[1]
        if self.active.shape != (C, P):
            raise ValueError(f"active shape {self.active.shape} != ({C}, {P})")
        if self.direction.shape != (C, P, 3):
            raise ValueError(f"direction shape {self.direction.shape} != ({C}, {P}, 3)")
        if self.witness.shape != (C, P, 3):
            raise ValueError(f"witness shape {self.witness.shape} != ({C}, {P}, 3)")
        for name in ("distance", "direction", "witness", "active"):
            getattr(self, name).flags.writeable = False

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def n_points(self) -> int:
        return self.distance.shape[1]


# =============================================================================
# cibles par frame -> solve
# =============================================================================
@dataclass(frozen=True)
class StyleTargets:
    """Objectif de style, une frame : suivi de posture/style du robot, G1-ready via le mapping de joints.
    Le canal "comment le corps devrait bouger", indépendant des objets. Forme provisoire — l'objectif
    de style est encore en cours de conception (voir ``targets/style/``).

    GÉOMÉTRIE par frame seulement : OÙ chaque link suivi devrait être (``position``) et comment il devrait
    être orienté (``orientation``). À quel POINT suivre chaque link (les poids de suivi / gains de coût)
    est une affaire de SOLVEUR — statique, pas par frame — donc ce n'est PAS porté ici ; ``solve`` le
    définit dans sa propre config à sa construction (V1 ``w_p`` / ``w_r``)."""

    link_names: tuple[str, ...]            # (L,)
    position: np.ndarray                   # (L, 3) cible monde par link
    orientation: np.ndarray | None = None  # (L, 4) wxyz, ou None si position seule

    def __post_init__(self) -> None:
        L = len(self.link_names)
        if self.position.shape != (L, 3):
            raise ValueError(f"position shape {self.position.shape} != ({L}, 3)")
        if self.orientation is not None and self.orientation.shape != (L, 4):
            raise ValueError(f"orientation shape {self.orientation.shape} != ({L}, 4)")
        self.position.flags.writeable = False
        if self.orientation is not None:
            self.orientation.flags.writeable = False


@dataclass(frozen=True)
class RobotInteractionTargets:
    """Champ humain transporté sur les M points de correspondance du robot, UNE frame.
    La liaison statique (à quel link chaque point s'attache) vit dans
    ``InteractionContext.correspondence`` — PAS dupliquée ici par frame."""

    field: MultiChannelField               # sur les M points robot


@dataclass(frozen=True)
class EnvironmentInteractionTargets:
    """Nuages objets vs les canaux (objet<->sol / objet<->objet), UNE frame ; NON transportés.

    Entrée de solve de première classe pour les termes OBJET-COMME-VARIABLE : quand l'objet est une
    variable de décision, ces contacts côté scène (objet vs sol, objet vs autres objets, en frame
    objet-local) pilotent sa cohérence. Même matrice d'éval que le côté humain (peu coûteuse, homogène),
    avec UN terme supplémentaire que le côté humain n'a pas : la DIAGONALE (objet i vs son PROPRE canal i).
    Le nuage y repose sur sa propre surface, donc elle est remplie par le self-contact en forme close
    (distance 0, witness = le point lui-même ; voir ``eval_fields`` ``self_idx``), PAS un vrai échantillon
    — le solve ignore ce canal diagonal."""

    per_object: tuple[MultiChannelField, ...]  # un par nuage objet


@dataclass(frozen=True)
class FrameTargets:
    """Sortie de ``process_frame`` pour UNE frame ; le contrat targets -> solve. Une séquence est
    ``list[FrameTargets]``.

    Le seam du solve est ``(FrameTargets, InteractionContext)`` : le solve lit aussi l'``InteractionContext``
    statique (la liaison de correspondance pour les points de contrôle du robot, et les SDFs de canaux
    qu'il re-interroge en ces points). ``env_interaction`` alimente les termes objet-comme-variable (les
    propres contacts de l'objet), donc il fait partie du chemin prod — pas seulement viz."""

    style: StyleTargets
    robot_interaction: RobotInteractionTargets
    env_interaction: EnvironmentInteractionTargets
    object_rot: np.ndarray                 # (N, 3, 3) rotations monde objets par frame — frame du canal
                                           # objet du solve + init/référence de l'objet-variable
    object_pos: np.ndarray                 # (N, 3)    positions monde objets par frame

    def __post_init__(self) -> None:
        n = len(self.env_interaction.per_object)
        if not (self.object_rot.shape[0] == self.object_pos.shape[0] == n):
            raise ValueError(
                f"object poses ({self.object_rot.shape[0]} rot, {self.object_pos.shape[0]} pos) "
                f"must match env_interaction.per_object count ({n})")
        self.object_rot.flags.writeable = False
        self.object_pos.flags.writeable = False


# =============================================================================
# état par frame partagé + trace viz
# =============================================================================
@dataclass(frozen=True)
class FramePose:
    """Transforms monde par frame, calculés UNE fois et partagés par les deux traitements : ``style``
    utilise les joints démo (depuis GroundedScene) ; ``interaction`` utilise ces transforms bone + objet
    pour positionner ses nuages. ``J_bones`` = squelette SMPL (distinct de J_demo)."""

    bone_rot: np.ndarray    # (J_bones, 3, 3) rotations monde des bones SMPL
    bone_pos: np.ndarray    # (J_bones, 3)    origines monde des bones SMPL
    object_rot: np.ndarray  # (N, 3, 3) rotations monde objets
    object_pos: np.ndarray  # (N, 3)    positions monde objets

    def __post_init__(self) -> None:
        J = self.bone_rot.shape[0]
        if self.bone_pos.shape != (J, 3):
            raise ValueError(
                f"bone_pos shape {self.bone_pos.shape} != ({J}, 3) — "
                f"must match bone_rot leading dim")
        N = self.object_rot.shape[0]
        if self.object_pos.shape != (N, 3):
            raise ValueError(
                f"object_pos shape {self.object_pos.shape} != ({N}, 3) — "
                f"must match object_rot leading dim")
        for name in ("bone_rot", "bone_pos", "object_rot", "object_pos"):
            getattr(self, name).flags.writeable = False


@dataclass(frozen=True)
class FrameTrace:
    """TOUS les artefacts d'une frame, pour inspection / visualisation. Produit par
    ``targets.pipeline.trace_frame`` — les MÊMES ops purs que ``process_frame``, intermédiaires
    gardés. Le seam propre pour ``viz/`` : zéro hook dans le calcul."""

    pose: FramePose
    human_cloud_world: np.ndarray                  # (P, 3) nuage SMPL posé
    object_clouds_world: tuple[np.ndarray, ...]    # par objet, (P_i, 3)
    human_field: MultiChannelField                 # sur le nuage humain (PRÉ-transport)
    targets: FrameTargets                          # sorties finales (style + robot + env)


# =============================================================================
# EVAL (q-dépendant) — état géométrique courant + Jacobiennes analytiques (targets.Evaluator)
# =============================================================================
# Miroir des références ci-dessus pour la MÊME op conceptuelle (poser une config, lire style + contact),
# appliquée à la config OPTIMISÉE (robot @ q + objets @ SE(3)). Reference-free, cost-free : le
# résidu (cur - ref) et le coût vivent dans ``solve``. Convention de tangente : pinocchio v
# (nv = 6 + n_joints) pour q ; world-aligned (δt, δθ) pour chaque objet (LOCAL_WORLD_ALIGNED).
@dataclass(frozen=True)
class StyleEval:
    """État courant des links suivis à ``q`` (FK), + jacobiennes géométriques. Reference-free,
    cost-free. Ordre = ``StyleTargets.link_names`` (mêmes links que la référence de style)."""

    position: np.ndarray         # (L, 3)      position monde courante du link
    rotation: np.ndarray         # (L, 3, 3)   rotation monde courante du link
    jac_pos: np.ndarray          # (L, 3, nv)  ∂position/∂v   (monde)
    jac_rot: np.ndarray          # (L, 3, nv)  ∂ω/∂v          (jac angulaire géométrique, monde)
    link_names: tuple[str, ...]  # (L,)

    def __post_init__(self) -> None:
        L = len(self.link_names)
        if self.position.shape != (L, 3):
            raise ValueError(f"position shape {self.position.shape} != ({L}, 3)")
        if self.rotation.shape != (L, 3, 3):
            raise ValueError(f"rotation shape {self.rotation.shape} != ({L}, 3, 3)")
        if self.jac_pos.ndim != 3 or self.jac_pos.shape[:2] != (L, 3):
            raise ValueError(f"jac_pos shape {self.jac_pos.shape} != ({L}, 3, nv)")
        nv = self.jac_pos.shape[2]
        if self.jac_rot.shape != (L, 3, nv):
            raise ValueError(f"jac_rot shape {self.jac_rot.shape} != jac_pos ({L}, 3, {nv})")
        for name in ("position", "rotation", "jac_pos", "jac_rot"):
            getattr(self, name).flags.writeable = False


@dataclass(frozen=True)
class ContactEnvEval:
    """Côté env : nuage objet ``i`` vs canaux. Dépend des poses objets seules (pas de ``q``).
    Diagonale self-contact déjà neutralisée par ``eval_fields`` (``self_idx``) côté ``field`` ;
    ``probe_jac_obj`` y est rempli par la formule générique (inoffensif, la diagonale est ignorée
    par ``solve``). Tangente objet world-aligned ``(δt, δθ)``."""

    field: MultiChannelField   # (C, P_i)
    cloud_jac_self: np.ndarray  # (P_i, 3, 6)    ∂(point du nuage objet i, monde)/∂(tangente objet i)
    probe_jac_obj: np.ndarray  # (C, P_i, 3, 6) ∂(probe dans le frame canal)/∂(tangente SE(3) objet du canal)

    def __post_init__(self) -> None:
        C, P = self.field.n_channels, self.field.n_points
        if self.cloud_jac_self.shape != (P, 3, 6):
            raise ValueError(f"cloud_jac_self shape {self.cloud_jac_self.shape} != ({P}, 3, 6)")
        if self.probe_jac_obj.shape != (C, P, 3, 6):
            raise ValueError(f"probe_jac_obj shape {self.probe_jac_obj.shape} != ({C}, {P}, 3, 6)")
        self.cloud_jac_self.flags.writeable = False
        self.probe_jac_obj.flags.writeable = False


@dataclass(frozen=True)
class ContactEval:
    """Géométrie de contact courante (robot) + jacobiennes géométriques pour ``(q, object_poses)``.
    Reference-free, cost-free. Canal-first ``(C, M)`` sur les M points de contrôle robot. ``field``
    suit la convention ``MultiChannelField`` (sol en monde, canal objet en objet-local) ; ``point_jac``
    est en MONDE. ``probe_jac_obj`` : lignes du canal sol = 0 ; canal ``c`` -> objet
    ``channels[c].object_idx`` (creux). Tangente objet world-aligned ``(δt, δθ)``."""

    field: MultiChannelField   # (C, M)
    point_jac: np.ndarray      # (M, 3, nv)     ∂(point robot monde)/∂v
    probe_jac_obj: np.ndarray  # (C, M, 3, 6)   ∂(probe dans le frame canal)/∂(tangente SE(3) objet du canal)
    env: tuple[ContactEnvEval, ...]  # côté environnement, un par nuage objet

    def __post_init__(self) -> None:
        C, M = self.field.n_channels, self.field.n_points
        if self.point_jac.ndim != 3 or self.point_jac.shape[:2] != (M, 3):
            raise ValueError(f"point_jac shape {self.point_jac.shape} != ({M}, 3, nv)")
        if self.probe_jac_obj.shape != (C, M, 3, 6):
            raise ValueError(f"probe_jac_obj shape {self.probe_jac_obj.shape} != ({C}, {M}, 3, 6)")
        self.point_jac.flags.writeable = False
        self.probe_jac_obj.flags.writeable = False
