"""Config de l'étage ``targets`` — les KNOBS de style (dataclasses gelées) ET la RECETTE de style
robot, co-localisés en un seul endroit (stdlib-only, torch-free, importable partout).

Deux types de contenu vivent ici, maintenus visuellement séparés :

1. KNOBS — ``StyleConfig`` / ``TargetsConfig`` (dataclasses gelées). ``TargetsConfig()`` EST le
   défaut ; override inline (``TargetsConfig(style=StyleConfig(scale_arms=0.85))``). Ce sont ceux qu'un
   utilisateur accorde légitimement : les valeurs SCALE morphologiques GMR et les hauteurs de référence.

2. La RECETTE de style clé-robot — ``SMPL_BODY_INDEX`` / ``ROOT_BODY`` / ``ARM_BODIES`` /
   ``_STYLE_TABLE`` + ``style_table()``. C'est une DATA de référence fixe (quel lien robot suit quel
   corps SMPL, avec quels offsets pos/rot par lien) : ajouter un robot est une entrée de données,
   jamais un changement de code — comme ``prepare/load/robot.CORRESPONDENCE_REST_POSE``. PAS un knob.
   (Les POIDS DE SUIVI par lien sont une préoccupation SOLVER et vivent dans ``solve``, pas ici.)

(``prepare/config.py`` garde les knobs et les données strictement séparés ; ici les données de style
sont repliées pour que toute la surface de style soit un fichier. Les dataclasses restent des knobs purs ;
la recette reste des données brutes en dessous.)

Porté de HoloNew ``test_socp/tables.py`` (``IK_MATCH_TABLE1`` + ``HUMAN_SCALE_TABLE``). Les corps
humains GMR sont adressés par NOM et résolus à NOTRE ordre d'os SMPL-X (``SMPL_BODY_INDEX``, dérivé de
``prepare/load/smpl.SMPLX_BODY_JOINTS``) — PAS la disposition MuJoCo 52 articulations V1 indexée.
``style/build.py`` fait les mathématiques float64.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# =============================================================================== KNOBS
@dataclass(frozen=True)
class StyleConfig:
    """Knobs du canal STYLE (``targets/style``) : les valeurs SCALE morphologiques GMR + les hauteurs
    de référence alimentant les étapes SCALE/OFFSET de ``style.build``. La RECETTE par lien (carte des
    corps, poids de suivi, offsets d'alignement) est une DATA robot ci-dessous (``_STYLE_TABLE``), jamais
    un knob."""

    human_height_assumption: float = 1.8   # taille du sujet de référence GMR (m) : ratio = stature / ceci
    ground_height: float = 0.0             # z du plan de mise au sol cible GMR (l'OFFSET le soustrait)
    scale_torso_legs: float = 0.9          # scale morphologique des bodies torse + jambes (incl. l'ancre
                                           # du pelvis) — GMR smplx_to_g1 ``HUMAN_SCALE_TABLE``
    scale_arms: float = 0.8                # scale morphologique des bodies bras (épaule/coude/poignet)

    def __post_init__(self) -> None:
        if self.human_height_assumption <= 0.0:
            raise ValueError(f"human_height_assumption must be > 0, got {self.human_height_assumption}")
        if self.scale_torso_legs <= 0.0:
            raise ValueError(f"scale_torso_legs must be > 0, got {self.scale_torso_legs}")
        if self.scale_arms <= 0.0:
            raise ValueError(f"scale_arms must be > 0, got {self.scale_arms}")


@dataclass(frozen=True)
class SceneScaleConfig:
    """Échelle de scène (placement), partagée par ``style`` et ``interaction`` et appliquée en étape
    finale sur les RÉFÉRENCES (jamais avant l'éval des contacts). Similarité diagonale ancrée
    statiquement : xy autour de l'origine monde, z autour du sol (``StyleConfig.ground_height``).

    ``None`` => le facteur de cet axe est ``ratio = stature / StyleConfig.human_height_assumption``
    (le MÊME ratio que le style, pour la cohérence style↔interaction). Un float = facteur fixe.
    Défaut ``(None, None)`` => ``ratio`` partout. ``scale_xy=1.0, scale_z=None`` reproduit le
    comportement style natif (xy non scalé, z par ``ratio``)."""

    scale_xy: float | None = None   # facteur xy autour de l'origine ; None -> ratio
    scale_z: float | None = None    # facteur z autour du sol ; None -> ratio

    def __post_init__(self) -> None:
        if self.scale_xy is not None and self.scale_xy <= 0.0:
            raise ValueError(f"scale_xy must be > 0 when set, got {self.scale_xy}")
        if self.scale_z is not None and self.scale_z <= 0.0:
            raise ValueError(f"scale_z must be > 0 when set, got {self.scale_z}")


@dataclass(frozen=True)
class TargetsConfig:
    """Tous les knobs de l'étape ``targets``, composés — l'objet unique que ``pipeline`` reçoit ; chaque op
    lit uniquement sa sous-config. ``style`` = recette + scalaires morpho ; ``scene_scale`` = la similarité
    de scène partagée par style + interaction (placement). L'``InteractionContext.margin`` reste un
    knob ``prepare``."""

    style: StyleConfig = field(default_factory=StyleConfig)
    scene_scale: SceneScaleConfig = field(default_factory=SceneScaleConfig)


# =============================================================================== RECETTE DE STYLE ROBOT (données)
# Un fait SMPL-squelette fixe (comme les conventions de frame) : NOM du corps humain GMR → INDEX d'os dans NOTRE
# ordre d'os SMPL-X (``prepare/load/smpl.SMPLX_BODY_JOINTS``, i.e. l'ordre de ``FramePose.bone_rot``
# / ``bone_pos``). GMR "left_foot" / "right_foot" sont les articulations ANKLE de la chaîne jambe (l'orteil est
# atteint via le pos_offset de la table), donc ils se résolvent en L_Ankle (7) / R_Ankle (8), PAS les os
# pied/orteil SMPL (10 / 11). Hardcodé pour que ``targets`` n'importe jamais le ``load/smpl`` qui tire torch.
SMPL_BODY_INDEX: dict[str, int] = {
    "pelvis": 0,
    "left_hip": 1, "right_hip": 2,
    "left_knee": 4, "right_knee": 5,
    "left_foot": 7, "right_foot": 8,        # L_Ankle / R_Ankle
    "spine3": 9,
    "left_shoulder": 16, "right_shoulder": 17,
    "left_elbow": 18, "right_elbow": 19,
    "left_wrist": 20, "right_wrist": 21,
}

ROOT_BODY = "pelvis"            # l'ancre SCALE (V1 ``HUMAN_ROOT_NAME``)

# Quels corps GMR sont ARMS : dans la SCALE morphologique ils prennent ``StyleConfig.scale_arms`` ; tous les autres
# corps suivis (torse + jambes, incl. l'ancre du bassin) prennent ``StyleConfig.scale_torso_legs``.
# La division est un fait FIXE de la recette smplx_to_g1 GMR (V1 ``HUMAN_SCALE_TABLE`` : torse/jambes 0.9,
# bras 0.8) — seules les deux VALEURS d'échelle sont des knobs (``StyleConfig`` ci-dessus).
ARM_BODIES: frozenset[str] = frozenset({
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
})

# Recette de style G1 GMR (portée de V1 ``IK_MATCH_TABLE1``) : robot_link -> (smpl_body, pos_offset xyz,
# rot_offset wxyz). ``pos_offset`` est ajouté dans le frame body RÉ-ORIENTÉ (corrige le point de référence
# humain vs le point du link robot) ; ``rot_offset`` (w EN PREMIER) est composé sur l'orientation du body
# (alignement zéro-pose SMPL-X vs G1 par body). L'ordre d'insertion est l'ordre des links de ``StyleTargets``
# (14 links G1 suivis). Les POIDS de SUIVI par link (V1 ``w_p`` / ``w_r``) sont une affaire de SOLVEUR et
# ne sont délibérément PAS gardés ici — ``solve`` définit ses gains de coût dans sa propre config à sa construction.
_H = 0.4267755048530407
_K = 0.5637931078484661
_STYLE_TABLE: dict[str, dict[str, tuple]] = {
    "g1": {
        "pelvis":                  ("pelvis",        (0.0,  0.0,  0.0), (0.5, -0.5, -0.5, -0.5)),
        "left_hip_roll_link":      ("left_hip",      (0.0,  0.0,  0.0), (_H, -_K, -_K, -_H)),
        "left_knee_link":          ("left_knee",     (0.0,  0.0,  0.0), (0.5, -0.5, -0.5, -0.5)),
        "left_ankle_roll_link":    ("left_foot",     (0.0,  0.02, 0.0), (0.5, -0.5, -0.5, -0.5)),
        "right_hip_roll_link":     ("right_hip",     (0.0,  0.0,  0.0), (_H, -_K, -_K, -_H)),
        "right_knee_link":         ("right_knee",    (0.0,  0.0,  0.0), (0.5, -0.5, -0.5, -0.5)),
        "right_ankle_roll_link":   ("right_foot",    (0.0, -0.02, 0.0), (0.5, -0.5, -0.5, -0.5)),
        "torso_link":              ("spine3",        (0.0,  0.0,  0.0), (0.5, -0.5, -0.5, -0.5)),
        "left_shoulder_yaw_link":  ("left_shoulder", (0.0,  0.0,  0.0), (0.70710678, 0.0, -0.70710678, 0.0)),
        "left_elbow_link":         ("left_elbow",    (0.0,  0.0,  0.0), (1.0, 0.0, 0.0, 0.0)),
        "left_wrist_yaw_link":     ("left_wrist",    (0.0,  0.0,  0.0), (1.0, 0.0, 0.0, 0.0)),
        "right_shoulder_yaw_link": ("right_shoulder",(0.0,  0.0,  0.0), (0.0, 0.70710678, 0.0, 0.70710678)),
        "right_elbow_link":        ("right_elbow",   (0.0,  0.0,  0.0), (0.0, 0.0, 0.0, -1.0)),
        "right_wrist_yaw_link":    ("right_wrist",   (0.0,  0.0,  0.0), (0.0, 0.0, 0.0, -1.0)),
    },
}


def style_table(robot_name: str) -> dict[str, tuple]:
    """Recette de style GMR pour ``robot_name`` (robot_link -> (smpl_body, pos_offset, rot_offset) ; lève
    si non défini). Reflète ``load/robot.correspondence_rest_angles``."""
    try:
        return _STYLE_TABLE[robot_name]
    except KeyError:
        raise ValueError(f"no style table for robot {robot_name!r} — add an entry to "
                         f"_STYLE_TABLE") from None
