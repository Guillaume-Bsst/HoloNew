"""Config of the ``targets`` stage — the style KNOBS (frozen dataclasses) AND the robot style RECIPE,
co-located in one place (stdlib-only, torch-free, importable anywhere).

Two kinds of content live here, kept visually apart:

1. KNOBS — ``StyleConfig`` / ``TargetsConfig`` (frozen dataclasses). ``TargetsConfig()`` IS the
   default; override inline (``TargetsConfig(style=StyleConfig(scale_arms=0.85))``). These are what a
   user legitimately tunes: the GMR morphological SCALE values and the reference heights.

2. The robot-keyed style RECIPE — ``SMPL_BODY_INDEX`` / ``ROOT_BODY`` / ``ARM_BODIES`` /
   ``_STYLE_TABLE`` + ``style_table()``. This is fixed reference DATA (which robot link tracks which
   SMPL body, with which per-link pos/rot offsets): adding a robot is a data entry, never a code
   change — like ``prepare/load/robot.CORRESPONDENCE_REST_POSE``. NOT a knob. (Per-link TRACKING
   WEIGHTS are a SOLVER concern and live in ``solve``, not here.)

(``prepare/config.py`` keeps knobs and data strictly apart; here the style data is folded in so the
whole style surface is one file. The dataclasses stay pure knobs; the recipe stays plain data below.)

Ported from HoloNew ``test_socp/tables.py`` (``IK_MATCH_TABLE1`` + ``HUMAN_SCALE_TABLE``). The GMR
human bodies are addressed by NAME and resolved to OUR SMPL-X bone order (``SMPL_BODY_INDEX``, derived
from ``prepare/load/smpl.SMPLX_BODY_JOINTS``) — NOT the 52-joint MuJoCo layout V1 indexed into.
``style/build.py`` does the float64 math.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# =============================================================================== KNOBS
@dataclass(frozen=True)
class StyleConfig:
    """Knobs of the STYLE channel (``targets/style``): the GMR morphological SCALE values + the
    reference heights feeding ``style.build``'s SCALE/OFFSET steps. The per-link RECIPE (body map,
    tracking weights, alignment offsets) is robot DATA below (``_STYLE_TABLE``), never a knob."""

    human_height_assumption: float = 1.8   # GMR reference subject height (m): ratio = stature / this
    ground_height: float = 0.0             # GMR target grounding plane z (the OFFSET subtracts it)
    scale_torso_legs: float = 0.9          # morphological scale of the torso + leg bodies (incl. the
                                           # pelvis anchor) — GMR smplx_to_g1 ``HUMAN_SCALE_TABLE``
    scale_arms: float = 0.8                # morphological scale of the arm bodies (shoulder/elbow/wrist)

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
    """All knobs of the ``targets`` step, composed — the single object ``pipeline`` receives; each op
    reads only its sub-config. ``style`` = recette + scalaires morpho ; ``scene_scale`` = la similarité
    de scène partagée par style + interaction (placement). L'``InteractionContext.margin`` reste un
    knob ``prepare``."""

    style: StyleConfig = field(default_factory=StyleConfig)
    scene_scale: SceneScaleConfig = field(default_factory=SceneScaleConfig)


# =============================================================================== ROBOT STYLE RECIPE (data)
# A fixed SMPL-skeleton fact (like the frame conventions): GMR human-body NAME -> bone INDEX in OUR
# SMPL-X bone order (``prepare/load/smpl.SMPLX_BODY_JOINTS``, i.e. the order of ``FramePose.bone_rot``
# / ``bone_pos``). GMR "left_foot" / "right_foot" are the leg-chain ANKLE joints (the toe is reached
# via the table's pos_offset), so they resolve to L_Ankle (7) / R_Ankle (8), NOT the SMPL foot/toe
# bones (10 / 11). Hardcoded so ``targets`` never imports the torch-pulling ``load/smpl``.
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

ROOT_BODY = "pelvis"            # the SCALE anchor (V1 ``HUMAN_ROOT_NAME``)

# Which GMR bodies are ARMS: in the morphological SCALE they take ``StyleConfig.scale_arms``; every
# other tracked body (torso + legs, incl. the pelvis anchor) takes ``StyleConfig.scale_torso_legs``.
# The split is a FIXED fact of the GMR smplx_to_g1 recipe (V1 ``HUMAN_SCALE_TABLE``: torso/legs 0.9,
# arms 0.8) — only the two scale VALUES are knobs (``StyleConfig`` above).
ARM_BODIES: frozenset[str] = frozenset({
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
})

# G1 GMR style recipe (ported from V1 ``IK_MATCH_TABLE1``): robot_link -> (smpl_body, pos_offset xyz,
# rot_offset wxyz). ``pos_offset`` is added in the RE-ORIENTED body frame (corrects the human reference
# point vs the robot link point); ``rot_offset`` (w FIRST) is composed onto the body orientation
# (SMPL-X vs G1 zero-pose alignment per body). Insertion order is the link order of ``StyleTargets``
# (14 tracked G1 links). The per-link TRACKING WEIGHTS (V1 ``w_p`` / ``w_r``) are a SOLVER concern and
# are deliberately NOT kept here — ``solve`` defines its cost gains in its own config when built.
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
    """GMR style recipe for ``robot_name`` (robot_link -> (smpl_body, pos_offset, rot_offset); raises
    if undefined). Mirrors ``load/robot.correspondence_rest_angles``."""
    try:
        return _STYLE_TABLE[robot_name]
    except KeyError:
        raise ValueError(f"no style table for robot {robot_name!r} — add an entry to "
                         f"_STYLE_TABLE") from None
