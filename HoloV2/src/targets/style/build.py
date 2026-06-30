"""style.build — os SMPL par frame → ``StyleTargets`` (suivi posture/style GMR, G1-ready).

Le canal "comment le corps devrait bouger" indépendant des objets : chaque link robot suivi est tiré
vers UN SEUL corps SMPL après une SCALE morphologique (les proportions du sujet, en frame pelvis-local)
et un OFFSET de frame par link (alignement SMPL-X vs pose zéro robot + un petit décalage de point de
référence). Porté de HoloNew ``test_socp/preprocess`` (``scale`` puis ``offset``) ; les scalaires
ajustables sont la ``StyleConfig`` de ``targets/config`` et la recette spécifique au robot est la
``style_table`` clé-robot (aussi ``targets/config``).

Lit l'os (R, t) de chaque corps mappé à partir de la ``FramePose`` partagée (J_bones, ordre SMPL-X) —
PAS les articulations démo : les transformations mondiales des os portent déjà l'orientation d'articulation
que le terme de style suit, et partager ``FramePose`` évite un recalcul. Pur, float64, sans torch (scipy
uniquement) ; pas d'I/O, pas de mutation d'entrée, sortie numpy gelée.

La chute FLOOR au niveau du clip V1 (soustraire le z le plus bas du corps mappé sur TOUTE la séquence)
n'est intentionnellement PAS appliquée ici : le style reste strictement par frame. Si les pieds flottent,
une chute de clip précomputée doit être ré-ajoutée en amont plus tard — pas comme un passage par frame.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..config import ARM_BODIES, ROOT_BODY, SMPL_BODY_INDEX, SceneScaleConfig, StyleConfig, style_table
from ..scale import resolve_scale
from ..contracts import FramePose, StyleTargets
from ...prepare.contracts import RobotSpec


def _quat_wxyz_to_R(quat_wxyz) -> np.ndarray:
    """Quaternion (4,) wxyz → matrice de rotation (3, 3) (scipy est xyzw ; gardé sans torch)."""
    q = np.asarray(quat_wxyz, np.float64)
    return R.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def _R_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    """Matrice de rotation (3, 3) → quaternion (4,) wxyz."""
    q = R.from_matrix(np.asarray(rot, np.float64)).as_quat()           # xyzw
    return q[[3, 0, 1, 2]]


def build(pose: FramePose, robot: RobotSpec, stature: float,
          cfg: StyleConfig = StyleConfig(), scene: SceneScaleConfig = SceneScaleConfig()) -> StyleTargets:
    """Une frame d'os SMPL → ``StyleTargets`` pour les links suivis du ``robot``.

    ``pose`` porte l'os SMPL (R, t) par frame (J_bones, ordre SMPL-X) ; ``robot.name`` sélectionne la
    RECETTE de style GMR (``config.style_table``) ; ``stature`` (la hauteur au repos du sujet,
    ``GroundedScene.body.stature``) définit le ``ratio`` morphologique = stature /
    cfg.human_height_assumption``. ``cfg`` (``StyleConfig``) porte les valeurs SCALE ajustables + hauteurs.
    Par link suivi : SCALE le corps source en frame pelvis-local, puis OFFSET (composer rot_offset, ajouter
    pos_offset dans le frame ré-orienté).
    """
    table = style_table(robot.name)
    ratio = stature / cfg.human_height_assumption

    bone_rot = np.asarray(pose.bone_rot, np.float64)                   # (J_bones, 3, 3)
    bone_pos = np.asarray(pose.bone_pos, np.float64)                   # (J_bones, 3)

    # Ancre SCALE : la position mondiale du root (pelvis) place rigidement le squelette entièrement scalé.
    # Défaut SceneScaleConfig() (None,None) → s_xy=ratio, s_z=ratio (xy ET z scalés par ratio).
    # scale_xy=1.0, scale_z=None reproduit le natif (xy brut, z morphologique·ratio). Les proportions du
    # corps (pelvis-local) restent inchangées sinon.
    root_pos = bone_pos[SMPL_BODY_INDEX[ROOT_BODY]]                    # (3,)
    # PLACEMENT du root via l'échelle de scène (None -> ratio) ; xy autour de l'origine, z autour du
    # sol. Le morphologique du pelvis (scale_torso_legs, le pelvis est torse/jambes) reste sur z.
    # scale_xy=1.0, scale_z=None reproduit le natif : xy brut, z = scale_torso_legs * ratio.
    s_xy, s_z = resolve_scale(scene, ratio)
    base_z = cfg.scale_torso_legs * s_z
    scaled_root = np.array([root_pos[0] * s_xy, root_pos[1] * s_xy,
                            cfg.ground_height + (root_pos[2] - cfg.ground_height) * base_z])
    ground = np.array([0.0, 0.0, cfg.ground_height])

    links = tuple(table.keys())
    L = len(links)
    position = np.empty((L, 3), np.float64)
    orientation = np.empty((L, 4), np.float64)

    for i, link in enumerate(links):
        body, pos_off, rot_off = table[link]
        idx = SMPL_BODY_INDEX[body]
        src_rot = bone_rot[idx]                                        # (3, 3) orientation monde
        src_pos = bone_pos[idx]                                        # (3,)

        # SCALE : morphe le vecteur pelvis-local du corps par ``SCALE[body] * ratio``, ré-ancré sur le
        # root scalé (l'orientation n'est pas touchée par l'échelle).
        if body == ROOT_BODY:
            scaled_pos = scaled_root
        else:
            s = (cfg.scale_arms if body in ARM_BODIES else cfg.scale_torso_legs) * ratio
            scaled_pos = (src_pos - root_pos) * s + scaled_root

        # OFFSET : composer rot_offset sur l'orientation, puis ajouter pos_offset exprimé dans le frame
        # du corps ré-orienté (moins cfg.ground_height*z, une no-op à sol 0). Miroir V1 ``offset``.
        updated_rot = src_rot @ _quat_wxyz_to_R(rot_off)              # (3, 3)
        world_pos = scaled_pos + updated_rot @ (np.asarray(pos_off, np.float64) - ground)

        position[i] = world_pos
        orientation[i] = _R_to_quat_wxyz(updated_rot)

    return StyleTargets(link_names=links, position=position, orientation=orientation)
