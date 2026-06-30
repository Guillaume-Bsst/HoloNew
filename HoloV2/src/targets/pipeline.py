"""Orchestrateur targets/ — construction par frame des cibles.

Compose les ops purs (style + pose/eval/transport d'interaction) dans ``FrameTargets``. L'entrée est la
surface publique prepare ``(GroundedScene, InteractionContext)`` plus la ``RobotSpec`` (elle clé la table
de style) — l'ancrage ``Calibration`` existe dans ``grounded.calibration``, le corps du sujet dans
``grounded.body``. ``process_frame`` et ``trace_frame`` partagent UN cœur de flux de données (``_build_frame``)
donc les chemins maigres et instrumentés ne peuvent jamais dériver ; les spans ``prof`` vivent dans ce cœur
(l'orchestrateur), jamais dans les ops purs. Voir docs/TARGETS.md, VIZ.md, OBS.md.

Les knobs d'étage vivent sur ``cfg`` (``TargetsConfig``, ``targets/config.py``) : actuellement seul ``style``
porte des knobs — ``cfg.style`` est remis à ``style.build``. Le knob par frame d'interaction
(``margin``) reste sur ``InteractionContext`` (une sortie ``prepare``), pas dans ``cfg``.
"""
from __future__ import annotations

import numpy as np

from ..obs import NULL
from ..prepare.contracts import GroundedScene, InteractionContext, RobotSpec
from .config import TargetsConfig
from .contracts import FramePose, FrameTargets, FrameTrace
from .scale import apply_scene_scale, resolve_scale, scale_ground_channels
from .interaction import (environment_interaction_targets, eval_fields, pose_cloud,
                          robot_interaction_targets, transport)
from . import style


def _pose7_to_Rt(pose7: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Une pose du monde ``[x, y, z, qw, qx, qy, qz]`` (quaternion wxyz, supposé unitaire) →
    ``(R (3,3), t (3,))``. Le chemin unique quaternion→matrice pour poser des objets (gardé sans torch)."""
    x, y, z, qw, qx, qy, qz = (float(v) for v in pose7)
    rot = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz),     2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy),     2 * (qy * qz + qw * qx),     1 - 2 * (qx * qx + qy * qy)],
    ])
    return rot, np.array([x, y, z])


def frame_pose(grounded: GroundedScene, f: int) -> FramePose:
    """Transformations mondiales par frame, calculées UNE FOIS et partagées par LES DEUX traitements :
    os SMPL (R, t) de la FK du corps et (R, t) de chaque objet de sa pose7 ancrée. ``interaction`` pose
    ses nuages à partir de ceux-ci (et réutilise l'objet (R, t) comme frame du canal objet dans l'éval) ;
    ``style`` lit aussi l'os mappé (R, t) (son suivi GMR suit la pose monde de l'os, pas les articulations
    démo). ``body is None`` (positions uniquement) ⇒ pas d'os."""
    if grounded.body is not None:
        bone_rot, bone_pos = grounded.body.bone_transforms(grounded.smpl_params, f)
    else:
        bone_rot, bone_pos = np.zeros((0, 3, 3)), np.zeros((0, 3))
    n = len(grounded.object_poses)
    object_rot, object_pos = np.empty((n, 3, 3)), np.empty((n, 3))
    for i, poses in enumerate(grounded.object_poses):
        object_rot[i], object_pos[i] = _pose7_to_Rt(poses[f])
    return FramePose(bone_rot=bone_rot, bone_pos=bone_pos,
                     object_rot=object_rot, object_pos=object_pos)


def _build_frame(grounded: GroundedScene, ctx: InteractionContext, robot: RobotSpec, f: int,
                 cfg: TargetsConfig = TargetsConfig(), prof=NULL):
    """Exécuter chaque op pur d'une frame UNE FOIS, retournant tous les intermédiaires. La source unique du
    flux de données par frame, partagée par ``process_frame`` (garde uniquement les cibles) et ``trace_frame``
    (garde tout). L'interaction est un flux unidirectionnel : pose → eval → transport → assemble.
    L'instrumentation (spans) vit ici, dans l'orchestrateur — les ops purs restent propres. ``robot`` clé
    la recette de style (et porte l'identité du robot) ; l'échelle morphologique utilise ``cfg.style`` et
    la ``body.stature`` du sujet."""
    if grounded.body is None:
        # Le pipeline est basé sur les os : ``style`` suit les os SMPL et ``interaction`` pose le
        # nuage humain, tous deux via la FK du corps. Une source positions uniquement (``body is None``)
        # est un placeholder structurel dans le contrat, pas un chemin câblé — échouer explicitement ici
        # plutôt qu'avec une ``AttributeError`` nue sur ``grounded.body.stature``.
        raise ValueError("targets pipeline requires a parametric body (GroundedScene.body): style is "
                         "bone-based and interaction poses the SMPL cloud; positions-only is not wired")
    with prof.span("frame", f=f):
        with prof.span("pose"):
            pose = frame_pose(grounded, f)
        with prof.span("style"):
            style_t = style.build(pose, robot, grounded.body.stature, cfg.style, cfg.scene_scale)
        with prof.span("interaction.pose"):
            human_world = pose_cloud(ctx.human_cloud, pose.bone_rot, pose.bone_pos)
            object_worlds = tuple(
                pose_cloud(c, pose.object_rot[i][None], pose.object_pos[i][None])
                for i, c in enumerate(ctx.object_clouds))
        with prof.span("interaction.eval", n_channels=len(ctx.channels), n_points=ctx.human_cloud.n_points):
            human_field = eval_fields(human_world, ctx.channels, pose.object_rot, pose.object_pos, ctx.margin)
            object_fields = tuple(
                eval_fields(ow, ctx.channels, pose.object_rot, pose.object_pos, ctx.margin, self_idx=i)
                for i, ow in enumerate(object_worlds))
        with prof.span("interaction.transport"):
            robot_field = transport(human_field, ctx.correspondence)
        ratio = grounded.body.stature / cfg.style.human_height_assumption
        s_xy, s_z = resolve_scale(cfg.scene_scale, ratio)
        ground_h = cfg.style.ground_height
        scaled_object_pos = apply_scene_scale(pose.object_pos, s_xy, s_z, ground_h)  # (N, 3) centre objet
        ground_idx = tuple(c for c, ch in enumerate(ctx.channels) if ch.object_idx is None)
        robot_field = scale_ground_channels(robot_field, ground_idx, s_xy, s_z, ground_h)
        object_fields = tuple(
            scale_ground_channels(of, ground_idx, s_xy, s_z, ground_h) for of in object_fields)
        targets = FrameTargets(
            style=style_t,
            robot_interaction=robot_interaction_targets(robot_field),
            env_interaction=environment_interaction_targets(object_fields),
            object_rot=pose.object_rot,
            object_pos=scaled_object_pos,
        )
        return pose, human_world, object_worlds, human_field, targets


def process_frame(grounded: GroundedScene, ctx: InteractionContext, robot: RobotSpec, f: int,
                  cfg: TargetsConfig = TargetsConfig(), prof=NULL) -> FrameTargets:
    """Une frame -> ``FrameTargets`` (maigre, chemin prod)."""
    *_, targets = _build_frame(grounded, ctx, robot, f, cfg, prof)
    return targets


def trace_frame(grounded: GroundedScene, ctx: InteractionContext, robot: RobotSpec, f: int,
                cfg: TargetsConfig = TargetsConfig(), prof=NULL) -> FrameTrace:
    """Mêmes ops purs que ``process_frame``, intermédiaires gardés -> ``FrameTrace`` (le seam pour ``viz``)."""
    pose, human_world, object_worlds, human_field, targets = _build_frame(grounded, ctx, robot, f, cfg, prof)
    return FrameTrace(pose=pose, human_cloud_world=human_world, object_clouds_world=object_worlds,
                      human_field=human_field, targets=targets)


def run_sequence(grounded: GroundedScene, ctx: InteractionContext, robot: RobotSpec,
                 cfg: TargetsConfig = TargetsConfig(), prof=NULL) -> list[FrameTargets]:
    """Piloter toutes les frames : la boucle online ``for f: process_frame``. Un batch vectorisé sur T
    (mêmes ops array-oriented, T sur l'axe de tête) est une optimisation ultérieure — voir la note sur le
    batch ``bone_transforms`` dans ``load/smpl.py``."""
    with prof.span("sequence", T=grounded.n_frames):
        return [process_frame(grounded, ctx, robot, f, cfg, prof) for f in range(grounded.n_frames)]
