"""targets/ orchestrator — per-frame construction of the cibles.

Composes the pure ops (style + interaction's pose/eval/transport) into ``FrameTargets``. Input is the
prepare public surface ``(GroundedScene, InteractionContext)`` plus the ``RobotSpec`` (it keys the style
table) — the grounding ``Calibration`` rides inside ``grounded.calibration``, the subject body inside
``grounded.body``. ``process_frame`` and ``trace_frame`` share ONE dataflow core (``_build_frame``)
so the lean and the instrumented paths can never drift; the ``prof`` spans live in that core (the
orchestrator), never in the pure ops. See docs/TARGETS.md, VIZ.md, OBS.md.

No targets ``config`` yet: the only per-frame knob (the field ``margin``) is carried by the
``InteractionContext``; a ``targets/config.py`` is added when a real knob appears.
"""
from __future__ import annotations

import numpy as np

from ..obs import NULL
from ..prepare.contracts import GroundedScene, InteractionContext, RobotSpec
from .contracts import FramePose, FrameTargets, FrameTrace
from .interaction import (environment_interaction_targets, eval_fields, pose_cloud,
                          robot_interaction_targets, transport)
from . import style


def _pose7_to_Rt(pose7: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One world pose ``[x, y, z, qw, qx, qy, qz]`` (quaternion wxyz, assumed unit) -> ``(R (3,3),
    t (3,))``. The single quaternion->matrix path for object posing (kept torch-free)."""
    x, y, z, qw, qx, qy, qz = (float(v) for v in pose7)
    rot = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz),     2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy),     2 * (qy * qz + qw * qx),     1 - 2 * (qx * qx + qy * qy)],
    ])
    return rot, np.array([x, y, z])


def frame_pose(grounded: GroundedScene, f: int) -> FramePose:
    """Per-frame world transforms, computed ONCE and shared by BOTH treatments: SMPL bone (R, t) from
    the body's FK and each object's (R, t) from its grounded pose7. ``interaction`` poses its clouds
    from these (and reuses the object (R, t) as each object channel's frame in the eval); ``style``
    reads the mapped bone (R, t) too (its GMR tracking follows the bone world pose, not the demo
    joints). ``body is None`` (positions-only) => no bones."""
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


def _build_frame(grounded: GroundedScene, ctx: InteractionContext, robot: RobotSpec, f: int, prof=NULL):
    """Run every pure op of one frame ONCE, returning all intermediates. The single source of the
    per-frame dataflow, shared by ``process_frame`` (keeps only the targets) and ``trace_frame``
    (keeps everything). Interaction is a one-way flow: pose -> eval -> transport -> assemble. The
    instrumentation (spans) lives here, in the orchestrator — the pure ops stay clean. ``robot`` keys
    the style table (and carries the robot identity); the morphological scale uses the subject's
    ``body.stature``."""
    if grounded.body is None:
        # The pipeline is bone-based: ``style`` tracks the SMPL bones and ``interaction`` poses the
        # human cloud, both via the body's FK. A positions-only source (``body is None``) is a
        # structural placeholder in the contract, not a wired path — fail explicitly here rather than
        # with a bare ``AttributeError`` on ``grounded.body.stature``.
        raise ValueError("targets pipeline requires a parametric body (GroundedScene.body): style is "
                         "bone-based and interaction poses the SMPL cloud; positions-only is not wired")
    with prof.span("frame", f=f):
        with prof.span("pose"):
            pose = frame_pose(grounded, f)
        with prof.span("style"):
            style_t = style.build(pose, robot, grounded.body.stature)
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
        targets = FrameTargets(
            style=style_t,
            robot_interaction=robot_interaction_targets(robot_field),
            env_interaction=environment_interaction_targets(object_fields),
            object_rot=pose.object_rot,
            object_pos=pose.object_pos,
        )
        return pose, human_world, object_worlds, human_field, targets


def process_frame(grounded: GroundedScene, ctx: InteractionContext, robot: RobotSpec, f: int,
                  prof=NULL) -> FrameTargets:
    """One frame -> ``FrameTargets`` (lean, prod path)."""
    *_, targets = _build_frame(grounded, ctx, robot, f, prof)
    return targets


def trace_frame(grounded: GroundedScene, ctx: InteractionContext, robot: RobotSpec, f: int,
                prof=NULL) -> FrameTrace:
    """Same pure ops as ``process_frame``, intermediates kept -> ``FrameTrace`` (the seam for ``viz``)."""
    pose, human_world, object_worlds, human_field, targets = _build_frame(grounded, ctx, robot, f, prof)
    return FrameTrace(pose=pose, human_cloud_world=human_world, object_clouds_world=object_worlds,
                      human_field=human_field, targets=targets)


def run_sequence(grounded: GroundedScene, ctx: InteractionContext, robot: RobotSpec,
                 prof=NULL) -> list[FrameTargets]:
    """Drive all frames: the online loop ``for f: process_frame``. A vectorised batch over T (same
    array-oriented ops, T on the leading axis) is a later optimisation — see the ``bone_transforms``
    batch note in ``load/smpl.py``."""
    with prof.span("sequence", T=grounded.n_frames):
        return [process_frame(grounded, ctx, robot, f, prof) for f in range(grounded.n_frames)]
