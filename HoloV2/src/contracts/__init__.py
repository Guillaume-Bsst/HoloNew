"""Shared data contracts for the HoloV2 retargeting pipeline.

These types cross the module boundaries between ``prepare`` + ``targets`` (everything that does
NOT depend on the robot configuration the solver optimises) and ``solve`` (which does).

Every artifact that crosses a module boundary is defined here as a FROZEN dataclass of numpy
arrays (Structure-of-Arrays, channel-first where speed matters) or as an interface PROTOCOL.
DATA + PROTOCOLS only — no logic, no I/O, no heavy deps. Modules depend on these types, never
on each other's code, which keeps the dependency graph acyclic. The definitions are split by
domain into the submodules of this package (protocols, inputs, motion, scene, fields, assets,
targets) and re-exported here, so every call site keeps importing from ``contracts`` directly.

Two distinct inputs:
- ``SceneSpec`` = WHAT to run (data identity: dataset, sequence, robot, model dirs).
- the step CONFIG = HOW (algorithm knobs). It lives at the TOP of the repo, apart from both the
  code and the data: ``config_types/`` (the dataclass SCHEMAS, one module per step) +
  ``config_values/`` (the named-preset factory). NOT here: ``contracts`` holds only the data that
  flows through the pipeline. Cache keys mix the relevant config subset with the ``SceneSpec`` identity.

Conventions
-----------
- Per-frame is the canonical unit. The current target is OFFLINE replay: ``process_frame``
  indexes a loaded ``GroundedScene`` at frame ``f``. Live teleoperation (a single ``RawFrame``
  fed per tick) is a future variant on the same pure ops — not yet a contract.
- Field-eval result arrays (``ContactField`` / ``MultiChannelField``) are channel-first
  ``(C, P)`` = C channels over P points (per-channel ops
  contiguous). C = number of ``Channel`` (ground + N objects).
- Two joint sets, kept distinct: ``J_demo`` (the dataset's joints, used by ``style``) and
  ``J_bones`` (the SMPL skeleton, used to pose clouds). Never conflate them.
- Quaternions are wxyz. Rigid poses are ``(x, y, z, qw, qx, qy, qz)``.
- Arrays are read-only by convention (``frozen`` freezes the binding, not the buffer).
"""
from __future__ import annotations

from .assets import CorrespondenceTable, InteractionContext, PointCloud
from .fields import Channel, ContactField, MultiChannelField, SDF
from .inputs import RobotSpec, SceneSpec
from .motion import RawMotion, SmplParams
from .protocols import AssetBuilder, BodyModel, RobotModel
from .scene import Calibration, GroundedScene, ObjectMesh
from .targets import (EnvironmentInteractionTargets, FramePose, FrameTargets, FrameTrace,
                      RobotInteractionTargets, StyleTargets)

__all__ = [
    # protocols
    "BodyModel", "RobotModel", "AssetBuilder",
    # inputs (data identity)
    "RobotSpec", "SceneSpec",
    # motion
    "SmplParams", "RawMotion",
    # scene & calibration
    "ObjectMesh", "Calibration", "GroundedScene",
    # field evaluation
    "ContactField", "MultiChannelField", "Channel", "SDF",
    # build-once assets
    "PointCloud", "CorrespondenceTable", "InteractionContext",
    # per-frame targets + shared state/trace
    "StyleTargets", "RobotInteractionTargets", "EnvironmentInteractionTargets",
    "FrameTargets", "FramePose", "FrameTrace",
]
