"""Data contracts of the ``targets`` stage — its PUBLIC type surface.

The per-frame field-evaluation results and target artifacts (the ``targets`` -> ``solve`` contract),
plus the shared per-frame pose state and the viz trace. FROZEN dataclasses of numpy arrays, numpy-only
(no logic, no I/O), so this module is importable everywhere.

``targets`` consumes the upstream ``prepare`` contracts (``from ..prepare.contracts import ...``) and
exposes these as its own public types; ``solve`` and ``viz`` import their inputs from here. The
pipeline is linear (prepare -> targets -> solve), so each stage owns its contracts and depends only on
the public types of the stage upstream — the dependency graph stays acyclic.

Channel-first convention: ``ContactField`` / ``MultiChannelField`` arrays are ``(C, P)`` = C channels
over P points (per-channel ops contiguous). C = ground + N objects. ``J_bones`` (SMPL skeleton, in
``FramePose``) is distinct from ``J_demo`` (the dataset's joints) — never conflate them. A sequence is
``list[FrameTargets]``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# =============================================================================
# interaction/ — field evaluation results
# =============================================================================
@dataclass(frozen=True)
class ContactField:
    """One cloud vs ONE channel, ONE frame. Inactive probes: distance=+margin, rest 0."""

    distance: np.ndarray   # (P,)    signed distance
    direction: np.ndarray  # (P, 3)  contact normal (surface -> point)
    witness: np.ndarray    # (P, 3)  nearest surface point
    active: np.ndarray     # (P,)    bool, within margin


@dataclass(frozen=True)
class MultiChannelField:
    """One cloud vs ALL channels, ONE frame. Channel-first, homogeneous (C = ground + N obj)."""

    distance: np.ndarray         # (C, P)
    direction: np.ndarray        # (C, P, 3)
    witness: np.ndarray          # (C, P, 3)
    active: np.ndarray           # (C, P) bool
    channels: tuple[str, ...]    # (C,) channel names

    def __post_init__(self) -> None:
        c = len(self.channels)
        for name in ("distance", "direction", "witness", "active"):
            got = getattr(self, name).shape[0]
            if got != c:
                raise ValueError(f"{name} has {got} channels, expected {c}")

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def n_points(self) -> int:
        return self.distance.shape[1]


# =============================================================================
# per-frame targets -> solve
# =============================================================================
@dataclass(frozen=True)
class StyleTargets:
    """Style objective, one frame: robot posture/style tracking, G1-ready via joint mapping.
    The object-agnostic "how the body should move" channel. Provisional shape — the style
    objective is still being designed (see ``targets/style/``)."""

    link_names: tuple[str, ...]            # (L,)
    position: np.ndarray                   # (L, 3) world target per link
    weight: np.ndarray                     # (L,) tracking weight
    orientation: np.ndarray | None = None  # (L, 4) wxyz, or None if position-only


@dataclass(frozen=True)
class RobotInteractionTargets:
    """Human field transported onto the robot's M correspondence points, ONE frame.
    The static binding (which link each point attaches to) lives in
    ``InteractionContext.correspondence`` — NOT duplicated here per frame."""

    field: MultiChannelField               # on the M robot points


@dataclass(frozen=True)
class EnvironmentInteractionTargets:
    """Object clouds vs the channels (object-ground / object-object), ONE frame; NOT transported.

    Consumer status: scene-side contact, currently for viz / diagnostics; a potential ``solve``
    constraint later (object consistency). Cheap (same eval matrix as the human side)."""

    per_object: tuple[MultiChannelField, ...]  # one per object cloud


@dataclass(frozen=True)
class FrameTargets:
    """Output of ``process_frame`` for ONE frame; the targets -> solve contract.
    A sequence is ``list[FrameTargets]``. Solve also receives the static
    ``InteractionContext`` (for the correspondence binding)."""

    style: StyleTargets
    robot_interaction: RobotInteractionTargets
    env_interaction: EnvironmentInteractionTargets


# =============================================================================
# shared per-frame state + viz trace
# =============================================================================
@dataclass(frozen=True)
class FramePose:
    """Per-frame world transforms, computed ONCE and shared by both treatments: ``style``
    uses the demo joints (from GroundedScene); ``interaction`` uses these bone + object
    transforms to pose its clouds. ``J_bones`` = SMPL skeleton (distinct from J_demo)."""

    bone_rot: np.ndarray    # (J_bones, 3, 3) SMPL bone world rotations
    bone_pos: np.ndarray    # (J_bones, 3)    SMPL bone world origins
    object_rot: np.ndarray  # (N, 3, 3) object world rotations
    object_pos: np.ndarray  # (N, 3)    object world positions


@dataclass(frozen=True)
class FrameTrace:
    """ALL artifacts of one frame, for inspection / visualisation. Produced by
    ``targets.pipeline.trace_frame`` — the SAME pure ops as ``process_frame``, intermediates
    kept. The clean seam for ``viz/``: zero hooks in the compute."""

    pose: FramePose
    human_cloud_world: np.ndarray                  # (P, 3) posed SMPL cloud
    object_clouds_world: tuple[np.ndarray, ...]    # per object, (P_i, 3)
    human_field: MultiChannelField                 # on the human cloud (PRE-transport)
    targets: FrameTargets                          # final outputs (style + robot + env)
