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
    """One cloud vs ONE channel, ONE frame. Inactive probes: distance=+margin, rest 0.
    ``direction``/``witness`` are in the CHANNEL's frame (see ``MultiChannelField``)."""

    distance: np.ndarray   # (P,)    signed distance
    direction: np.ndarray  # (P, 3)  contact normal (surface -> point)
    witness: np.ndarray    # (P, 3)  nearest surface point
    active: np.ndarray     # (P,)    bool, within margin


@dataclass(frozen=True)
class MultiChannelField:
    """One cloud vs ALL channels, ONE frame. Channel-first, homogeneous (C = ground + N obj).

    Per-channel NATURAL frame (object-as-variable ready): the GROUND channel is in the WORLD frame;
    each OBJECT channel is in THAT object's LOCAL frame — the probe is mapped into the object frame,
    the field is read there, and ``direction``/``witness`` are KEPT there (no world round-trip).
    ``distance`` is a length (frame-invariant). This is exactly the frame the solve's object terms are
    built in (the object's rigid-motion Jacobian couples in object-local), so the object channel needs
    NO rewrite when the object becomes a decision variable."""

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
    objective is still being designed (see ``targets/style/``).

    Per-frame GEOMETRY only: WHERE each tracked link should be (``position``) and how it should be
    oriented (``orientation``). HOW HARD to track each link (the tracking weights / cost gains) is a
    SOLVER concern — static, not per-frame — so it is NOT carried here; ``solve`` defines it in its
    own config when it is built (V1 ``w_p`` / ``w_r``)."""

    link_names: tuple[str, ...]            # (L,)
    position: np.ndarray                   # (L, 3) world target per link
    orientation: np.ndarray | None = None  # (L, 4) wxyz, or None if position-only


@dataclass(frozen=True)
class RobotInteractionTargets:
    """Human field transported onto the robot's M correspondence points, ONE frame.
    The static binding (which link each point attaches to) lives in
    ``InteractionContext.correspondence`` — NOT duplicated here per frame."""

    field: MultiChannelField               # on the M robot points


@dataclass(frozen=True)
class EnvironmentInteractionTargets:
    """Object clouds vs the channels (object<->ground / object<->object), ONE frame; NOT transported.

    First-class solve input for the OBJECT-AS-VARIABLE terms: when the object is a decision variable,
    these scene-side contacts (object vs ground, object vs other objects, in object-local frame) drive
    its consistency. Same eval matrix as the human side (cheap, homogeneous), with ONE extra term the
    human side lacks: the DIAGONAL (object i vs its OWN channel i). The cloud sits on its own surface
    there, so it is filled with the closed-form self-contact (distance 0, witness = the point itself;
    see ``eval_fields`` ``self_idx``), NOT a real sample — the solve ignores that diagonal channel."""

    per_object: tuple[MultiChannelField, ...]  # one per object cloud


@dataclass(frozen=True)
class FrameTargets:
    """Output of ``process_frame`` for ONE frame; the targets -> solve contract. A sequence is
    ``list[FrameTargets]``.

    The solve seam is ``(FrameTargets, InteractionContext)``: solve also reads the static
    ``InteractionContext`` (the correspondence binding for the robot control points, and the channel
    SDFs it re-queries at those points). ``env_interaction`` feeds the object-as-variable terms (the
    object's own contacts), so it is part of the prod path — not viz-only."""

    style: StyleTargets
    robot_interaction: RobotInteractionTargets
    env_interaction: EnvironmentInteractionTargets
    object_rot: np.ndarray                 # (N, 3, 3) per-frame object world rotations — solve's
                                           # object-channel frame + the object-variable init/reference
    object_pos: np.ndarray                 # (N, 3)    per-frame object world positions

    def __post_init__(self) -> None:
        n = len(self.env_interaction.per_object)
        if not (self.object_rot.shape[0] == self.object_pos.shape[0] == n):
            raise ValueError(
                f"object poses ({self.object_rot.shape[0]} rot, {self.object_pos.shape[0]} pos) "
                f"must match env_interaction.per_object count ({n})")


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
