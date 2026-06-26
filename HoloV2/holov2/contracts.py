"""Shared data contracts for the HoloV2 retargeting pipeline.

These types cross the module boundaries between ``prepare`` + ``targets`` (everything that does
NOT depend on the robot configuration the solver optimises) and ``solve`` (which does).

Every artifact that crosses a module boundary is defined here as a FROZEN dataclass of numpy
arrays (Structure-of-Arrays, channel-first where speed matters) or as an interface PROTOCOL.
DATA + PROTOCOLS only — no logic, no I/O, no heavy deps. Modules depend on these types, never
on each other's code, which keeps the dependency graph acyclic.

Two distinct inputs:
- ``SceneSpec`` = WHAT to run (data identity: dataset, sequence, robot, model dirs).
- ``Config``    = HOW (algorithm knobs). Cache keys mix the relevant ``Config`` subset with
  the ``SceneSpec`` identity.

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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


# =============================================================================
# Protocols (interfaces — concrete impls live in their modules)
# =============================================================================
@runtime_checkable
class BodyModel(Protocol):
    """Parametric human body (SMPL family). Concrete impl in ``prepare/load/smpl.py``.
    Poses the body from per-frame params; ``bone_transforms`` gives the per-bone world
    transforms used to pose the human cloud (mesh-free, via the sparse skinning)."""

    faces: np.ndarray  # (F, 3) int — topology, frame-invariant
    n_bones: int       # J_bones (52 SMPL-H / 55 SMPL-X)

    def posed_vertices(self, params: "SmplParams", t: int) -> np.ndarray:
        """(V, 3) world mesh vertices at frame ``t`` (offline use: sampling, viz)."""

    def bone_transforms(self, params: "SmplParams", t: int) -> tuple[np.ndarray, np.ndarray]:
        """(J_bones,3,3) world rotations and (J_bones,3) world origins at frame ``t`` (FK)."""

    def rest_vertices(self, params: "SmplParams") -> np.ndarray:
        """(V, 3) rest-pose vertices for the subject (sampling the cloud once)."""


@runtime_checkable
class RobotModel(Protocol):
    """Robot kinematics. Rest transforms (q-independent) are used by ``prepare`` to sample
    the G1 surface / build the correspondence; full FK (q-dependent) is used by ``solve``.
    Concrete impl in a kinematics module."""

    link_names: tuple[str, ...]
    dof: int

    def link_transforms(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(L,3,3) rotations, (L,3) positions: world transform of each link for ``qpos``."""

    def rest_transforms(self) -> tuple[np.ndarray, np.ndarray]:
        """Link transforms at the rest configuration."""


@runtime_checkable
class AssetBuilder(Protocol):
    """Common interface of the offline deliverable builders (``prepare/``): calibration,
    sdf, point_cloud. Each hashes ONLY its relevant ``Config`` subset (+ inputs + upstream
    keys) so a param change invalidates only the affected items."""

    def cache_key(self, config: "Config", *inputs: Any) -> str:
        """Stable key from the relevant config subset + inputs (geometry/subject hash)."""

    def build(self, config: "Config", *inputs: Any) -> Any:
        """The heavy offline computation -> the asset."""

    def load(self, path: Path) -> Any: ...
    def save(self, asset: Any, path: Path) -> None: ...


# =============================================================================
# Entry: what to run (data identity) — distinct from Config
# =============================================================================
@dataclass(frozen=True)
class RobotSpec:
    """Identity of the target robot (drives loading, FK, cache keys)."""

    name: str                      # "g1", "h1", "t1"
    urdf_path: Path
    link_names: tuple[str, ...]
    dof: int
    height: float                  # used by calibration scale


@dataclass(frozen=True)
class SceneSpec:
    """WHAT to run. The loader turns this into ``RawMotion``; cache keys mix this identity
    with the relevant ``Config`` subset."""

    dataset: str                              # loader key (omomo/hodome/sfu/lafan/...)
    motion_path: Path                         # the sequence
    robot: RobotSpec
    smpl_model_dir: Path | None = None        # parametric body model dir (None ok => style-only)
    object_mesh_paths: tuple[Path, ...] = ()  # optional override; else resolved by the loader
    ground_mesh_path: Path | None = None      # None => flat ground (no SDF); else terrain -> SDF
    cache_dir: Path | None = None             # default: HoloV2/cache/


# =============================================================================
# Configuration — drives BOTH prepare and targets; cache keys derive from it
# =============================================================================
# The default VALUES below are illustrative (from the previous HoloNew implementation);
# finalised when each builder lands.
@dataclass(frozen=True)
class CalibrationConfig:
    mat_height: float = 0.1          # tolerated mat height when grounding feet
    scale_strategy: str = "betas"    # "betas" (FK stature) | "height" | "fixed"


@dataclass(frozen=True)
class SdfConfig:
    spacing: float = 0.01            # isotropic voxel size (m)
    margin: float = 0.05             # band beyond the surface that is stored
    # (the default flat ground has no SDF — handled analytically in the eval; no size here)


@dataclass(frozen=True)
class CloudConfig:
    human_density: float = 2000.0    # points / m^2 on the SMPL surface
    object_density: float = 2000.0   # points / m^2 on each object
    seed: int = 0                    # deterministic sampling — KEY component (shared by the
                                     # human cloud AND the correspondence; they MUST agree)
    k_influences: int = 4            # K of the sparse LBS-on-cloud skinning


@dataclass(frozen=True)
class CorrespondenceConfig:
    rest_pose: str = "tpose"         # G1 rest config for the OT alignment
    ot_reg: float = 0.05             # OT entropic regularisation


@dataclass(frozen=True)
class Config:
    """Single configuration of the whole q-independent pipeline (prepare + targets).
    ``cloud`` feeds both the human cloud and the correspondence (the dependency chain)."""

    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    sdf: SdfConfig = field(default_factory=SdfConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    correspondence: CorrespondenceConfig = field(default_factory=CorrespondenceConfig)
    margin: float = 0.05             # field-eval activation margin (used by targets)


# =============================================================================
# load/ — raw motion & parametric body params
# =============================================================================
@dataclass(frozen=True)
class SmplParams:
    """Per-frame parameters of a parametric body (SMPL-H / SMPL-X). Includes the HANDS
    (needed for grasp); SMPL-X face params are optional."""

    betas: np.ndarray            # (B,)        subject shape (time-invariant)
    global_orient: np.ndarray    # (T, 3)      root orientation, axis-angle
    body_pose: np.ndarray        # (T, 21*3)   body joint rotations, axis-angle
    left_hand_pose: np.ndarray   # (T, 15*3)
    right_hand_pose: np.ndarray  # (T, 15*3)
    transl: np.ndarray           # (T, 3)      root translation
    gender: str                  # "neutral" | "male" | "female"
    model_type: str              # "smplh" | "smplx"
    jaw_pose: np.ndarray | None = None     # (T, 3)   SMPL-X only
    leye_pose: np.ndarray | None = None    # (T, 3)
    reye_pose: np.ndarray | None = None    # (T, 3)
    expression: np.ndarray | None = None   # (T, E)

    @property
    def n_frames(self) -> int:
        return self.transl.shape[0]


@dataclass(frozen=True)
class RawMotion:
    """Output of a ``prepare/load/`` dataset loader — uniform across formats, BEFORE
    calibration. ``smpl_params is None`` => positions-only source (lafan/mocap): no body mesh
    to sample, so only the STYLE treatment runs (no interaction)."""

    joint_pos: np.ndarray                 # (T, J_demo, 3) world joint positions (always present)
    joint_names: tuple[str, ...]          # (J_demo,)
    fps: float
    source_format: str
    object_poses_raw: tuple[np.ndarray, ...]  # one (T, 7) per object
    object_mesh_paths: tuple[Path, ...]       # one per object, aligned with poses
    smpl_params: SmplParams | None = None

    @property
    def is_parametric(self) -> bool:
        return self.smpl_params is not None

    @property
    def n_frames(self) -> int:
        return self.joint_pos.shape[0]


# =============================================================================
# load/ + prepare/calibration — scene & calibration
# =============================================================================
@dataclass(frozen=True)
class ObjectMesh:
    """A rigid object: geometry in its local frame + per-frame world pose. Built on demand
    by ``prepare/load/mesh.py`` (offline only — never reaches the runtime/solve)."""

    vertices: np.ndarray  # (V, 3) object-local frame
    faces: np.ndarray     # (F, 3) int
    poses: np.ndarray     # (T, 7) world pose per frame [x,y,z,qw,qx,qy,qz]
    name: str
    static: bool = False  # constant pose over T -> eval can skip the per-frame transform


@dataclass(frozen=True)
class Calibration:
    """Per-subject / per-take params placing the WHOLE scene (human + objects) on the ground.
    Offline asset, but NOT a geometry cache: scoped to (subject, take)."""

    scale: float            # human -> robot size
    floor_offset: float     # z shift grounding the scene
    root_frame: np.ndarray  # (4, 4) world transform framing the root


@dataclass(frozen=True)
class GroundedScene:
    """Output of ``prepare`` (loaded motion with calibration applied). The single input of both
    treatments (style, interaction).

    LIGHT by design: no live ``BodyModel``, no trimesh — only grounded motion, params and
    mesh PATHS. Heavy geometry is built ON DEMAND (``prepare/load/smpl.py`` -> ``BodyModel``,
    ``prepare/load/mesh.py`` -> ``ObjectMesh``) inside ``prepare/``, so geometry never reaches
    the style treatment or the solve."""

    joint_pos: np.ndarray                  # (T, J_demo, 3) grounded demo joints — style
    joint_names: tuple[str, ...]           # (J_demo,)
    object_poses: tuple[np.ndarray, ...]   # grounded world pose (T, 7) per object
    object_mesh_paths: tuple[Path, ...]    # geometry pulled on demand by prepare
    calibration: Calibration
    fps: float
    smpl_params: SmplParams | None = None  # grounded params -> build BodyModel on demand

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


@dataclass(frozen=True)
class Channel:
    """One evaluation channel = a signed-distance source + its per-frame pose binding. Makes the
    ground/object alignment EXPLICIT (no implicit N vs N+1 offset). The three cases:

    - ``sdf is None`` & ``object_idx is None`` => the default FLAT ground (analytic z-distance;
      the eval samples it without a grid).
    - ``sdf`` set & ``object_idx is None``     => a TERRAIN ground (stairs/slope/climbing) as an SDF.
    - ``sdf`` set & ``object_idx`` set         => object ``object_idx``, posed by
      ``object_poses[object_idx][f]``.

    The flat ground is the only channel WITHOUT an SDF — handled analytically in the eval."""

    name: str
    object_idx: int | None        # None = static ground (world) ; else index into object_poses/clouds
    sdf: "SDF | None" = None       # None = flat analytic ground ; else the object/terrain SDF grid


# =============================================================================
# prepare/ — geometry assets (build-once, cached)
# =============================================================================
@dataclass(frozen=True)
class SDF:
    """Signed-distance grid of a rigid surface, in its local frame — for objects AND terrain
    ground. The flat default ground has NO SDF (``Channel.sdf is None``, handled analytically).

    Sampled by trilinear interpolation in the eval (``targets/interaction/eval.py``); pure data
    here (no method) so ``contracts`` stays logic-free."""

    grid: np.ndarray     # (Nx, Ny, Nz) signed distance (negative = inside)
    origin: np.ndarray   # (3,) local coords of node (0, 0, 0)
    spacing: float       # isotropic voxel size (m)
    name: str            # channel name, e.g. "obj0" / "ground"


@dataclass(frozen=True)
class PointCloud:
    """Surface samples carrying their own SPARSE SKINNING, posed from part transforms alone
    (mesh-free, torch-free), uniformly for every part kind:
      - object: K=1, weight 1, part = the rigid body.
      - robot : K=1, weight 1, part = the link (posed by FK).
      - human : K~4, LBS-on-cloud blend over the dominant SMPL bones (closes joint creases).

    Posing one frame, given each part's world transform ``T[j] = (R_j, t_j)``:
        p_world[i] = sum_k weights[i,k] * (R[parts[i,k]] @ offsets[i,k] + t[parts[i,k]])
    ``offsets`` are in each part's REST-local frame (skinning baked once offline)."""

    parts: np.ndarray     # (P, K) int    part/bone index per influence
    weights: np.ndarray   # (P, K) float  rows sum to 1 (K=1 => rigid)
    offsets: np.ndarray   # (P, K, 3)     point in part k's rest-local frame
    sampling_id: str = "" # identity of the sampling (density/seed/topology) — binds to the
                          # correspondence built against it (see CorrespondenceTable)

    @property
    def n_points(self) -> int:
        return self.parts.shape[0]

    @property
    def n_influences(self) -> int:
        return self.parts.shape[1]


@dataclass(frozen=True)
class CorrespondenceTable:
    """Fixed SMPL <-> robot surface correspondence (built once by optimal transport, OT).

    Pairs M points: human side (``smpl_idx`` into the SMPL cloud) and robot side
    (``link_idx`` + ``offset_local`` in that link's frame). Transport copies the human field
    at ``smpl_idx[m]`` onto robot point m. VALID ONLY for the SMPL cloud whose
    ``sampling_id == smpl_sampling_id`` (assert at assembly)."""

    smpl_idx: np.ndarray         # (M,) index into the SMPL PointCloud's point order
    link_idx: np.ndarray         # (M,) robot link index (into link_names)
    offset_local: np.ndarray     # (M, 3) robot point in that link's frame
    link_names: tuple[str, ...]  # (L,)
    smpl_sampling_id: str = ""   # the human-cloud sampling this was built against

    @property
    def n_points(self) -> int:
        return self.smpl_idx.shape[0]


@dataclass(frozen=True)
class InteractionContext:
    """All build-once assets for the interaction treatment, passed explicitly (no globals).

    Invariants (checked at assembly):
    - ``channels[0]`` is the GROUND (static; flat by default = no SDF, or a terrain SDF);
      the rest are object channels with ``object_idx`` aligned to ``object_clouds`` and the
      scene's object order.
    - ``human_cloud.sampling_id == correspondence.smpl_sampling_id``."""

    channels: tuple[Channel, ...]          # ground (static) + one per object
    human_cloud: PointCloud                # on the SMPL surface
    object_clouds: tuple[PointCloud, ...]  # one per object (object_clouds[i] <-> channel object_idx=i)
    correspondence: CorrespondenceTable    # SMPL -> robot (STATIC binding)
    margin: float                          # field activation margin (m)

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.channels)


# =============================================================================
# targets (per frame) -> solve
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
# targets/ shared per-frame state + viz trace
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
