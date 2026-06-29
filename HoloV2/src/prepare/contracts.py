"""Data contracts of the ``prepare`` stage — its PUBLIC type surface.

Every artifact ``prepare`` produces or consumes that crosses a module boundary is a FROZEN
dataclass of numpy arrays (Structure-of-Arrays) or an interface PROTOCOL. DATA + PROTOCOLS only —
no logic, no I/O, no heavy deps (numpy-only), so this module is importable everywhere.

This is the stage's contract: downstream stages import their inputs FROM HERE (e.g. ``targets`` does
``from ..prepare.contracts import GroundedScene, InteractionContext``), never from ``prepare``'s
internal submodules. Knobs (the HOW) live apart in ``prepare/config.py``; this holds only the data
(the WHAT) that flows. The pipeline is linear (prepare -> targets -> solve), so each stage owns its
own contracts and depends only on the public types of the stage upstream — the dependency graph
stays acyclic.

Conventions
-----------
- Per-frame is the canonical unit. The current target is OFFLINE replay (``process_frame`` indexes a
  loaded ``GroundedScene`` at frame ``f``); live teleoperation is a future variant on the same ops.
- Two joint sets, kept distinct: ``J_demo`` (the dataset's joints, used by ``style``) and
  ``J_bones`` (the SMPL skeleton, used to pose clouds). Never conflate them.
- Quaternions are wxyz. Rigid poses are ``(x, y, z, qw, qx, qy, qz)``.
- Arrays are read-only by convention (``frozen`` freezes the binding, not the buffer).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


# =============================================================================
# Protocols (interfaces — concrete impls live in prepare/load/*)
# =============================================================================
@runtime_checkable
class BodyModel(Protocol):
    """Parametric human body (SMPL family). Concrete impl in ``prepare/load/smpl.py``.
    Poses the body from per-frame params; ``bone_transforms`` gives the per-bone world
    transforms used to pose the human cloud (mesh-free, via the sparse skinning)."""

    faces: np.ndarray  # (F, 3) int — topology, frame-invariant
    n_bones: int       # J_bones (52 SMPL-H / 55 SMPL-X)
    stature: float     # subject rest stature (m), betas-FK — a pure rest-mesh property (no motion).
                       # Lives on the body (its natural owner), NOT the calibration; feeds the
                       # human->robot scale = robot_height / stature, composed at the transport seam.

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


class AssetBuilder(Protocol):
    """Common SHAPE of the offline deliverable builders (``prepare/``): calibration, sdf,
    point_cloud. A NOMINAL guide (cache_key / build / load / save), NOT a strict polymorphic
    interface: each concrete builder takes its OWN sub-config (a schema from ``prepare/config.py``)
    plus its own specific inputs, so the real signatures differ. ``config``/inputs are typed
    ``Any`` here for that reason, and ``@runtime_checkable`` is deliberately omitted — an
    ``isinstance`` check over ``Any`` signatures would be a false guarantee. Each builder hashes
    ONLY its relevant config subset (+ inputs + upstream keys), so a param change invalidates only
    the affected items."""

    def cache_key(self, config: Any, *inputs: Any) -> str:
        """Stable key from the relevant config subset + inputs (geometry/subject hash)."""

    def build(self, config: Any, *inputs: Any) -> Any:
        """The heavy offline computation -> the asset."""

    def load(self, path: Path) -> Any: ...
    def save(self, asset: Any, path: Path) -> None: ...


# =============================================================================
# Entry: what to run (data identity) — distinct from the step config
# =============================================================================
@dataclass(frozen=True)
class RobotSpec:
    """Identity of the target robot (drives loading, FK, cache keys)."""

    name: str                      # "g1", "h1", "t1"
    urdf_path: Path
    link_names: tuple[str, ...]
    dof: int
    height: float                  # nominal robot height (m); consumed DOWNSTREAM by the
                                   # correspondence/transport layer as scale = robot_height /
                                   # body.stature — NOT by the (robot-free) calibration


@dataclass(frozen=True)
class SceneSpec:
    """WHAT to run. The loader turns this into ``RawMotion``; cache keys mix this identity
    with the relevant step-config subset (the schemas in ``prepare/config.py``)."""

    dataset: str                              # loader key (omomo/hodome/sfu/hoim3)
    motion_path: Path                         # the sequence
    robot: RobotSpec
    smpl_model_dir: Path | None = None        # parametric body model dir (None ok => style-only)
    object_mesh_paths: tuple[Path, ...] = ()  # optional override; else resolved by the loader
    ground_mesh_path: Path | None = None      # None => flat ground (plane SDF); else terrain mesh -> SDF
    cache_dir: Path | None = None             # default: HoloV2/cache/
    dataset_root: Path | None = None          # release root for auxiliary metadata kept apart from
                                              # motion_path (OMOMO betas/scales + captured meshes)
    person_id: int | None = None             # multi-person datasets: which person to retarget
                                              # (None => the first present); ignored if single-person
    object_names: tuple[str, ...] | None = None  # named-object datasets: subset to load
                                              # (None => all); ignored if objects are unnamed
    smplh_dir: Path | None = None             # HOI-M3 only: SMPL-H model dir (holds <gender>/model.npz);
                                              # None => derive by convention from smpl_model_dir
    smpl2smplx_pkl: Path | None = None        # HOI-M3 only: SMPL->SMPL-X deformation-transfer .pkl;
                                              # None => derive by convention from smpl_model_dir


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
    calibration. Every current loader is PARAMETRIC (fills ``smpl_params``); the ``| None`` is a
    structural provision for a future positions-only source, NOT an active path. When
    ``smpl_params is None`` there is no body to pose, and the bone-based ``targets`` pipeline (style +
    interaction both need the body's FK) does not run — see ``is_parametric``."""

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
# calibration — scene geometry & grounding
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
    """Per-(subject, take) GROUNDING. ROBOT-FREE *and* BODY-FREE: built from the mocap demo joints
    (human floor) and the object meshes/poses (object floor) alone — no betas/body needed — so it
    caches per take independently of the target robot. The subject's ``stature`` lives on the
    ``BodyModel`` (its natural rest-mesh owner), and the human->robot scale (a (human, robot)
    quantity) is composed downstream at the transport seam — neither belongs here.

    Single-human, multi-object: the human and the objects each ground by their OWN z-shift (the human
    may float while the objects already rest on the floor, so one shared scene shift would push them
    through it). ``human_offset`` grounds the human (its feet); ``object_offset`` is a SINGLE shift
    shared by ALL objects (grounds the lowest-reaching object just above the floor, keeping
    inter-object geometry). Offline asset, NOT a geometry cache: (subject, take).

    TODO: a finer per-object / inter-object calibration could ground each object and jointly optimise
    the object<->object & object<->floor contacts (then ``object_offset`` -> per-object offsets)."""

    human_offset: float                  # z-shift grounding the human (feet -> floor)
    object_offset: float                 # z-shift shared by ALL objects (lowest-reaching object -> ~floor)
    root_frame: np.ndarray               # (4, 4) world transform framing the root


@dataclass(frozen=True)
class GroundedScene:
    """Output of ``prepare`` (loaded motion with calibration applied). The single input of both
    treatments (style, interaction). The grounding ``Calibration`` rides inside (provenance/viz), so
    ``prepare`` returns just ``(GroundedScene, InteractionContext)``.

    Carries the subject's ``body`` (the live posing engine): per frame, ``interaction`` poses the
    human cloud via ``body.bone_transforms(smpl_params, f)``. The body is typed by the numpy-only
    ``BodyModel`` PROTOCOL, so ``targets`` calls it while staying torch-free at import (torch is
    hidden inside the instance, built once in ``prepare``). Object meshes stay mesh PATHS, NOT live
    geometry — the asymmetry is principled: the human DEFORMS (needs per-frame FK), objects are RIGID
    (a pose7 + the pre-sampled object cloud suffice). ``style`` also reads the body (it tracks the SMPL
    BONES, not the demo joints); ``solve`` never sees a ``GroundedScene`` (it consumes ``FrameTargets``)
    — so no heavy object reaches it. ``body is None`` <=> a positions-only source (no SMPL params): a
    STRUCTURAL placeholder, not a wired path — the ``targets`` pipeline is bone-based (style + cloud
    posing both need the body's FK) and raises on ``body is None``."""

    joint_pos: np.ndarray                  # (T, J_demo, 3) grounded demo joints — style
    joint_names: tuple[str, ...]           # (J_demo,)
    object_poses: tuple[np.ndarray, ...]   # grounded world pose (T, 7) per object
    object_mesh_paths: tuple[Path, ...]    # geometry pulled on demand by prepare
    calibration: Calibration
    fps: float
    smpl_params: SmplParams | None = None  # grounded params -> consumed by ``body.bone_transforms``
    body: BodyModel | None = None          # the subject's live posing engine (None => positions-only)

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
# sdf / point_cloud — build-once geometry assets (the interaction inputs)
# =============================================================================
@dataclass(frozen=True)
class SDF:
    """Signed-distance grid of a rigid surface, in its local frame — for objects, terrain ground
    AND the flat ground (a plane is an affine field, so trilinear sampling reproduces it EXACTLY on a
    tiny grid; it is an ordinary SDF too, keeping every channel homogeneous — see ``build_plane_sdf``).

    Carries a WITNESS grid (nearest surface point per node) alongside the distance: the eval
    reconstructs the contact direction as ``normalize(probe - witness)`` from the trilinearly
    interpolated witness, which stays a true unit vector near sharp box edges/corners — where a
    finite-difference gradient of the distance grid is unstable. Sampled by trilinear interpolation
    in the eval (``targets/interaction/eval.py``); pure data here (no method)."""

    grid: np.ndarray     # (Nx, Ny, Nz) signed distance (negative = inside)
    witness: np.ndarray  # (Nx, Ny, Nz, 3) nearest surface point per node, local frame
    origin: np.ndarray   # (3,) local coords of node (0, 0, 0)
    spacing: float       # isotropic voxel size (m)
    name: str            # channel name, e.g. "obj0" / "ground"

    def __post_init__(self) -> None:
        if self.witness.shape != self.grid.shape + (3,):
            raise ValueError(
                f"witness shape {self.witness.shape} != grid shape {self.grid.shape} + (3,)")


@dataclass(frozen=True)
class GeodesicTable:
    """All-pairs géodésique (distance de graphe k-NN) sur les points de surface d'un mesh rigide, en
    frame locale. AUTO-CONTENU : porte SES points + normales, donc consommable sans le ``object_cloud``.
    La ligne ``geo[j]`` EST le champ géodésique mono-source depuis le point ``j`` (lookup O(1), ligne
    contiguë) — c'est ce qu'on lit à un ``witness(q)`` continu pour le résidu witness (côté solve).
    Géométrie rigide ⇒ pose-invariant (une translation/rotation préserve les géodésiques)."""

    points: np.ndarray    # (P, 3) f32  échantillons de surface (= sampling object_cloud), frame locale
    normals: np.ndarray   # (P, 3) f32  normale unitaire par point (gating snap/interp thin/concave)
    geo: np.ndarray       # (P, P) f32  geo[i, j] = géodésique de graphe i->j (symétrique)
    name: str             # nom de canal ("obj0"/"terrain") — provenance, aligné SDF/cloud
    sampling_id: str = "" # identité du sampling (densité/seed/topo) — provenance

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    def __post_init__(self) -> None:
        p = self.points.shape[0]
        if self.geo.shape != (p, p):
            raise ValueError(f"geo shape {self.geo.shape} != (P, P) with P={p}")
        if self.normals.shape != self.points.shape:
            raise ValueError(
                f"normals shape {self.normals.shape} != points shape {self.points.shape}")


@dataclass(frozen=True)
class Channel:
    """One evaluation channel = a signed-distance source + its per-frame pose binding. Makes the
    ground/object alignment EXPLICIT (no implicit N vs N+1 offset). EVERY channel carries an ``sdf``
    so the eval has a SINGLE trilinear path (homogeneous, no flat-ground special case); ``object_idx``
    only sets the pose binding:

    - ``object_idx is None`` => the static GROUND in the world frame. Its ``sdf`` is a plane grid by
      default (a plane is affine, so a tiny grid reproduces ``z`` EXACTLY) or a TERRAIN grid
      (stairs/slope/climbing).
    - ``object_idx`` set      => object ``object_idx``, its ``sdf`` posed by ``object_poses[object_idx][f]``."""

    name: str
    object_idx: int | None        # None = static ground (world) ; else index into object_poses/clouds
    sdf: SDF                       # the signed-distance grid (ground plane / terrain / object)
    geodesic: "GeodesicTable | None" = None   # None = sol PLAN (le coût retombe sur l'euclidien
                                              # analytique, qui EST la géodésique exacte d'un plan) ;
                                              # sinon objet/terrain. Seule entorse au "jamais de None"
                                              # du sdf, assumée : le plan est le seul cas à forme close.


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
    - ``channels[0]`` is the GROUND (static; a plane SDF by default, or a terrain SDF);
      the rest are object channels with ``object_idx`` aligned to ``object_clouds`` and the
      scene's object order.
    - ``human_cloud.sampling_id == correspondence.smpl_sampling_id``.
    - ``robot_cloud.n_points == correspondence.n_points`` (same M points)."""

    channels: tuple[Channel, ...]          # ground (static) + one per object
    human_cloud: PointCloud                # on the SMPL surface
    object_clouds: tuple[PointCloud, ...]  # one per object (object_clouds[i] <-> channel object_idx=i)
    correspondence: CorrespondenceTable    # SMPL -> robot (STATIC binding)
    margin: float                          # field activation margin (m)
    robot_cloud: PointCloud                # the M correspondence robot points as a K=1 cloud, parts
                                           # in robot FK link order — solve poses it at q (online re-eval)
    robot: RobotModel                      # q-dependent kinematics engine (FK to pose robot_cloud);
                                           # mirrors GroundedScene.body, heavy deps hidden in the instance

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.channels)
