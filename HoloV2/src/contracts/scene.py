"""Scene geometry + calibration: rigid objects, per-subject grounding, the grounded scene."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
    """Per-(subject, take) grounding + subject characterisation. ROBOT-FREE, so it caches per
    subject independently of the target robot. The human->robot scale is deliberately NOT here: it
    is a (human, robot) quantity owned and applied by the correspondence + transport layer (where
    both bodies meet), composed from ``human_stature`` and the robot height.

    Single-human, multi-object: ONE ``human_stature`` + the human and the objects each grounded by
    their OWN z-shift (the human may float while the objects already rest on the floor, so one shared
    scene shift would push them through it). ``human_offset`` grounds the human (its feet);
    ``object_offset`` is a SINGLE shift shared by ALL objects (grounds the lowest-reaching object just
    above the floor, keeping inter-object geometry). Offline asset, NOT a geometry cache: (subject, take).

    TODO: a finer per-object / inter-object calibration could ground each object and jointly optimise
    the object<->object & object<->floor contacts (then ``object_offset`` -> per-object offsets)."""

    human_stature: float                 # subject rest stature (m), betas-FK — feeds scale = robot_h / stature
    human_offset: float                  # z-shift grounding the human (feet -> floor)
    object_offset: float                 # z-shift shared by ALL objects (lowest-reaching object -> ~floor)
    root_frame: np.ndarray               # (4, 4) world transform framing the root


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
