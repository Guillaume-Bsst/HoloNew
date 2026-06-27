"""Interface PROTOCOLS — the concrete impls live in their own modules.

Body/robot kinematics protocols + the common shape of the offline asset builders. No data, no
logic: these only fix the method signatures the rest of the pipeline depends on.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


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


class AssetBuilder(Protocol):
    """Common SHAPE of the offline deliverable builders (``prepare/``): calibration, sdf,
    point_cloud. A NOMINAL guide (cache_key / build / load / save), NOT a strict polymorphic
    interface: each concrete builder takes its OWN sub-config (a schema from ``config_types``)
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
