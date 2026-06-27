"""Raw motion + parametric body params (``prepare/load/`` outputs, BEFORE calibration)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
    ``smpl_params is None`` there is no body mesh to sample, so only the STYLE treatment would run
    (no interaction) — see ``is_parametric``."""

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
