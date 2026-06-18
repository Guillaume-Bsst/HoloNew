"""Configuration types for retargeting (top-level config)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from HoloNew.config_types.data_type import MotionDataConfig
from HoloNew.config_types.retargeter import RetargeterConfig
from HoloNew.config_types.robot import RobotConfig
from HoloNew.config_types.task import TaskConfig


@dataclass
class RetargetingConfig:
    """Top-level retargeting configuration used by the Tyro CLI.

    This combines all configuration types needed for retargeting.
    """

    # --- Task type selection ---
    task_type: Literal["robot_only", "object_interaction", "climbing"] = "object_interaction"
    """Type of retargeting task."""

    # --- top-level run knobs ---
    robot: str = "g1"
    """Robot type. Use str to allow dynamic robot types via _ROBOT_DEFAULTS."""

    data_format: str | None = None
    """Motion data format. Auto-determined by task_type if None.
    Can be any format registered in DEMO_JOINTS_REGISTRY
    (e.g., 'lafan', 'smplh', 'mocap', 'smplx', or custom formats)."""

    task_name: str = "sub3_largebox_003"
    """Name of the task/sequence."""

    data_path: Path = Path("demo_data/OMOMO_new")
    """Path to data directory."""

    # --- New 3-path façade (optional; when `dataset` is set these drive loading) ---
    dataset: str | None = None
    """Dataset key for the 3-path loader façade (omomo, hoim3, lafan, sfu, climbing).
    When set, model_path/motion_path/obj_path are used instead of data_path/task_name."""

    model_path: Path | None = None
    """Façade slot 1 — meaning is dataset-specific (see data_loaders)."""

    motion_path: Path | None = None
    """Façade slot 2 — the motion source file."""

    obj_path: Path | None = None
    """Façade slot 3 — object source (required for object_interaction)."""

    motion_name: str | None = None
    """Sequence name (e.g. sub3_largebox_003, subject01_baseball). When set with
    --dataset, model/motion/obj paths are resolved automatically from the global
    dataset roots (env vars), so the explicit paths are not needed."""

    smpl_model_dir: Path | None = None
    """Explicit body-model directory (no default). Required by datasets that need
    forward kinematics from a separate model (omomo, for its betas-based height)."""

    save_dir: Path | None = None
    """Directory to save results. Auto-determined if None."""

    augmentation: bool = False
    """Whether to use augmentation."""

    # --- Nested configs ---
    robot_config: RobotConfig = field(default_factory=lambda: RobotConfig(robot_type="g1"))
    """Robot configuration (nested - can override robot_urdf_file, robot_dof, etc.
    via --robot-config.robot-urdf-file)."""

    motion_data_config: MotionDataConfig = field(
        default_factory=lambda: MotionDataConfig(data_format="smplh", robot_type="g1")
    )
    """Motion data configuration (nested - can override demo_joints, joints_mapping, etc.
    via --motion-data-config.demo-joints).
    Note: data_format default will be set based on task_type in main()."""

    task_config: TaskConfig = field(default_factory=TaskConfig)
    """Task-specific configuration (nested - can override ground_size, surface_weight_threshold, etc.
    via --task-config.ground-size)."""

    retargeter: RetargeterConfig = field(default_factory=RetargeterConfig)
    """Retargeter configuration (nested - can override q_a_init_idx, activate_joint_limits, etc.
    via --retargeter.q-a-init-idx)."""


@dataclass
class ParallelRetargetingConfig(RetargetingConfig):
    """Extended retargeting config for parallel processing.

    Adds parallel-specific fields while inheriting all retargeting config fields.
    This config is used for processing multiple files in parallel.
    """

    # Parallel processing specific fields
    data_dir: Path = Path("demo_data/OMOMO_new")
    """Directory containing input data files for parallel processing.
    This overrides data_path from RetargetingConfig when processing multiple files."""

    max_workers: int | None = None
    """Maximum number of parallel workers. Auto-determined if None."""
