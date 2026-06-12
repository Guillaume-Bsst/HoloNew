"""Run the selected retargeting methods, then open the per-method stage viewer.

For every requested method (holosoma native SOCP, GMR-SOCP v1, GMR-SOCP v2) this
builds a ``MethodViz`` carrying the solved robot trajectory plus the named
skeleton stages of that method's preprocessing pipeline, then binds them to the
viewer's Method/Stage dropdowns.

Use ``--methods`` to pick a subset, e.g. ``--methods gmr_socp_v1`` to solve only
that optimizer instead of all three.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro

from HoloNew.config_types.data_type import MotionDataConfig
from HoloNew.config_types.robot import RobotConfig
from HoloNew.examples.robot_retarget import (
    DEFAULT_DATA_FORMATS,
    RetargetingConfig,
    create_task_constants,
    load_motion_data,
    run_headless,
)
from HoloNew.src.correspondence.constants import SMPLX_MODEL_DIR_DEFAULT
from HoloNew.src.correspondence.human_body import HumanBody
from HoloNew.src.correspondence.human_metadata import load_human_metadata
from HoloNew.src.gmr_socp_v1.gmr_socp_v1 import GmrSocpRetargeterV1
from HoloNew.src.gmr_socp_v2.gmr_socp_v2 import GmrSocpRetargeterV2
from HoloNew.src.holosoma.preprocess import compute_holosoma_stages
from HoloNew.src.utils import load_intermimic_quats
from HoloNew.src.viewer import MethodViz, Viewer

logger = logging.getLogger(__name__)

Method = Literal["holosoma", "gmr_socp_v1", "gmr_socp_v2"]


@dataclass
class ViewStagesConfig(RetargetingConfig):
    # Which optimizers to solve and show, in the given order. Defaults to all three.
    methods: tuple[Method, ...] = ("holosoma", "gmr_socp_v1", "gmr_socp_v2")
    # Original OMOMO dataset root (the one holding
    # data/{train,test}_diffusion_manip_seq_joints24.p, NOT OMOMO_new). Supplies
    # the subject's SMPL-X betas + gender for the mesh; omit for the neutral shape.
    omomo_dir: Path | None = None


def view(cfg: ViewStagesConfig) -> None:
    if not cfg.methods:
        raise ValueError("--methods must select at least one optimizer")

    data_format = cfg.data_format or DEFAULT_DATA_FORMATS[cfg.task_type]

    # Keep the nested configs consistent with the top-level selections, the same
    # way robot_retarget.main() does before building constants / loading data.
    if cfg.robot_config.robot_type != cfg.robot:
        cfg.robot_config = RobotConfig(robot_type=cfg.robot)
    if cfg.motion_data_config.robot_type != cfg.robot or cfg.motion_data_config.data_format != data_format:
        cfg.motion_data_config = MotionDataConfig(data_format=data_format, robot_type=cfg.robot)

    constants = create_task_constants(
        robot_config=cfg.robot_config,
        motion_data_config=cfg.motion_data_config,
        task_config=cfg.task_config,
        task_type=cfg.task_type,
    )
    # load_motion_data returns RAW joints (preprocessing is a later, separate step).
    raw_joints, _object_poses, smpl_scale = load_motion_data(
        cfg.task_type, data_format, cfg.data_path, cfg.task_name, constants, cfg.motion_data_config
    )

    # Per-joint quaternions only exist for the smplh .pt format; the SMPL-X mesh
    # needs them. Everything else degrades gracefully to skeleton-only.
    original_quats = None
    if data_format == "smplh":
        try:
            original_quats = load_intermimic_quats(str(cfg.data_path / f"{cfg.task_name}.pt"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("No per-joint quats (%s); SMPL-X mesh disabled.", exc)

    # SMPL-X shape (betas) + gender for this subject, read from the original OMOMO
    # .p files keyed by sequence name. Without them the mesh uses the neutral mean
    # shape. The .pt motion (OMOMO_new) does not carry betas, hence the separate dir.
    betas, gender = None, "neutral"
    if cfg.omomo_dir is not None:
        betas, gender = load_human_metadata(cfg.omomo_dir, cfg.task_name)
        if betas is not None:
            logger.info("Loaded SMPL-X shape for %s: gender=%s, %d betas",
                        cfg.task_name, gender, betas.shape[0])
        else:
            logger.warning("No SMPL-X shape found for %s in %s; using neutral shape.",
                           cfg.task_name, cfg.omomo_dir)

    human_body = None
    if original_quats is not None:
        try:
            human_body = HumanBody(SMPLX_MODEL_DIR_DEFAULT, betas, gender)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SMPL-X unavailable (%s); mesh disabled.", exc)

    toe = [constants.DEMO_JOINTS.index(name) for name in cfg.motion_data_config.toe_names]
    # JOINTS_MAPPING is a dict {human_joint_name -> robot_link_name}; the mapped
    # indices select the human-side joints out of the 52-joint DEMO_JOINTS array.
    mapped = [constants.DEMO_JOINTS.index(name) for name in constants.JOINTS_MAPPING]

    def build_holosoma() -> MethodViz:
        # holosoma: native solved qpos + reproduced preprocessing stages.
        native = run_headless(cfg=cfg)
        hs = compute_holosoma_stages(raw_joints, smpl_scale, toe, mapped)
        return MethodViz("holosoma", "holosoma", native.qpos, hs)

    def build_gmr(label: str, key: str, cls) -> MethodViz:
        # GMR v1 / v2: solved qpos + full per-stage mapped-body point clouds.
        rt = cls.from_config(cfg)
        res = rt.retarget()
        T = res.qpos.shape[0]
        stages = {name.capitalize(): rt.gmr_stages[name]["pos"]
                  for name in ("mapped", "scaled", "offset", "ground")}
        # The 52-joint raw source skeleton, trimmed to the solved horizon.
        stages = {"Original": raw_joints[:T, :, :], **stages}
        return MethodViz(label, key, res.qpos, stages)

    builders = {
        "holosoma": build_holosoma,
        "gmr_socp_v1": lambda: build_gmr("GMR-SOCP v1", "gmr_socp_v1", GmrSocpRetargeterV1),
        "gmr_socp_v2": lambda: build_gmr("GMR-SOCP v2", "gmr_socp_v2", GmrSocpRetargeterV2),
    }
    methods = [builders[name]() for name in cfg.methods]

    T = min(m.qpos.shape[0] for m in methods)
    for m in methods:
        m.stages["Original"] = raw_joints[:T, :, :]

    keys = tuple(m.robot_key for m in methods)
    # TODO: pass the object URDF + has_dynamic_object for object_interaction /
    # climbing tasks so the object appears in the viewer (robot_only has none).
    viewer = Viewer(
        robot_model_path=cfg.robot_config.ROBOT_URDF_FILE,
        object_model_path=None,
        stage_keys=keys,
        original_joints=raw_joints[:T, :, :],
        original_quats=None if original_quats is None else original_quats[:T],
        object_poses=None,
        human_body=human_body,
    )
    viewer.bind_methods(methods)
    input("Stage viewer at http://localhost:8080 — Enter to exit ...")


if __name__ == "__main__":
    view(tyro.cli(ViewStagesConfig))
