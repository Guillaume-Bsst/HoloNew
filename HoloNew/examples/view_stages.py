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

import numpy as np
import tyro

from HoloNew.config_types.data_type import MotionDataConfig
from HoloNew.config_types.robot import RobotConfig
from HoloNew.examples.robot_retarget import (
    DEFAULT_DATA_FORMATS,
    RetargetingConfig,
    convert_object_poses_to_mujoco_order,
    create_task_constants,
    load_motion_data,
    run_headless,
)
from HoloNew.src import skeleton
from HoloNew.src.correspondence.constants import SMPLX_MODEL_DIR_DEFAULT
from HoloNew.src.correspondence.human_body import HumanBody
from HoloNew.src.correspondence.human_metadata import load_human_metadata
from HoloNew.src.gmr_socp_v1.gmr_socp_v1 import _BODY_NAME_REMAP, GmrSocpRetargeterV1
from HoloNew.src.gmr_socp_v1.tables import HUMAN_BODY_TO_IDX, IK_MATCH_TABLE1
from HoloNew.src.gmr_socp_v2.gmr_socp_v2 import GmrSocpRetargeterV2
from HoloNew.src.holosoma.preprocess import compute_holosoma_stages, scale_object_poses_to_center
from HoloNew.src.robot_fk import robot_link_positions
from HoloNew.src.stages import ROBOT_STAGE
from HoloNew.src.holosoma.interaction_mesh import transform_points_local_to_world
from HoloNew.src.utils import load_intermimic_data, load_intermimic_quats, load_object_data
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

    # Re-ground the human like test_pipe: the raw capture floats a few cm above the
    # floor, so drop every joint by the median lowest SMPL-X sole height. This puts
    # the posed mesh and the Original skeleton on the ground instead of floating.
    if human_body is not None and original_quats is not None:
        try:
            offset = human_body.floor_offset(original_quats, raw_joints)
            raw_joints = raw_joints.copy()
            raw_joints[:, :, 2] -= offset
            logger.info("Re-grounded human by %.2f cm (median sole).", offset * 100)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Human re-grounding skipped (%s).", exc)

    # Object motion (from the .pt) pulled toward the world centre by the same
    # holosoma scale factor as the robot, so object and robot stay aligned. The
    # .pt pose is [qw,qx,qy,qz,x,y,z]; the viewer wants MuJoCo order. The object
    # name is the second token of the sequence (e.g. sub3_largebox_003 -> largebox).
    object_model_path, object_poses = None, None
    object_points, object_points_demo = None, None
    if data_format == "smplh":
        parts = cfg.task_name.split("_")
        obj_dir = Path("models") / parts[1] if len(parts) >= 2 else None
        try:
            _, obj_poses = load_intermimic_data(str(cfg.data_path / f"{cfg.task_name}.pt"))
            obj_urdf = obj_dir / f"{parts[1]}.urdf" if obj_dir is not None else None
            if obj_urdf is not None and obj_urdf.exists():
                obj_poses = scale_object_poses_to_center(obj_poses, smpl_scale)
                object_poses = convert_object_poses_to_mujoco_order(obj_poses)
                object_model_path = str(obj_urdf)
                # Object interaction samples, placed per frame exactly like the classic
                # holosoma viewer: local points from load_object_data lifted to world by
                # transform_points_local_to_world. *_demo are the smpl_scale source points
                # the solve tracks; the others are the full-mesh target points.
                local_pts, local_pts_demo = load_object_data(
                    str(obj_dir / f"{parts[1]}.obj"), smpl_scale=smpl_scale, sample_count=100)
                object_points = np.stack(
                    [transform_points_local_to_world(p[3:7], p[:3], local_pts) for p in object_poses])
                object_points_demo = np.stack(
                    [transform_points_local_to_world(p[3:7], p[:3], local_pts_demo) for p in object_poses])
            else:
                logger.warning("No object URDF at %s; object disabled.", obj_urdf)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Object unavailable (%s); object disabled.", exc)

    toe = [constants.DEMO_JOINTS.index(name) for name in cfg.motion_data_config.toe_names]
    # JOINTS_MAPPING is a dict {human_joint_name -> robot_link_name}; the mapped
    # indices select the human-side joints out of the 52-joint DEMO_JOINTS array.
    mapped = [constants.DEMO_JOINTS.index(name) for name in constants.JOINTS_MAPPING]

    # Bone topologies + the solved-robot link set, shared by every method. The
    # mapped GMR stages and the holosoma Mapped stage carry different joint sets,
    # so each gets its own bones; the robot stage reads g1 link world positions
    # (FK from qpos) drawn with the GMR mapped-body topology.
    gmr_bones = skeleton.bones_for_subset(list(HUMAN_BODY_TO_IDX.values()))
    holo_mapped_bones = skeleton.bones_for_subset(mapped)
    robot_bodies = [_BODY_NAME_REMAP.get(f, f) for f in IK_MATCH_TABLE1]
    robot_mjcf = cfg.robot_config.ROBOT_URDF_FILE.replace(".urdf", ".xml")

    def build_holosoma() -> MethodViz:
        # holosoma: native solved qpos + reproduced preprocessing stages.
        native = run_headless(cfg=cfg)
        hs = compute_holosoma_stages(raw_joints, smpl_scale, toe, mapped)
        rs = robot_link_positions(robot_mjcf, robot_bodies, native.qpos)
        sb = {"Mapped": holo_mapped_bones, ROBOT_STAGE: gmr_bones}
        return MethodViz("holosoma", "holosoma", native.qpos, hs,
                         stage_bones=sb, robot_skeleton=rs)

    def build_gmr(label: str, key: str, cls) -> MethodViz:
        # GMR v1 / v2: solved qpos + full per-stage mapped-body point clouds.
        rt = cls.from_config(cfg)
        res = rt.retarget()
        T = res.qpos.shape[0]
        stages = {name.capitalize(): rt.gmr_stages[name]["pos"]
                  for name in ("mapped", "scaled", "offset", "ground")}
        # The 52-joint raw source skeleton, trimmed to the solved horizon.
        stages = {"Original": raw_joints[:T, :, :], **stages}
        rs = robot_link_positions(robot_mjcf, robot_bodies, res.qpos)
        sb = {name: gmr_bones for name in ("Mapped", "Scaled", "Offset", "Ground")}
        sb[ROBOT_STAGE] = gmr_bones
        return MethodViz(label, key, res.qpos, stages,
                         stage_bones=sb, robot_skeleton=rs)

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
    viewer = Viewer(
        robot_model_path=cfg.robot_config.ROBOT_URDF_FILE,
        object_model_path=object_model_path,
        stage_keys=keys,
        original_joints=raw_joints[:T, :, :],
        original_quats=None if original_quats is None else original_quats[:T],
        object_poses=None if object_poses is None else object_poses[:T],
        object_points=None if object_points is None else object_points[:T],
        object_points_demo=None if object_points_demo is None else object_points_demo[:T],
        human_body=human_body,
    )
    viewer.bind_methods(methods)
    input("Stage viewer at http://localhost:8080 — Enter to exit ...")


if __name__ == "__main__":
    view(tyro.cli(ViewStagesConfig))
