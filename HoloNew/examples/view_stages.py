"""Run each retargeting method, then open the per-method stage viewer.

For every method (holosoma native SOCP, GMR-SOCP v1, GMR-SOCP v2) this builds a
``MethodViz`` carrying the solved robot trajectory plus the named skeleton
stages of that method's preprocessing pipeline, then binds them to the viewer's
Method/Stage dropdowns.
"""
from __future__ import annotations

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
from HoloNew.src.gmr_socp_v1.gmr_socp_v1 import GmrSocpRetargeterV1
from HoloNew.src.gmr_socp_v2.gmr_socp_v2 import GmrSocpRetargeterV2
from HoloNew.src.holosoma.preprocess import compute_holosoma_stages
from HoloNew.src.viewer import MethodViz, Viewer


def view(cfg: RetargetingConfig) -> None:
    data_format = cfg.data_format or DEFAULT_DATA_FORMATS[cfg.task_type]

    # Keep the nested configs consistent with the top-level selections, the same
    # way robot_retarget.main() does before building constants / loading data.
    if cfg.robot_config.robot_type != cfg.robot:
        cfg.robot_config = RobotConfig(robot_type=cfg.robot)
    if cfg.motion_data_config.robot_type != cfg.robot or cfg.motion_data_config.data_format != data_format:
        cfg.motion_data_config = MotionDataConfig(data_format=data_format, robot_type=cfg.robot)

    # holosoma: native solved qpos + reproduced preprocessing stages.
    native = run_headless(cfg=cfg)
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
    toe = [constants.DEMO_JOINTS.index(name) for name in cfg.motion_data_config.toe_names]
    # JOINTS_MAPPING is a dict {human_joint_name -> robot_link_name}; the mapped
    # indices select the human-side joints out of the 52-joint DEMO_JOINTS array.
    mapped = [constants.DEMO_JOINTS.index(name) for name in constants.JOINTS_MAPPING]
    hs = compute_holosoma_stages(raw_joints, smpl_scale, toe, mapped)
    holosoma = MethodViz("holosoma", "holosoma", native.qpos, hs)

    # GMR v1 / v2: solved qpos + full per-stage mapped-body point clouds.
    def gmr_method(label: str, key: str, cls) -> MethodViz:
        rt = cls.from_config(cfg)
        res = rt.retarget()
        T = res.qpos.shape[0]
        stages = {name.capitalize(): rt.gmr_stages[name]["pos"]
                  for name in ("mapped", "scaled", "offset", "ground")}
        # The 52-joint raw source skeleton, trimmed to the solved horizon.
        stages = {"Original": raw_joints[:T, :, :], **stages}
        return MethodViz(label, key, res.qpos, stages)

    methods = [
        holosoma,
        gmr_method("GMR-SOCP v1", "gmr_socp_v1", GmrSocpRetargeterV1),
        gmr_method("GMR-SOCP v2", "gmr_socp_v2", GmrSocpRetargeterV2),
    ]

    keys = tuple(m.robot_key for m in methods)
    # TODO: pass the object URDF + has_dynamic_object for object_interaction /
    # climbing tasks so the object appears in the viewer (robot_only has none).
    viewer = Viewer(
        robot_model_path=cfg.robot_config.ROBOT_URDF_FILE,
        object_model_path=None,
        stage_keys=keys,
    )
    viewer.bind_methods(methods)
    input("Stage viewer at http://localhost:8080 — Enter to exit ...")


if __name__ == "__main__":
    view(tyro.cli(RetargetingConfig))
