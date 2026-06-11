"""Run a retarget, then open the multi-stage viewer."""
from __future__ import annotations

import tyro

from HoloNew.examples.robot_retarget import RetargetingConfig, run_headless
from HoloNew.src.stages import STAGE_SPECS
from HoloNew.src.viewer import Viewer


def view(cfg: RetargetingConfig) -> None:
    result = run_headless(cfg=cfg)
    qpos_keys = tuple(s.key for s in STAGE_SPECS if s.produces_qpos)
    viewer = Viewer(
        robot_model_path=cfg.robot_config.ROBOT_URDF_FILE,
        object_model_path=None,
        stage_keys=qpos_keys,
    )
    viewer.bind(result)
    input("Viewer at http://localhost:8080 — Enter to exit ...")


if __name__ == "__main__":
    view(tyro.cli(RetargetingConfig))
