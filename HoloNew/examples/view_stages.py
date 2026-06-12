"""Run a retarget, then open the multi-stage viewer."""
from __future__ import annotations

import tyro

from HoloNew.examples.robot_retarget import RetargetingConfig, run_headless
from HoloNew.src.gmr_socp_v1.gmr_socp_v1 import GmrSocpRetargeterV1
from HoloNew.src.gmr_socp_v2.gmr_socp_v2 import GmrSocpRetargeterV2
from HoloNew.src.stages import STAGE_SPECS
from HoloNew.src.viewer import Viewer


def view(cfg: RetargetingConfig) -> None:
    result = run_headless(cfg=cfg)                        # native SOCP (RetargetResult)
    gmr1 = GmrSocpRetargeterV1.from_config(cfg).retarget().qpos
    gmr2 = GmrSocpRetargeterV2.from_config(cfg).retarget().qpos
    extra = {"gmr_socp_v1": gmr1, "gmr_socp_v2": gmr2}

    qpos_keys = tuple(s.key for s in STAGE_SPECS if s.produces_qpos)
    # TODO: pass the object URDF + has_dynamic_object for object_interaction /
    # climbing tasks so the object appears in the viewer (robot_only has none).
    viewer = Viewer(
        robot_model_path=cfg.robot_config.ROBOT_URDF_FILE,
        object_model_path=None,
        stage_keys=qpos_keys,
    )
    viewer.bind(result, extra_qpos=extra)
    input("Viewer at http://localhost:8080 — Enter to exit ...")


if __name__ == "__main__":
    view(tyro.cli(RetargetingConfig))
