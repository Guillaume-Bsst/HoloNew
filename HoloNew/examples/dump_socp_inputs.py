"""Dump ONLY the per-frame targets sent to the TEST-SOCP solve (no solve run).

Ultra-simple sibling of dump_test_socp_targets.py. What is actually handed to the
SOCP objective, per frame t and per tracked robot body, is a target POSE: a position
``p_target`` (3,) and an orientation ``q_target`` (wxyz quaternion; ``R_target`` is just
its matrix form). These come straight from the GMR 'ground' stage and are the INPUTS to
the solve — so this script stops right after building the retargeter and never calls
retarget() (fast: seconds, not minutes).

The targets are essentially robot-DOF-independent: the orientations don't depend on the
robot at all, and the positions only shift if g1_27dof and g1_29dof differ in height.

Usage:
    python examples/dump_socp_inputs.py --robot g1_27dof --scale-xy-robot None  --out a.pkl
    python examples/dump_socp_inputs.py --robot g1_29dof --scale-xy-robot 1.0   --out b.pkl
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from HoloNew.examples.view_stages import ViewStagesConfig, _solve_dataset_key
from HoloNew.src.data_loaders.facade import normalize_dataset_cfg
from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
from HoloNew.src.test_socp.tables import IK_MATCH_TABLE1, MAPPED_BODY_NAMES
from HoloNew.src.test_socp.targets import ground_frame_targets
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter


@dataclass
class DumpConfig:
    dataset: str = "SFU"
    motion_name: str = "0008_ChaCha001_stageii"
    # Robot (dof suffix auto-split): g1_27dof or g1_29dof.
    robot: str = "g1_27dof"
    # Cap frames (None = whole clip).
    max_frames: int | None = 271
    # Root XY placement: None = AUTO smpl_scale ('xy_scaled'); 1.0 = raw ('xy_unscaled').
    scale_xy_robot: float | None = None
    out: Path = Path("socp_inputs.pkl")


def main(cfg: DumpConfig) -> None:
    vcfg = ViewStagesConfig(
        dataset=cfg.dataset,
        motion_name=cfg.motion_name,
        methods=("test_socp",),
        max_frames=cfg.max_frames,
        robot=cfg.robot,
    )
    vcfg.retargeter = TestSocpRetargeterConfig(scale_xy_robot=cfg.scale_xy_robot)
    normalize_dataset_cfg(vcfg)
    vcfg.dataset = _solve_dataset_key(vcfg, vcfg.dataset)

    # Build the retargeter (preprocessing only) — the 'ground' targets are populated here.
    # We deliberately do NOT call rt.retarget(): these are the SOCP INPUTS, not the solve.
    rt = TestSocpRetargeter.from_config(vcfg)

    gpos = rt.gmr_ground["pos"]        # (T, B, 3) MAPPED_BODY_NAMES order
    gquat = rt.gmr_ground["quat"]      # (T, B, 4) wxyz
    T = gpos.shape[0]
    if cfg.max_frames is not None:
        T = min(T, int(cfg.max_frames))

    robot_frames = list(IK_MATCH_TABLE1.keys())          # 14 tracked robot frames
    human_bodies = [IK_MATCH_TABLE1[f][0] for f in robot_frames]
    B = len(robot_frames)

    p_target = np.zeros((T, B, 3), dtype=np.float64)
    q_target = np.zeros((T, B, 4), dtype=np.float64)
    for t in range(T):
        tg1 = ground_frame_targets(gpos[t], gquat[t], IK_MATCH_TABLE1)
        for j, frame in enumerate(robot_frames):
            p_t, _R_t, _w_p, _w_r = tg1[frame]
            p_target[t, j] = p_t
            q_target[t, j] = gquat[t, MAPPED_BODY_NAMES.index(human_bodies[j])]

    out = {
        "dataset": cfg.dataset,
        "motion_name": cfg.motion_name,
        "robot": cfg.robot,
        "scale_xy_robot": cfg.scale_xy_robot,
        "T": T,
        "robot_frames": robot_frames,        # which robot body each (p,q) target drives
        "human_bodies": human_bodies,        # the GMR 'ground' body each one tracks
        "p_target": p_target,                # (T, 14, 3) position targets sent to the SOCP
        "q_target": q_target,                # (T, 14, 4) wxyz orientation targets sent to the SOCP
    }

    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.out, "wb") as f:
        pickle.dump(out, f)

    print(f"Wrote {cfg.out}")
    print(f"  robot={cfg.robot} scale_xy_robot={cfg.scale_xy_robot} T={T} bodies={B}")
    print(f"  p_target {p_target.shape}, q_target {q_target.shape}")
    print(f"  keys ({len(out)}): {list(out)}")


if __name__ == "__main__":
    main(tyro.cli(DumpConfig))
