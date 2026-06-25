"""Dump the per-frame SOCP targets (and the frame-0 warmstart) for GMR-SOCP.

Reproduces the exact config path of

    python examples/view_stages.py --dataset OMOMO \
        --motion-name sub3_largebox_003 --methods gmr_socp

then, instead of opening the viewer, extracts what is actually handed to the
GMR-SOCP solve and writes it to a pickle.

What gets dumped (see retarget()/solve_single_iteration in
src/gmr_socp/gmr_socp.py and ground_frame_targets in src/gmr_socp/targets.py):

  * Per frame t and per tracked robot body, the SOCP objective tracks a target
    POSITION p_target(3,) and a target ORIENTATION (fed as the rotation matrix
    R_target(3,3), built from the wxyz quaternion q_target). These come from the
    GMR 'floor' stage: gpos[t] / gquat[t]. They are IDENTICAL across the two
    passes (table1 / table2); only the cost weights differ.
  * q0_warmstart = rt.q_init_full, the full robot qpos used to initialise the
    iterate() loop on the very first frame (its base is set from the frame-0
    pelvis target; all joint DoFs start at 0).

Usage:
    python examples/dump_socp_targets.py
    python examples/dump_socp_targets.py --motion-name sub3_largebox_003 \
        --out /tmp/socp_targets.pkl
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from HoloNew.examples.view_stages import ViewStagesConfig
from HoloNew.src.data_loaders.facade import normalize_dataset_cfg
from HoloNew.src.gmr_socp.gmr_socp import GmrSocpRetargeter
from HoloNew.src.gmr_socp.tables import (
    IK_MATCH_TABLE1,
    IK_MATCH_TABLE2,
    MAPPED_BODY_NAMES,
)
from HoloNew.src.gmr_socp.targets import ground_frame_targets


@dataclass
class DumpConfig:
    # Sequence to extract, resolved through the same --dataset/--motion-name façade.
    dataset: str = "OMOMO"
    motion_name: str = "sub3_largebox_003"
    # Cap frames (None = whole clip), mirrors view_stages --max-frames.
    max_frames: int | None = None
    # Root XY placement scale (preprocess scale stage). None = CLI default
    # (smpl_scale ~0.68, holosoma-style root pull-in); 1.0 = raw root XY (mink-GMR).
    scale_xy_robot: float | None = None
    # Output pickle path.
    out: Path = Path("socp_targets_sub3_largebox_003.pkl")


def main(cfg: DumpConfig) -> None:
    # --- Rebuild the retargeter exactly as view_stages' build_gmr() does ---------
    vcfg = ViewStagesConfig(
        dataset=cfg.dataset,
        motion_name=cfg.motion_name,
        methods=("gmr_socp",),
        max_frames=cfg.max_frames,
    )
    # Honour the root-XY scale knob (default None reproduces the literal CLI run).
    from HoloNew.src.gmr_socp.config import GmrSocpRetargeterConfig
    vcfg.retargeter = GmrSocpRetargeterConfig(scale_xy_robot=cfg.scale_xy_robot)
    normalize_dataset_cfg(vcfg)        # --dataset/--motion-name -> legacy fields + data_format
    vcfg.dataset = None                # view_stages clears it before from_config; harmless here

    rt = GmrSocpRetargeter.from_config(vcfg)

    # --- The targets handed to the SOCP, per frame --------------------------------
    gpos = rt.gmr_floor["pos"]        # (T, B, 3)  GMR 'floor' positions, MAPPED_BODY_NAMES order
    gquat = rt.gmr_floor["quat"]      # (T, B, 4)  wxyz
    T = gpos.shape[0]
    if cfg.max_frames is not None:
        T = min(T, int(cfg.max_frames))

    # Robot frames in the order the SOCP iterates them (table key order). p/R are
    # shared across passes; only the weights differ between table1 and table2.
    robot_frames = list(IK_MATCH_TABLE1.keys())
    human_bodies = [IK_MATCH_TABLE1[f][0] for f in robot_frames]
    B = len(robot_frames)

    p_target = np.zeros((T, B, 3), dtype=np.float64)
    q_target = np.zeros((T, B, 4), dtype=np.float64)   # wxyz orientation target
    R_target = np.zeros((T, B, 3, 3), dtype=np.float64)

    for t in range(T):
        tg1 = ground_frame_targets(gpos[t], gquat[t], IK_MATCH_TABLE1)
        for j, frame in enumerate(robot_frames):
            p_t, R_t, _w_p, _w_r = tg1[frame]
            p_target[t, j] = p_t
            R_target[t, j] = R_t
            # wxyz quaternion that R_target was built from (the 'floor' body quat).
            q_target[t, j] = gquat[t, MAPPED_BODY_NAMES.index(human_bodies[j])]

    # The GMR-SOCP solve result: the qpos trajectory the two-pass solve produces
    # from these targets, starting from q0_warmstart (rt.retarget()'s output).
    res = rt.retarget(max_frames=cfg.max_frames)
    q_result = np.asarray(res.qpos, dtype=np.float64)   # (T, nq)

    # Cost weights per pass (constant across frames; read straight from the tables).
    weights_pass1 = {f: (float(IK_MATCH_TABLE1[f][1]), float(IK_MATCH_TABLE1[f][2]))
                     for f in robot_frames}
    weights_pass2 = {f: (float(IK_MATCH_TABLE2[f][1]), float(IK_MATCH_TABLE2[f][2]))
                     for f in robot_frames}

    out = {
        "dataset": cfg.dataset,
        "motion_name": cfg.motion_name,
        "method": "gmr_socp",
        # Root XY placement: None -> smpl_scale (~0.68, CLI default), 1.0 -> raw XY.
        "scale_xy_robot": cfg.scale_xy_robot,
        "T": T,
        "robot_frames": robot_frames,        # SOCP target keys, in iteration order
        "human_bodies": human_bodies,        # GMR 'floor' body each frame tracks
        # --- targets sent to the SOCP (per frame, per body) ---
        "p_target": p_target,                # (T, B, 3) position target
        "q_target": q_target,                # (T, B, 4) wxyz orientation target
        "R_target": R_target,                # (T, B, 3, 3) rotation-matrix orientation target
        "weights_pass1": weights_pass1,      # {frame: (pos_w, rot_w)} IK_MATCH_TABLE1
        "weights_pass2": weights_pass2,      # {frame: (pos_w, rot_w)} IK_MATCH_TABLE2
        # --- frame-0 warmstart ---
        "q0_warmstart": np.asarray(rt.q_init_full, dtype=np.float64),  # (nq,)
        "q0_layout": "qpos: base_xyz(3), base_quat_wxyz(4), joint_dofs(nq-7)",
        # --- GMR-SOCP solve output: the solved qpos trajectory ---
        "q_result": q_result,                # (T, nq) solved qpos per frame
        "q_result_layout": "qpos: base_xyz(3), base_quat_wxyz(4), joint_dofs(nq-7)",
        # raw 'floor' stage source (MAPPED_BODY_NAMES order), for cross-checking
        "floor_pos": np.asarray(gpos[:T], dtype=np.float64),   # (T, B, 3)
        "floor_quat": np.asarray(gquat[:T], dtype=np.float64),  # (T, B, 4) wxyz
    }

    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.out, "wb") as f:
        pickle.dump(out, f)

    print(f"Wrote {cfg.out}")
    print(f"  T={T} frames, B={B} tracked bodies")
    print(f"  robot_frames={robot_frames}")
    print(f"  p_target {p_target.shape}, q_target {q_target.shape}, R_target {R_target.shape}")
    print(f"  q0_warmstart {out['q0_warmstart'].shape}")
    print(f"  q_result {q_result.shape}  (solved GMR-SOCP qpos trajectory)")


if __name__ == "__main__":
    main(tyro.cli(DumpConfig))
