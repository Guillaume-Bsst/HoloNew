"""Dump the per-frame TEST-SOCP targets, warmstart and solved trajectory to a pickle.

The TEST-SOCP sibling of ``dump_socp_targets.py``. Reproduces the exact config path of

    python examples/view_stages.py --dataset SFU \
        --motion-name 0008_ChaCha001_stageii --methods test_socp --robot g1_27dof \
        --max_frames 271

then, instead of opening the viewer, extracts what is handed to the TEST-SOCP solve
(plus the solver's centroidal diagnostics) and writes it to a pickle.

Unlike the GMR-SOCP dumper, TEST-SOCP reads the AMASS SMPL-X (``data_format="smplx"``)
clips that GMR-SOCP cannot load, and honours ``--robot`` (e.g. g1_27dof). The solve is
built the same way view_stages' build_gmr() builds it, including
``rt.collect_diagnostics = True`` so the result carries the CoM / angular-momentum /
foot-slip channels.

What gets dumped (see TestSocpRetargeter / build_from_config in
src/test_socp/builder.py and ground_frame_targets in src/test_socp/targets.py):

  * Per frame t and per tracked robot body, the SOCP objective tracks a target
    POSITION p_target(3,) and a target ORIENTATION (rotation matrix R_target(3,3),
    built from the wxyz quaternion q_target). These come from the GMR 'ground' stage:
    gpos[t] / gquat[t]. They are IDENTICAL across the two passes (table1 / table2);
    only the cost weights differ.
  * q0_warmstart = rt.q_init_full, the full robot qpos used to initialise the first
    frame (base set from the frame-0 pelvis target; all joint DoFs start at 0).
  * q_result = the solved qpos trajectory, plus the centroidal diagnostics the solve
    emits (CoM, angular momentum, foot slip, per-frame cost) when present.

Usage:
    python examples/dump_test_socp_targets.py
    python examples/dump_test_socp_targets.py --dataset SFU \
        --motion-name 0008_ChaCha001_stageii --robot g1_27dof --max-frames 271 \
        --out /tmp/test_socp_targets.pkl
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from HoloNew.examples.view_stages import ViewStagesConfig, _solve_dataset_key
from HoloNew.src.data_loaders.facade import normalize_dataset_cfg
from HoloNew.src.test_socp.tables import (
    IK_MATCH_TABLE1,
    IK_MATCH_TABLE2,
    MAPPED_BODY_NAMES,
)
from HoloNew.src.test_socp.targets import ground_frame_targets
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter


@dataclass
class DumpConfig:
    # Sequence to extract, resolved through the same --dataset/--motion-name façade.
    dataset: str = "SFU"
    motion_name: str = "0008_ChaCha001_stageii"
    # Robot (may carry a dof suffix, e.g. 'g1_27dof'); build_from_config auto-splits it.
    robot: str = "g1_27dof"
    # Cap frames (None = whole clip), mirrors view_stages --max-frames.
    max_frames: int | None = 271
    # Root XY placement scale (preprocess scale stage). None = AUTO = smpl_scale
    # (~0.68, root pulled toward the world centre) -> the 'xy_scaled' output; 1.0 = raw
    # root XY (TEST-SOCP's native default, GMR targets and probe share one frame) -> the
    # 'xy_unscaled' output. Mirrors dump_socp_targets.py's scale_xy_robot knob.
    scale_xy_robot: float | None = None
    # Output pickle path.
    out: Path = Path("test_socp_targets_0008_ChaCha001_stageii.pkl")


def main(cfg: DumpConfig) -> None:
    # --- Rebuild the retargeter exactly as view_stages' build_gmr() does ---------
    vcfg = ViewStagesConfig(
        dataset=cfg.dataset,
        motion_name=cfg.motion_name,
        methods=("test_socp",),
        max_frames=cfg.max_frames,
        robot=cfg.robot,
    )
    # Honour the root-XY scale knob (None = AUTO smpl_scale 'xy_scaled', 1.0 = raw
    # 'xy_unscaled'). Injecting the typed config mirrors dump_socp_targets.py; with
    # scale_xy_robot=1.0 it reproduces the literal CLI run (TEST-SOCP's native default),
    # since build_from_config's own TestSocpRetargeterConfig() fallback uses the same value.
    from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
    vcfg.retargeter = TestSocpRetargeterConfig(scale_xy_robot=cfg.scale_xy_robot)
    normalize_dataset_cfg(vcfg)        # --dataset/--motion-name -> legacy fields + data_format
    dataset_key = vcfg.dataset         # canonical lower-cased key (e.g. 'sfu')
    # Mirror build_gmr: re-expose the dataset to the solve's object resolver only when there
    # is no legacy .pt (SFU has none); harmless for object-less SFU (robot_only).
    vcfg.dataset = _solve_dataset_key(vcfg, dataset_key)

    rt = TestSocpRetargeter.from_config(vcfg)
    # Enable the CoM / angular-momentum / foot-slip diagnostics, as view_stages does.
    if hasattr(rt, "collect_diagnostics"):
        rt.collect_diagnostics = True

    # --- The targets handed to the SOCP, per frame --------------------------------
    gpos = rt.gmr_ground["pos"]        # (T, B, 3)  GMR 'ground' positions, MAPPED_BODY_NAMES order
    gquat = rt.gmr_ground["quat"]      # (T, B, 4)  wxyz
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
            # wxyz quaternion that R_target was built from (the 'ground' body quat).
            q_target[t, j] = gquat[t, MAPPED_BODY_NAMES.index(human_bodies[j])]

    # The TEST-SOCP solve result: the qpos trajectory produced from these targets,
    # starting from q0_warmstart.
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
        "method": "test_socp",
        # Root XY placement: None -> smpl_scale (~0.68, 'xy_scaled'); 1.0 -> raw ('xy_unscaled').
        "scale_xy_robot": cfg.scale_xy_robot,
        "T": T,
        "robot_frames": robot_frames,        # SOCP target keys, in iteration order
        "human_bodies": human_bodies,        # GMR 'ground' body each frame tracks
        # --- targets sent to the SOCP (per frame, per body) ---
        "p_target": p_target,                # (T, B, 3) position target
        "q_target": q_target,                # (T, B, 4) wxyz orientation target
        "R_target": R_target,                # (T, B, 3, 3) rotation-matrix orientation target
        "weights_pass1": weights_pass1,      # {frame: (pos_w, rot_w)} IK_MATCH_TABLE1
        "weights_pass2": weights_pass2,      # {frame: (pos_w, rot_w)} IK_MATCH_TABLE2
        # --- frame-0 warmstart ---
        "q0_warmstart": np.asarray(rt.q_init_full, dtype=np.float64),  # (nq,)
        "q0_layout": "qpos: base_xyz(3), base_quat_wxyz(4), joint_dofs(nq-7)",
        # --- TEST-SOCP solve output: the solved qpos trajectory ---
        "q_result": q_result,                # (T, nq) solved qpos per frame
        "q_result_layout": "qpos: base_xyz(3), base_quat_wxyz(4), joint_dofs(nq-7)",
        # raw 'ground' stage source (MAPPED_BODY_NAMES order), for cross-checking
        "ground_pos": np.asarray(gpos[:T], dtype=np.float64),   # (T, B, 3)
        "ground_quat": np.asarray(gquat[:T], dtype=np.float64),  # (T, B, 4) wxyz
    }

    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.out, "wb") as f:
        pickle.dump(out, f)

    print(f"Wrote {cfg.out}")
    print(f"  dataset={cfg.dataset} motion={cfg.motion_name} robot={cfg.robot}")
    print(f"  T={T} frames, B={B} tracked bodies")
    print(f"  robot_frames={robot_frames}")
    print(f"  p_target {p_target.shape}, q_target {q_target.shape}, R_target {R_target.shape}")
    print(f"  q0_warmstart {out['q0_warmstart'].shape}")
    print(f"  q_result {q_result.shape}  (solved TEST-SOCP qpos trajectory)")
    print(f"  keys ({len(out)}): {list(out)}")


if __name__ == "__main__":
    main(tyro.cli(DumpConfig))
