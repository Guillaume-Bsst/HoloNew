#!/usr/bin/env python
"""Run the real GMR-SOCP retargeter and dump the solved qpos trajectory as a pickle.

This is the SOLVER OUTPUT (q results), not the pre-solve inputs: it runs HoloNew's
`GmrSocpRetargeter` (cvxpy + mujoco, two-pass GMR solve) over the clip and saves the
per-frame robot configuration.

Env: hsretargeting (cvxpy + clarabel + mujoco + torch). The script adds the HoloNew
repo root to sys.path itself, so no PYTHONPATH is needed. Run from anywhere, e.g.:

    HSPY=/home/ecarn/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python
    $HSPY modules/01_retargeting/HoloNew/socp_handoff/run_gmr_socp.py
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    hn_root = os.path.abspath(os.path.join(here, ".."))         # HoloNew git repo root
    pkg = os.path.join(hn_root, "HoloNew")                      # importable HoloNew package dir
    wbt = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))  # wbt_rl root (holds data/)
    if hn_root not in sys.path:
        sys.path.insert(0, hn_root)                             # so `import HoloNew...` resolves

    ap = argparse.ArgumentParser()
    ap.add_argument("--task-name", default="sub3_largebox_003")
    ap.add_argument("--data-path", default=os.path.join(
        wbt, "data", "00_raw_datasets", "OMOMO_new", "OMOMO_new"))
    ap.add_argument("--robot", default="g1")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(here, "gmr_socp_qpos.pkl"))
    args = ap.parse_args()
    # Make paths absolute BEFORE we chdir into the HoloNew package dir below.
    args.out = os.path.abspath(args.out)
    args.data_path = os.path.abspath(args.data_path)

    from HoloNew.config_types.retargeting import RetargetingConfig
    from HoloNew.src.gmr_socp.gmr_socp import GmrSocpRetargeter
    from HoloNew.src.gmr_socp.gmr_socp import IK_MATCH_TABLE1  # noqa: F401 (sanity import)

    # HoloNew resolves robot/model asset paths (e.g. models/g1/g1_29dof.xml) relative to
    # the package dir, so run from there. data_path / out are absolute and unaffected.
    os.chdir(pkg)

    cfg = RetargetingConfig(
        task_type="robot_only",
        task_name=args.task_name,
        data_format="smplh",
        robot=args.robot,
        data_path=Path(args.data_path),
    )

    rt = GmrSocpRetargeter.from_config(cfg)
    res = rt.retarget(max_frames=args.max_frames)
    qpos = np.asarray(res.qpos)            # (T, nq) MuJoCo order

    # Robot joint layout (MuJoCo qpos order) for self-describing output.
    import mujoco as mj
    m = rt.robot_model
    joint_names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j)
                   for j in range(m.njnt)
                   if int(m.jnt_type[j]) != int(mj.mjtJoint.mjJNT_FREE)]

    out = {
        "description": "Solved qpos trajectory from GMR-SOCP (HoloNew GmrSocpRetargeter), "
                       "two-pass cvxpy solve (table1 then table2 per frame).",
        "method": "gmr_socp",
        "task_name": args.task_name,
        "robot": args.robot,
        "convention": "MuJoCo qpos per frame: [base_pos(3), base_quat_wxyz(4), joints(29)].",
        "T": int(qpos.shape[0]),
        "nq": int(qpos.shape[1]),
        "fps": 30,
        "qpos": qpos.astype(np.float64),                 # (T, nq) — the result
        "joint_names_mujoco_order": joint_names,
        "q_init": np.asarray(rt.q_init_full, np.float64), # warm-start (== q0.pkl)
        "source_pt": str(Path(args.data_path) / f"{args.task_name}.pt"),
    }
    with open(args.out, "wb") as fh:
        pickle.dump(out, fh)

    print(f"[gmr_socp] solved T={out['T']} frames, nq={out['nq']}")
    print(f"[gmr_socp] qpos[0,:7] = {np.round(qpos[0, :7], 4).tolist()}")
    print(f"[gmr_socp] qpos[-1,:7] = {np.round(qpos[-1, :7], 4).tolist()}")
    print(f"[gmr_socp] wrote {args.out}  ({qpos.nbytes/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
