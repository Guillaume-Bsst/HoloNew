#!/usr/bin/env python
"""Export the three SOCP retargeting inputs for one clip, for Loïc's solver.

Produces, for a given motion clip and the G1 (g1_29dof) robot:

  1. rp_targets.pkl  — the R + P tracking targets for every frame t
                       (the SOCP objective's W^pos / W^rot reference).
  2. q0.pkl          — the initial robot configuration (the optimiser warm-start),
                       and exactly how it is built.
  3. ik_tables.py    — the two IK match tables in native Python (written separately
                       by the handoff, copied verbatim from test_socp/tables.py).

WHY THIS IS FAITHFUL TO THE SOCP
--------------------------------
All three are *pre-solve* quantities of HoloNew's TEST-SOCP retargeter. They are
computed by exactly the code path `builder.build_from_config` runs BEFORE the SQP
solve — no pinocchio / cvxpy / contact / correspondence needed:

    raw_joints = load_pt_joints(pt)            # (T,52,3)  world positions
    human_quat = load_pt_quaternions(pt)       # (T,52,4)  wxyz
    gmr_grounded = ground_to_floor(raw_joints) # uniform z-shift over the WHOLE clip
    ground = compute_stages(gmr_grounded, human_quat,
                            scale_xy=1.0, scale_z=None)["ground"]
    q_init_full = zeros(nq)
    q_init_full[:3]  = ground["pos"][0, pelvis]
    q_init_full[3:7] = ground["quat"][0, pelvis]

`ground_to_floor` is a single constant z-shift applied to every frame, and the
'ground' stage re-grounds by subtracting the per-clip minimum z, so the constant
cancels:  compute_stages(raw)["ground"] == compute_stages(ground_to_floor(raw))["ground"].
This script asserts that invariance at runtime, so skipping ground_to_floor (and
thus the holosoma import) is provably exact.

The per-frame robot-frame targets are then built exactly like
evaluation/reference_context.ReferenceContext.reference_RP:

    ground_frame_targets(ground["pos"][t], ground["quat"][t], TABLE)
        -> {robot_frame: (p(3,), R(3,3), pos_weight, rot_weight)}

Run with any env that has numpy + scipy + torch + mujoco (e.g. the `gmr` env):

    PY=/home/ecarn/.wbt_deps/miniconda3/envs/gmr/bin/python
    $PY modules/01_retargeting/HoloNew/socp_handoff/export_socp_inputs.py
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import sys
import types

import numpy as np


def _load_test_socp_pure(hn_src: str):
    """Import test_socp.{tables,targets,preprocess} WITHOUT the heavy package __init__.

    A lightweight package shim makes the relative imports (`.tables`) resolve while
    pulling in only numpy/scipy/torch — never pinocchio/cvxpy/holosoma.
    """
    pkgname = "ts_pure"
    pkg = types.ModuleType(pkgname)
    pkg.__path__ = [os.path.join(hn_src, "test_socp")]
    sys.modules[pkgname] = pkg

    def load(mod):
        path = os.path.join(hn_src, "test_socp", f"{mod}.py")
        spec = importlib.util.spec_from_file_location(f"{pkgname}.{mod}", path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[f"{pkgname}.{mod}"] = m
        spec.loader.exec_module(m)
        return m

    return load("tables"), load("targets"), load("preprocess")


def _robot_joint_layout(g1_xml: str):
    """(nq, [29 actuated joint names in MuJoCo qpos order]) from the g1_29dof MJCF."""
    import mujoco as mj
    m = mj.MjModel.from_xml_path(g1_xml)
    names = []
    for j in range(m.njnt):
        if int(m.jnt_type[j]) == int(mj.mjtJoint.mjJNT_FREE):
            continue  # the floating base — unnamed, occupies qpos[:7]
        names.append(mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j))
    return int(m.nq), names


def _assert_ground_invariance(preprocess, raw, quat):
    """Empirically confirm the 'ground' stage is invariant to a constant z-shift
    (the only thing ground_to_floor does), so we may skip it."""
    g0 = preprocess.compute_stages(raw, quat, scale_xy=1.0, scale_z=None)["ground"]
    shifted = raw.copy()
    shifted[:, :, 2] += 0.137  # arbitrary constant z-shift over the whole clip
    g1 = preprocess.compute_stages(shifted, quat, scale_xy=1.0, scale_z=None)["ground"]
    assert np.allclose(g0["pos"], g1["pos"], atol=1e-6), "ground pos not z-shift invariant!"
    assert np.allclose(g0["quat"], g1["quat"], atol=1e-6), "ground quat not invariant!"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    hn = os.path.abspath(os.path.join(here, "..", "HoloNew"))   # HoloNew package dir
    wbt = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))  # wbt_rl root (holds data/)

    ap = argparse.ArgumentParser()
    ap.add_argument("--task-name", default="sub3_largebox_003")
    ap.add_argument("--pt", default=os.path.join(
        wbt, "data", "00_raw_datasets", "OMOMO_new", "OMOMO_new", "sub3_largebox_003.pt"))
    ap.add_argument("--hn-src", default=os.path.join(hn, "src"))
    ap.add_argument("--g1-xml", default=os.path.join(hn, "models", "g1", "g1_29dof.xml"))
    ap.add_argument("--out", default=here)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    tables, targets, preprocess = _load_test_socp_pure(args.hn_src)

    IK1, IK2 = tables.IK_MATCH_TABLE1, tables.IK_MATCH_TABLE2
    IKS = tables.IK_MATCH_TABLE_SINGLE
    MAPPED = list(tables.MAPPED_BODY_NAMES)
    pelvis_bi = MAPPED.index(tables.HUMAN_ROOT_NAME)

    # --- 1. load the source motion (OMOMO .pt, smplh layout) ---
    raw = targets.load_pt_joints(args.pt)          # (T,52,3) raw world positions
    quat = targets.load_pt_quaternions(args.pt)    # (T,52,4) wxyz
    T = min(raw.shape[0], quat.shape[0])
    raw, quat = raw[:T], quat[:T]

    # --- 2. GMR pre-IK stages -> the 'ground' targets (scale 1.0 xy, native z) ---
    _assert_ground_invariance(preprocess, raw, quat)
    ground = preprocess.compute_stages(raw, quat, scale_xy=1.0, scale_z=None)["ground"]
    gpos, gquat = ground["pos"], ground["quat"]    # (T,B,3), (T,B,4) wxyz, B=len(MAPPED)

    # --- 3. per-frame robot-frame R + P targets (same pos/R for all tables; weights differ) ---
    robot_frames = list(IK1.keys())
    human_bodies = [IK1[f][0] for f in robot_frames]
    K = len(robot_frames)

    pos = np.empty((T, K, 3), np.float64)
    Rmat = np.empty((T, K, 3, 3), np.float64)
    quat_wxyz = np.empty((T, K, 4), np.float64)
    for t in range(T):
        tg = targets.ground_frame_targets(gpos[t], gquat[t], IK1)
        for i, f in enumerate(robot_frames):
            p_t, R_t, _, _ = tg[f]
            pos[t, i] = p_t
            Rmat[t, i] = R_t
    # quaternion form of the same rotations (wxyz)
    from scipy.spatial.transform import Rotation as Rot
    q_xyzw = Rot.from_matrix(Rmat.reshape(-1, 3, 3)).as_quat().reshape(T, K, 4)
    quat_wxyz[:] = q_xyzw[:, :, [3, 0, 1, 2]]

    def weights(table):
        pw = np.array([float(table[f][1]) for f in robot_frames])
        rw = np.array([float(table[f][2]) for f in robot_frames])
        return {"pos": pw, "rot": rw}

    rp = {
        "description": "Per-frame SE3 body targets (R + P) for the GMR-SOCP tracking "
                       "objective. R/P are identical across the two IK tables (offsets "
                       "are baked into the 'ground' stage); the tables differ ONLY in the "
                       "per-frame cost weights.",
        "source": {"task_name": args.task_name, "pt_path": args.pt,
                   "format": "smplh / OMOMO .pt", "robot": "g1_29dof"},
        "convention": {
            "quat": "wxyz (scalar-first)",
            "R": "3x3 rotation matrix, world frame; R == quat_wxyz",
            "p": "world-frame position (m), after GMR scale(xy=1.0,z=native)+offset+ground",
            "axis_order": "K axis follows robot_frames / human_bodies",
        },
        "T": int(T),
        "fps": float(args.fps),
        "robot_frames": robot_frames,         # the G1 body each target is attached to
        "human_bodies": human_bodies,         # the human body each robot frame tracks
        "pos": pos,                           # (T,K,3) P targets
        "R": Rmat,                            # (T,K,3,3) rotation targets
        "quat_wxyz": quat_wxyz,               # (T,K,4) same rotation, quaternion form
        "weights": {                          # per IK table: position / orientation cost weights
            "table1": weights(IK1),
            "table2": weights(IK2),
            "single": weights(IKS),           # what TEST-SOCP actually solves (one IK pass)
        },
        "ground_stage": {                     # raw GMR 'ground' stage, mapped-body order
            "mapped_body_names": MAPPED,
            "pos": gpos.astype(np.float64),
            "quat_wxyz": gquat.astype(np.float64),
        },
    }
    with open(os.path.join(args.out, "rp_targets.pkl"), "wb") as fh:
        pickle.dump(rp, fh)

    # --- 4. q0: zero joints + base at frame-0 ground pelvis target (MuJoCo qpos order) ---
    nq, joint_names = _robot_joint_layout(args.g1_xml)   # nq=36, 29 actuated joints
    q0 = np.zeros(nq, np.float64)
    q0[0:3] = gpos[0, pelvis_bi]
    q0[3:7] = gquat[0, pelvis_bi]            # wxyz
    q0d = {
        "description": "Initial robot configuration (the SOCP optimiser warm-start). "
                       "It is NOT a cost term — it is where the solve starts.",
        "convention": "MuJoCo qpos order: [base_pos(3), base_quat_wxyz(4), joints(29)], length nq.",
        "construction": "All 29 joints = 0. Floating base = frame-0 'ground' pelvis target: "
                        "q0[:3]=ground.pos[0,pelvis], q0[3:7]=ground.quat[0,pelvis] (wxyz). "
                        "Mirrors builder.build_from_config; pinocchio init = qpos_mj_to_q_pin(q0) "
                        "(only the base quaternion is reordered wxyz->xyzw).",
        "robot": "g1_29dof",
        "nq": int(nq),
        "q0_mujoco": q0,
        "base_pos": q0[0:3].copy(),
        "base_quat_wxyz": q0[3:7].copy(),
        "joint_names_mujoco_order": joint_names,
        "joint_values": q0[7:].copy(),       # all zeros
        "source": {"task_name": args.task_name, "pt_path": args.pt},
    }
    with open(os.path.join(args.out, "q0.pkl"), "wb") as fh:
        pickle.dump(q0d, fh)

    # --- report ---
    print(f"[export] clip={args.task_name}  T={T}  K={K} robot frames  nq={nq}")
    print(f"[export] robot_frames = {robot_frames}")
    print(f"[export] q0 base_pos  = {np.round(q0[0:3], 4).tolist()}")
    print(f"[export] q0 base_quat = {np.round(q0[3:7], 4).tolist()} (wxyz)")
    print(f"[export] wrote rp_targets.pkl  ({pos.nbytes/1e6:.2f} MB pos)  and q0.pkl  -> {args.out}")


if __name__ == "__main__":
    main()
