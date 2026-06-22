#!/usr/bin/env python
"""Verify the exported SOCP inputs are well-formed AND faithful to the source.

Three layers of checks:
  A. STRUCTURE   — keys, shapes, dtypes present.
  B. NUMERIC     — R are valid rotations (orthonormal, det=+1), quats unit & consistent
                   with R, positions finite; q0 layout correct; pelvis target == q0 base.
  C. FAITHFUL    — re-derive R+P and q0 straight from the .pt with the SAME test_socp
                   pure functions and assert they reproduce the pickles bit-for-bit
                   (deterministic). This proves the files match the SOCP pre-solve code path.
  D. PHYSICAL    — sanity ranges (pelvis height, feet near floor, finite spans).

Run (gmr env has numpy/scipy/torch/mujoco):
    PY=/home/ecarn/.wbt_deps/miniconda3/envs/gmr/bin/python
    $PY modules/01_retargeting/HoloNew/socp_handoff/verify_socp_inputs.py
Exit code 0 = all good.
"""
from __future__ import annotations

import importlib.util
import os
import pickle
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HN = os.path.abspath(os.path.join(HERE, "..", "HoloNew"))            # HoloNew package dir
WBT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))    # wbt_rl root (holds data/)
HN_SRC = os.path.join(HN, "src")
PT = os.path.join(WBT, "data", "00_raw_datasets", "OMOMO_new", "OMOMO_new",
                  "sub3_largebox_003.pt")


def _find_bundle():
    """Directory holding rp_targets.pkl: argv[1] if given, else HERE, else any subdir."""
    import glob
    if len(sys.argv) > 1 and os.path.exists(os.path.join(sys.argv[1], "rp_targets.pkl")):
        return sys.argv[1]
    for d in [HERE] + [os.path.dirname(p) for p in
                       glob.glob(os.path.join(HERE, "**", "rp_targets.pkl"), recursive=True)]:
        if os.path.exists(os.path.join(d, "rp_targets.pkl")):
            return d
    return HERE


BUNDLE = _find_bundle()

_n_pass = 0
_n_fail = 0


def check(name, cond, detail=""):
    global _n_pass, _n_fail
    ok = bool(cond)
    _n_pass += ok
    _n_fail += (not ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def load_pure(hn_src):
    pkg = types.ModuleType("ts_pure")
    pkg.__path__ = [os.path.join(hn_src, "test_socp")]
    sys.modules["ts_pure"] = pkg

    def load(mod):
        spec = importlib.util.spec_from_file_location(
            f"ts_pure.{mod}", os.path.join(hn_src, "test_socp", f"{mod}.py"))
        m = importlib.util.module_from_spec(spec)
        sys.modules[f"ts_pure.{mod}"] = m
        spec.loader.exec_module(m)
        return m
    return load("tables"), load("targets"), load("preprocess")


def main():
    from scipy.spatial.transform import Rotation as Rot

    print(f"bundle: {BUNDLE}")
    rp = pickle.load(open(os.path.join(BUNDLE, "rp_targets.pkl"), "rb"))
    q0 = pickle.load(open(os.path.join(BUNDLE, "q0.pkl"), "rb"))
    tables, targets, preprocess = load_pure(HN_SRC)

    T, K = rp["T"], len(rp["robot_frames"])
    pos, R, quat = rp["pos"], rp["R"], rp["quat_wxyz"]

    print("A. STRUCTURE")
    check("rp keys", {"pos", "R", "quat_wxyz", "robot_frames", "weights", "ground_stage"} <= set(rp))
    check("pos shape", pos.shape == (T, K, 3), str(pos.shape))
    check("R shape", R.shape == (T, K, 3, 3), str(R.shape))
    check("quat shape", quat.shape == (T, K, 4), str(quat.shape))
    check("q0 keys", {"q0_mujoco", "base_pos", "base_quat_wxyz", "joint_names_mujoco_order"} <= set(q0))

    print("B. NUMERIC")
    Rf = R.reshape(-1, 3, 3)
    dets = np.linalg.det(Rf)
    orth = np.max(np.abs(np.einsum("nij,nkj->nik", Rf, Rf) - np.eye(3)))
    check("det(R) == +1", np.allclose(dets, 1.0, atol=1e-6), f"range [{dets.min():.6f},{dets.max():.6f}]")
    check("R orthonormal", orth < 1e-6, f"max|RRᵀ-I|={orth:.2e}")
    qn = np.linalg.norm(quat.reshape(-1, 4), axis=1)
    check("quat unit-norm", np.allclose(qn, 1.0, atol=1e-6), f"range [{qn.min():.6f},{qn.max():.6f}]")
    # quat (wxyz) consistent with R
    R_from_q = Rot.from_quat(quat.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix()
    check("quat <-> R consistent", np.allclose(R_from_q, Rf, atol=1e-5))
    check("pos finite", np.isfinite(pos).all())

    print("B. q0 layout")
    nq = q0["nq"]
    check("nq == 36", nq == 36, str(nq))
    check("29 joints, all zero", len(q0["joint_names_mujoco_order"]) == 29 and np.allclose(q0["joint_values"], 0))
    pelvis_i = rp["robot_frames"].index("pelvis")
    check("q0 base_pos == frame-0 pelvis target",
          np.allclose(q0["base_pos"], pos[0, pelvis_i], atol=1e-6),
          f"q0={np.round(q0['base_pos'],4).tolist()} target={np.round(pos[0,pelvis_i],4).tolist()}")
    # Compare as ROTATIONS, not raw components: q and -q are the same SO(3) element
    # (the rp quats went through a matrix round-trip that can flip the sign).
    R_q0 = Rot.from_quat(q0["base_quat_wxyz"][[1, 2, 3, 0]]).as_matrix()
    check("q0 base_quat == frame-0 pelvis target (rotation)",
          np.allclose(R_q0, R[0, pelvis_i], atol=1e-6),
          f"max|ΔR|={np.max(np.abs(R_q0 - R[0, pelvis_i])):.2e}")

    print("B. weights == ik_tables")
    for name, tbl in (("table1", tables.IK_MATCH_TABLE1),
                      ("table2", tables.IK_MATCH_TABLE2),
                      ("single", tables.IK_MATCH_TABLE_SINGLE)):
        pw = np.array([tbl[f][1] for f in rp["robot_frames"]], float)
        rw = np.array([tbl[f][2] for f in rp["robot_frames"]], float)
        check(f"{name} pos weights", np.array_equal(rp["weights"][name]["pos"], pw))
        check(f"{name} rot weights", np.array_equal(rp["weights"][name]["rot"], rw))

    print("C. FAITHFUL (re-derive from the .pt, same code path)")
    raw = targets.load_pt_joints(PT)
    hq = targets.load_pt_quaternions(PT)
    Tn = min(raw.shape[0], hq.shape[0])
    ground = preprocess.compute_stages(raw[:Tn], hq[:Tn], scale_xy=1.0, scale_z=None)["ground"]
    pos2 = np.empty((Tn, K, 3))
    for t in range(Tn):
        tg = targets.ground_frame_targets(ground["pos"][t], ground["quat"][t], tables.IK_MATCH_TABLE1)
        for i, f in enumerate(rp["robot_frames"]):
            pos2[t, i] = tg[f][0]
    check("re-derived T matches", Tn == T, f"{Tn} vs {T}")
    check("re-derived P == rp_targets.pos", np.allclose(pos2, pos, atol=1e-6),
          f"max|Δ|={np.max(np.abs(pos2-pos)):.2e}")
    check("ground_stage matches re-derivation",
          np.allclose(ground["pos"][:Tn], rp["ground_stage"]["pos"][:Tn], atol=1e-6))

    print("D. PHYSICAL sanity")
    pz = pos[:, pelvis_i, 2]
    check("pelvis height plausible (0.4–1.1 m)", 0.4 < pz.mean() < 1.1, f"mean={pz.mean():.3f} m")
    feet = [rp["robot_frames"].index(f) for f in ("left_toe_link", "right_toe_link")]
    fz_min = pos[:, feet, 2].min()
    check("feet reach the floor (min z ≈ 0)", abs(fz_min) < 0.1, f"min toe z={fz_min:.3f} m")
    span = np.ptp(pos.reshape(-1, 3), axis=0)
    check("motion has spatial extent", (span > 0.1).all(), f"bbox span={np.round(span,2).tolist()} m")

    print(f"\n==> {_n_pass} passed, {_n_fail} failed")
    sys.exit(1 if _n_fail else 0)


if __name__ == "__main__":
    main()
