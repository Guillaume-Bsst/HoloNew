"""Per-frame contact channels for the floor + object interaction channels.

For every robot correspondence control point and each channel we compute, per frame:
the robot-side signed distance to the surface and its in-contact flag (from
query_entities on the solved trajectory), the reference (human-side) signed distance +
flag, and the tangential slip during sustained contact. Mirrors the geometry of
TestSocpRetargeter._fill_diagnostics but keeps every per-frame value.

The ~8k correspondence points are then aggregated **per robot link** (the body part is
the unit a user reasons about): signed distance -> min (closest point / deepest
penetration), active -> any, slip -> max (worst drift on the link). Floor is always
present; the object channel is emitted only when the run carries an object SDF.
"""
from __future__ import annotations

import numpy as np


def _slip_row(active_t, active_tm1, direction, rel):
    """Tangential slip per point for one frame: |(I - n nᵀ) rel| where contact is sustained."""
    out = np.zeros(active_t.shape[0])
    sustained = active_t & active_tm1
    for i in np.nonzero(sustained)[0]:
        n = np.asarray(direction[i], dtype=float)
        out[i] = np.linalg.norm(rel[i] - n * float(n @ rel[i]))   # project off the normal
    return out


def contact_channels(rt, res) -> dict[str, np.ndarray]:
    """Per-link floor (+ object) contact channels for the solved trajectory.

    Returns {contacts/<channel>/<signal>/<link>: (T,)} with channel in {floor, object},
    signal in {robot_dist, robot_active, ref_dist, ref_active, slip}, aggregated over
    each link's correspondence points. Empty if the run lacks a correspondence / probe.
    """
    from HoloNew.src.test_socp.interaction import (
        robot_control_points, query_entities, frame_references)

    corr = getattr(rt, "correspondence", None)
    if corr is None or getattr(rt, "smplx_ground_probe", None) is None:
        return {}

    qpos = np.asarray(res.qpos)
    T = qpos.shape[0]
    M = corr.link_idx.shape[0]
    L_obj, L_flr = rt.L_object, rt.L_floor
    obj_raw = getattr(rt, "_obj_poses_raw", None)
    has_obj = getattr(rt, "object_sdf", None) is not None

    rd = {"floor": np.zeros((T, M)), "object": np.zeros((T, M))}     # robot signed dist
    ra = {"floor": np.zeros((T, M)), "object": np.zeros((T, M))}     # robot active
    fd = {"floor": np.zeros((T, M)), "object": np.zeros((T, M))}     # ref signed dist
    fa = {"floor": np.zeros((T, M)), "object": np.zeros((T, M))}     # ref active
    sl = {"floor": np.zeros((T, M)), "object": np.zeros((T, M))}     # slip

    Pt_prev = pr_prev = None
    ract_prev = {"floor": np.zeros(M, bool), "object": np.zeros(M, bool)}
    for t in range(T):
        q_pin = rt.pin.qpos_mj_to_q_pin(qpos[t, :36])
        Pt = robot_control_points(rt, q_pin)
        op = obj_raw[t] if obj_raw is not None else None
        fobj, fflr = query_entities(rt, Pt, op, margin_obj=L_obj, margin_flr=L_flr)
        d_obj_ref, _, d_flr_ref, _, pr = frame_references(rt, t)

        fields = {"floor": fflr, "object": fobj}
        ref_d = {"floor": np.asarray(d_flr_ref), "object": np.asarray(d_obj_ref)}
        ref_L = {"floor": L_flr, "object": L_obj}
        for ch, f in fields.items():
            rd[ch][t] = f.distance
            ra[ch][t] = np.asarray(f.active, dtype=float)
            fd[ch][t] = ref_d[ch]
            fa[ch][t] = (ref_d[ch] < ref_L[ch]).astype(float)

        if Pt_prev is not None:
            rel = (Pt - Pt_prev) - (pr - pr_prev)        # robot drift minus source drift
            for ch, f in fields.items():
                ract = np.asarray(f.active, dtype=bool)
                sl[ch][t] = _slip_row(ract, ract_prev[ch], f.direction, rel)
        for ch, f in fields.items():
            ract_prev[ch] = np.asarray(f.active, dtype=bool)
        Pt_prev, pr_prev = Pt, pr

    link_ids = np.asarray(corr.link_idx)
    present = [li for li in range(len(corr.link_names)) if np.any(link_ids == li)]
    out: dict[str, np.ndarray] = {}

    def _emit(ch: str):
        for li in present:
            idx = np.nonzero(link_ids == li)[0]
            nm = corr.link_names[li]
            out[f"contacts/{ch}/robot_dist/{nm}"] = rd[ch][:, idx].min(axis=1)
            out[f"contacts/{ch}/robot_active/{nm}"] = ra[ch][:, idx].max(axis=1)
            out[f"contacts/{ch}/ref_dist/{nm}"] = fd[ch][:, idx].min(axis=1)
            out[f"contacts/{ch}/ref_active/{nm}"] = fa[ch][:, idx].max(axis=1)
            out[f"contacts/{ch}/slip/{nm}"] = sl[ch][:, idx].max(axis=1)

    _emit("floor")
    if has_obj:
        _emit("object")
    return out
