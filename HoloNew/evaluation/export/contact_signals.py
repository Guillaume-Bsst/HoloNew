"""Per-frame contact channels for the floor + object interaction channels.

For every robot correspondence control point and each channel we emit, per frame: the
robot-side signed distance to the surface and its in-contact flag (from query_entities
on the solved trajectory), the reference (human-side) signed distance + flag, and the
tangential slip during sustained contact. Mirrors the geometry of
TestSocpRetargeter._fill_diagnostics but keeps every per-point series instead of
reducing to a single foot-slip scalar. Floor is always present; the object channel is
emitted only when the run carries an object SDF.
"""
from __future__ import annotations

import numpy as np


def _point_names(corr) -> list[str]:
    """Per-control-point labels: '<link>_<global index>' (unique, groups by link)."""
    return [f"{corr.link_names[corr.link_idx[i]]}_{i:03d}"
            for i in range(corr.link_idx.shape[0])]


def _slip_row(active_t, active_tm1, direction, rel):
    """Tangential slip per point for one frame: |(I - n nᵀ) rel| where contact is sustained."""
    out = np.zeros(active_t.shape[0])
    sustained = active_t & active_tm1
    for i in np.nonzero(sustained)[0]:
        n = np.asarray(direction[i], dtype=float)
        out[i] = np.linalg.norm(rel[i] - n * float(n @ rel[i]))   # project off the normal
    return out


def contact_channels(rt, res) -> dict[str, np.ndarray]:
    """Per-point floor (+ object) contact channels for the solved trajectory.

    Returns {contacts/<channel>/<signal>/<point>: (T,)} with channel in {floor, object},
    signal in {robot_dist, robot_active, ref_dist, ref_active, slip}. Empty if the run
    lacks a correspondence / source probe.
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

    names = _point_names(corr)
    out: dict[str, np.ndarray] = {}

    def _emit(ch: str):
        for i in range(M):
            p = names[i]
            out[f"contacts/{ch}/robot_dist/{p}"] = rd[ch][:, i]
            out[f"contacts/{ch}/robot_active/{p}"] = ra[ch][:, i]
            out[f"contacts/{ch}/ref_dist/{p}"] = fd[ch][:, i]
            out[f"contacts/{ch}/ref_active/{p}"] = fa[ch][:, i]
            out[f"contacts/{ch}/slip/{p}"] = sl[ch][:, i]

    _emit("floor")
    if has_obj:
        _emit("object")
    return out
