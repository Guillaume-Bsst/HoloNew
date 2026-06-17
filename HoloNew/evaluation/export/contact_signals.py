"""Per-frame contact channels for the floor + object interaction channels.

For every robot correspondence control point and each channel we compute, per frame:
the robot-side signed distance to the surface, the reference (human-side) signed
distance, and the tangential slip during sustained contact. Two in-contact flags are
derived per side:

  * ``*_active``      — the solver's interaction-length band (rt.L_floor / L_object).
                        Reflects what the solve actually saw; depends on a *config*
                        parameter, so it is NOT comparable across runs with different L.
  * ``*_active_phys`` — a fixed physical threshold (ctx.contact_threshold, default 2 cm),
                        independent of L. Use this to compare runs/methods.

Points are aggregated **per robot link**: signed distance -> min (closest point /
deepest penetration), active -> any, slip -> max (worst drift on the link). Floor is
always present; the object channel only when the run carries an object SDF.
"""
from __future__ import annotations

import numpy as np

CONTACT_THRESHOLD_DEFAULT = 0.02  # m, physical (eval_retargeting parity)


def _slip_row(active_t, active_tm1, direction, rel):
    """Tangential slip per point for one frame: |(I - n nᵀ) rel| where contact is sustained."""
    out = np.zeros(active_t.shape[0])
    for i in np.nonzero(active_t & active_tm1)[0]:
        n = np.asarray(direction[i], dtype=float)
        out[i] = np.linalg.norm(rel[i] - n * float(n @ rel[i]))   # project off the normal
    return out


def contact_arrays(rt, res, threshold: float = CONTACT_THRESHOLD_DEFAULT):
    """Per-point (T, M) contact arrays for each present channel + the correspondence.

    Returns (arrays, corr) where arrays is {channel: {robot_dist, robot_active,
    robot_active_phys, ref_dist, ref_active, ref_active_phys, slip}} (each (T, M)),
    channel in {floor, object (only with an object SDF)}. ({}, None) if no correspondence.
    """
    from HoloNew.src.test_socp.interaction import (
        robot_control_points, query_entities, frame_references)

    corr = getattr(rt, "correspondence", None)
    if corr is None or getattr(rt, "smplx_ground_probe", None) is None:
        return {}, None

    qpos = np.asarray(res.qpos)
    T = qpos.shape[0]
    M = corr.link_idx.shape[0]
    L = {"floor": rt.L_floor, "object": rt.L_object}
    obj_raw = getattr(rt, "_obj_poses_raw", None)
    channels = ["floor"] + (["object"] if getattr(rt, "object_sdf", None) is not None else [])

    A = {ch: {k: np.zeros((T, M)) for k in
              ("robot_dist", "robot_active", "robot_active_phys",
               "ref_dist", "ref_active", "ref_active_phys", "slip")}
         for ch in channels}

    Pt_prev = pr_prev = None
    ract_prev = {ch: np.zeros(M, bool) for ch in channels}
    for t in range(T):
        q_pin = rt.pin.qpos_mj_to_q_pin(qpos[t, :36])
        Pt = robot_control_points(rt, q_pin)
        op = obj_raw[t] if obj_raw is not None else None
        fobj, fflr = query_entities(rt, Pt, op, margin_obj=rt.L_object, margin_flr=rt.L_floor)
        d_obj_ref, _, d_flr_ref, _, pr = frame_references(rt, t)
        fields = {"floor": fflr, "object": fobj}
        ref_d = {"floor": np.asarray(d_flr_ref), "object": np.asarray(d_obj_ref)}

        rel = ((Pt - Pt_prev) - (pr - pr_prev)) if Pt_prev is not None else None
        for ch in channels:
            f = fields[ch]
            A[ch]["robot_dist"][t] = f.distance
            A[ch]["robot_active"][t] = np.asarray(f.active, dtype=float)
            A[ch]["robot_active_phys"][t] = (np.asarray(f.distance) <= threshold).astype(float)
            A[ch]["ref_dist"][t] = ref_d[ch]
            A[ch]["ref_active"][t] = (ref_d[ch] < L[ch]).astype(float)
            A[ch]["ref_active_phys"][t] = (ref_d[ch] <= threshold).astype(float)
            ract = np.asarray(f.active, dtype=bool)
            if rel is not None:
                A[ch]["slip"][t] = _slip_row(ract, ract_prev[ch], f.direction, rel)
            ract_prev[ch] = ract
        Pt_prev, pr_prev = Pt, pr

    return A, corr


def contact_channels_from_arrays(arrays, corr) -> dict[str, np.ndarray]:
    """Aggregate per-point contact arrays to per-link channels."""
    if not arrays or corr is None:
        return {}
    link_ids = np.asarray(corr.link_idx)
    present = [li for li in range(len(corr.link_names)) if np.any(link_ids == li)]
    # (signal, reducer) — distances/min, actives/any, slip/max.
    reducers = {"robot_dist": "min", "ref_dist": "min", "slip": "max",
                "robot_active": "max", "robot_active_phys": "max",
                "ref_active": "max", "ref_active_phys": "max"}
    out: dict[str, np.ndarray] = {}
    for ch, sigs in arrays.items():
        for li in present:
            idx = np.nonzero(link_ids == li)[0]
            nm = corr.link_names[li]
            for sig, how in reducers.items():
                col = sigs[sig][:, idx]
                out[f"contacts/{ch}/{sig}/{nm}"] = col.min(axis=1) if how == "min" else col.max(axis=1)
    return out


def contact_channels(rt, res, threshold: float = CONTACT_THRESHOLD_DEFAULT) -> dict[str, np.ndarray]:
    """Per-link floor (+ object) contact channels for the solved trajectory."""
    arrays, corr = contact_arrays(rt, res, threshold)
    return contact_channels_from_arrays(arrays, corr)


def contact_scoreboard(arrays) -> dict[str, dict[str, float]]:
    """Canonical per-channel contact scalars (precision/recall/F1/placement/slip).

    Uses the *physical* in-contact flags (threshold-based, L-independent) so the numbers
    are comparable across runs. Placement = |robot signed distance| at sustained contact.
    """
    from HoloNew.evaluation.metrics.contacts import compute_contacts

    out: dict[str, dict[str, float]] = {}
    for ch, s in (arrays or {}).items():
        out[ch] = compute_contacts(
            robot_contact=s["robot_active_phys"].astype(bool),
            ref_contact=s["ref_active_phys"].astype(bool),
            placement_dist=np.abs(s["robot_dist"]),
            slip=s["slip"])
    return out
