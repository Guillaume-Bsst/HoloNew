"""Canonical 7-family scoreboard scalars for the run summary.

These are the *real* metric scalars (the same `compute_*` the eval scoreboard uses),
not the generic mean/rms/min/max reductions of the padded export channels. Cheap
families (smoothness/effort/dynamics) recompute directly; tracking/style/roots reduce
the already-collected per-frame channels (unpadded, so the mean IS the scalar); contacts
reduce the per-point arrays with the physical (L-independent) flags.
"""
from __future__ import annotations

import numpy as np


def _mean_over_prefix(channels: dict[str, np.ndarray], prefix: str) -> float | None:
    vals = [np.asarray(v).ravel() for k, v in channels.items() if k.startswith(prefix)]
    return float(np.mean(np.concatenate(vals))) if vals else None


def compute_scoreboard(channels: dict[str, np.ndarray], res, ctx,
                       contact_arrays=None) -> dict[str, dict[str, float]]:
    """Assemble the canonical scoreboard from collected channels + cheap recompute."""
    from HoloNew.evaluation.metrics.smoothness import compute_smoothness
    from HoloNew.evaluation.metrics.effort import compute_effort
    from HoloNew.evaluation.metrics.dynamics import compute_dynamics

    sb: dict[str, dict[str, float]] = {}
    qpos = np.asarray(res.qpos)

    if ctx.dof:
        sb["smoothness"] = compute_smoothness(qpos, int(ctx.dof), ctx.dt)

    if ctx.joint_limit_cols is not None and len(ctx.joint_limit_cols):
        joints = qpos[:, 7:7 + int(ctx.dof)][:, np.asarray(ctx.joint_limit_cols, int)]
        sb["effort"] = compute_effort(joints, ctx.joint_limit_lower, ctx.joint_limit_upper, ctx.dt)

    com, com_ref = getattr(res, "com", None), getattr(res, "com_ref", None)
    if com is not None and com_ref is not None:
        T = np.asarray(com).shape[0]
        sb["dynamics"] = compute_dynamics(
            np.asarray(com), np.asarray(com_ref)[:T], ctx.dt,
            L=getattr(res, "angular_momentum", None),
            L_ref=getattr(res, "angular_momentum_ref", None))

    # tracking / style / roots: reduce the unpadded per-frame channels (mean over a
    # prefix == the canonical scalar, since each channel is length T).
    tracking = {}
    for key, prefix in (("mpjpe_global", "tracking/mpjpe/"),
                        ("mpjpe_root_rel", "tracking/mpjpe_root_rel/")):
        v = _mean_over_prefix(channels, prefix)
        if v is not None:
            tracking[key] = v
    if "tracking/base_track" in channels:
        tracking["base_track_err"] = float(np.mean(channels["tracking/base_track"]))
    if tracking:
        sb["tracking"] = tracking

    orient = _mean_over_prefix(channels, "tracking/orient/")
    if orient is not None:
        sb["style"] = {"orient_err": orient}  # shape not exported per-frame yet

    if "roots/base_pos_err" in channels:
        sb["roots"] = {"pos_err": float(np.mean(channels["roots/base_pos_err"])),
                       "rot_err": float(np.mean(channels["roots/base_rot_err"]))}

    if contact_arrays:
        from HoloNew.evaluation.export.contact_signals import contact_scoreboard
        for ch, m in contact_scoreboard(contact_arrays).items():
            sb[f"contacts_{ch}"] = m

    return sb
