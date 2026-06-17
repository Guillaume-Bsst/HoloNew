"""Signal producers: read per-frame quantities off a RetargetResult (+ context) and
emit flat (T,) channels.

A producer is fn(result, ctx) -> dict[str, np.ndarray] (each value 1-D), returning {}
when its source is absent. PRODUCERS is the ordered registry — the single place to
extend when adding a new signal. ``run_all`` merges every producer's output. Channels
built from finite differences are shorter than T; ``pad_to_T`` right-aligns them onto
the frame grid so the shared time column stays valid.
"""
from __future__ import annotations

import numpy as np

from .context import SignalContext

_XYZ = ("x", "y", "z")


def vec_channels(prefix: str, arr: np.ndarray, leaves) -> dict[str, np.ndarray]:
    """Expand a (T, K) field into {f"{prefix}/{leaf}": (T,)} for each leaf."""
    arr = np.asarray(arr)
    return {f"{prefix}/{leaf}": arr[:, i] for i, leaf in enumerate(leaves)}


def pad_to_T(arr: np.ndarray, T: int) -> np.ndarray:
    """Right-align a 1-D series onto a length-T grid (leading edge replicated).

    Finite-difference series (accel = T-2, jerk = T-3) are causal — sample i is built
    from frames ending at i — so they belong at the tail; the few warm-up frames repeat
    the first valid value. Longer-than-T input is truncated to the first T.
    """
    arr = np.asarray(arr, dtype=float)
    n = arr.shape[0]
    if n == T:
        return arr
    if n > T:
        return arr[:T]
    return np.concatenate([np.repeat(arr[:1], T - n), arr])


def _probe_leaves(n: int) -> list[str]:
    return [f"probe_{i:03d}" for i in range(n)]


def _com(result, ctx) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if getattr(result, "com", None) is not None:
        out |= vec_channels("dynamics/com", result.com, _XYZ)
    if getattr(result, "com_ref", None) is not None:
        out |= vec_channels("dynamics/com_ref", result.com_ref, _XYZ)
    return out


def _ang_momentum(result, ctx) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if getattr(result, "angular_momentum", None) is not None:
        out |= vec_channels("dynamics/ang_momentum", result.angular_momentum, _XYZ)
    if getattr(result, "angular_momentum_ref", None) is not None:
        out |= vec_channels("dynamics/ang_momentum_ref", result.angular_momentum_ref, _XYZ)
    return out


def _foot_slip(result, ctx) -> dict[str, np.ndarray]:
    fs = getattr(result, "foot_slip", None)
    return {"diag/foot_slip": np.asarray(fs)} if fs is not None else {}


def _seg_min_channels(prefix: str, arr: np.ndarray, segs: np.ndarray,
                      names) -> dict[str, np.ndarray]:
    """Aggregate (T, N) per-probe distances to per-segment minima (closest to surface)."""
    segs = np.asarray(segs)
    out: dict[str, np.ndarray] = {}
    for s in np.unique(segs):
        nm = names[int(s)] if names is not None and int(s) < len(names) else f"seg_{int(s):02d}"
        out[f"{prefix}/{nm}"] = arr[:, segs == s].min(axis=1)
    return out


def _distances(result, ctx) -> dict[str, np.ndarray]:
    """Human-side SDF distances, aggregated per body segment when ctx provides probe
    segment labels (else one column per probe)."""
    segs = getattr(ctx, "probe_segments", None) if ctx is not None else None
    names = getattr(ctx, "probe_segment_names", None) if ctx is not None else None
    out: dict[str, np.ndarray] = {}
    for field, prefix in (("human_flr_dist", "diag/human_flr_dist"),
                          ("human_obj_dist", "diag/human_obj_dist")):
        arr = getattr(result, field, None)
        if arr is None:
            continue
        arr = np.asarray(arr)
        if segs is not None and len(segs) == arr.shape[1]:
            out |= _seg_min_channels(prefix, arr, segs, names)
        else:
            out |= vec_channels(prefix, arr, _probe_leaves(arr.shape[1]))
    return out


def _solver_cost(result, ctx) -> dict[str, np.ndarray]:
    c = getattr(result, "per_frame_cost", None)
    return {"solver/cost": np.asarray(c)} if c is not None else {}


def _smoothness(result, ctx) -> dict[str, np.ndarray]:
    """Per-frame base / per-joint acceleration and jerk (needs ctx.dof + ctx.dt).

    Off unless ctx.dof is set, so qpos-derived channels never misread trailing object
    DOFs as joints. Base channels are the per-frame magnitudes; joint channels are
    per-actuated-joint, all right-aligned to T via pad_to_T.
    """
    if ctx is None or ctx.dof is None:
        return {}
    dof = int(ctx.dof)
    qpos = np.asarray(result.qpos)
    T = qpos.shape[0]
    if T < 4 or dof <= 0:
        return {}
    from HoloNew.evaluation.metrics.smoothness import smoothness_series
    s = smoothness_series(qpos, dof, ctx.dt)
    out: dict[str, np.ndarray] = {
        "smoothness/base_pos_accel": pad_to_T(np.linalg.norm(s["base_acc"], axis=1), T),
        "smoothness/base_ang_accel": pad_to_T(np.linalg.norm(s["base_ang_acc"], axis=1), T),
    }
    names = ctx.joint_names or [f"joint_{i:03d}" for i in range(dof)]
    for i, nm in enumerate(names):
        out[f"smoothness/joint_accel/{nm}"] = pad_to_T(s["joint_accel"][:, i], T)
        out[f"smoothness/joint_jerk/{nm}"] = pad_to_T(s["joint_jerk"][:, i], T)
    return out


def _effort(result, ctx) -> dict[str, np.ndarray]:
    """Per-frame limit margin / saturation / velocity for the limited joints.

    Off unless ctx carries the limited-joint columns + bounds. ``margin`` / ``saturated``
    are frame-aligned (T); ``vel`` (T-1) is right-aligned via pad_to_T.
    """
    cols = getattr(ctx, "joint_limit_cols", None) if ctx is not None else None
    if ctx is None or ctx.dof is None or cols is None or len(cols) == 0:
        return {}
    qpos = np.asarray(result.qpos)
    T = qpos.shape[0]
    if T < 2:
        return {}
    joints = qpos[:, 7:7 + int(ctx.dof)][:, np.asarray(cols, dtype=int)]   # (T, C)
    from HoloNew.evaluation.metrics.effort import effort_series
    s = effort_series(joints, ctx.joint_limit_lower, ctx.joint_limit_upper, ctx.dt)
    names = ctx.joint_limit_names or [f"joint_{int(c):03d}" for c in cols]
    out: dict[str, np.ndarray] = {}
    for i, nm in enumerate(names):
        out[f"effort/joint_margin/{nm}"] = pad_to_T(s["margin"][:, i], T)
        out[f"effort/joint_vel/{nm}"] = pad_to_T(s["vel"][:, i], T)
        out[f"effort/saturated/{nm}"] = pad_to_T(s["saturated"][:, i].astype(float), T)
    return out


def _extra(result, ctx) -> dict[str, np.ndarray]:
    """Emit CLI-injected channels (tracking / style / contacts) verbatim."""
    ex = getattr(ctx, "extra_channels", None) if ctx is not None else None
    return {k: np.asarray(v) for k, v in ex.items()} if ex else {}


PRODUCERS = [
    ("com", _com),
    ("ang_momentum", _ang_momentum),
    ("foot_slip", _foot_slip),
    ("distances", _distances),
    ("solver_cost", _solver_cost),
    ("smoothness", _smoothness),
    ("effort", _effort),
    ("extra", _extra),
]


def run_all(result, ctx: SignalContext | None = None) -> dict[str, np.ndarray]:
    """Merge every producer's channels (later producers cannot overwrite earlier keys)."""
    if ctx is None:
        ctx = SignalContext()
    channels: dict[str, np.ndarray] = {}
    for _name, fn in PRODUCERS:
        for cname, arr in (fn(result, ctx) or {}).items():
            channels[cname] = arr
    return channels
