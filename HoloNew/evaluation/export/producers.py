"""Signal producers: read (T,...) fields off a RetargetResult, emit flat (T,) channels.

A producer is fn(result) -> dict[str, np.ndarray] (each value 1-D length T), returning
{} when its source field is absent. PRODUCERS is the ordered registry — the single
place to extend when adding a new signal. ``run_all`` merges every producer's output.
"""
from __future__ import annotations

import numpy as np

_XYZ = ("x", "y", "z")


def vec_channels(prefix: str, arr: np.ndarray, leaves) -> dict[str, np.ndarray]:
    """Expand a (T, K) field into {f"{prefix}/{leaf}": (T,)} for each leaf."""
    arr = np.asarray(arr)
    return {f"{prefix}/{leaf}": arr[:, i] for i, leaf in enumerate(leaves)}


def _probe_leaves(n: int) -> list[str]:
    return [f"probe_{i:03d}" for i in range(n)]


def _com(result) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if getattr(result, "com", None) is not None:
        out |= vec_channels("dynamics/com", result.com, _XYZ)
    if getattr(result, "com_ref", None) is not None:
        out |= vec_channels("dynamics/com_ref", result.com_ref, _XYZ)
    return out


def _ang_momentum(result) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if getattr(result, "angular_momentum", None) is not None:
        out |= vec_channels("dynamics/ang_momentum", result.angular_momentum, _XYZ)
    if getattr(result, "angular_momentum_ref", None) is not None:
        out |= vec_channels("dynamics/ang_momentum_ref", result.angular_momentum_ref, _XYZ)
    return out


def _foot_slip(result) -> dict[str, np.ndarray]:
    fs = getattr(result, "foot_slip", None)
    return {"diag/foot_slip": np.asarray(fs)} if fs is not None else {}


def _distances(result) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    flr = getattr(result, "human_flr_dist", None)
    if flr is not None:
        flr = np.asarray(flr)
        out |= vec_channels("diag/human_flr_dist", flr, _probe_leaves(flr.shape[1]))
    obj = getattr(result, "human_obj_dist", None)
    if obj is not None:
        obj = np.asarray(obj)
        out |= vec_channels("diag/human_obj_dist", obj, _probe_leaves(obj.shape[1]))
    return out


def _solver_cost(result) -> dict[str, np.ndarray]:
    c = getattr(result, "per_frame_cost", None)
    return {"solver/cost": np.asarray(c)} if c is not None else {}


PRODUCERS = [
    ("com", _com),
    ("ang_momentum", _ang_momentum),
    ("foot_slip", _foot_slip),
    ("distances", _distances),
    ("solver_cost", _solver_cost),
]


def run_all(result) -> dict[str, np.ndarray]:
    """Merge every producer's channels (later producers cannot overwrite earlier keys)."""
    channels: dict[str, np.ndarray] = {}
    for _name, fn in PRODUCERS:
        for cname, arr in (fn(result) or {}).items():
            channels[cname] = arr
    return channels
