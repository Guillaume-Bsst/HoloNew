"""Per-segment entropic optimal transport between the human source cloud and the robot target cloud.

Keyed by robot points: for each robot surface point it returns the human point that drives it. Each
soft coupling column is turned into a single human source via a barycentric image snapped to the
nearest human sample within the SAME segment. The assignment is functional, not injective -- two
robot points may share a human point (the human field is simply read at the same body location
twice), which suits driving every robot point without forcing the robot cloud sparser than the human.

Both clouds are in a matching T-pose and the same world frame, so corresponding segments share their
orientation: the per-segment cost is centre + isotropic scale + squared distance (up/forward/left
stay consistent -- no end-to-end flips, no left<->right mirror).
"""
from __future__ import annotations

import numpy as np


def couple(human_pts: np.ndarray, human_seg: np.ndarray, robot_pts: np.ndarray,
           robot_seg: np.ndarray, reg: float) -> np.ndarray:
    """``smpl_idx (M,)``: for each robot point, the index of the human point that drives it (into the
    human cloud's point order). ``reg`` = Sinkhorn entropic regularisation on per-segment normalised
    coordinates. Entries index into ``human_pts`` and may repeat (functional, not injective)."""
    import ot
    from scipy.spatial import cKDTree

    m = robot_pts.shape[0]
    smpl_idx = np.full(m, -1, dtype=np.int64)
    for s in np.unique(robot_seg):
        ir = np.flatnonzero(robot_seg == s)
        ih = np.flatnonzero(human_seg == s)
        if ih.size == 0:
            raise ValueError(f"segment {int(s)} has robot points but no human source points")
        xh = human_pts[ih].astype(np.float64)
        xr = robot_pts[ir].astype(np.float64)

        # Centre + isotropic scale each segment locally so the cost is about relative shape.
        hn = (xh - xh.mean(0)) / (xh.std() + 1e-8)
        rn = (xr - xr.mean(0)) / (xr.std() + 1e-8)

        cost = ot.dist(hn, rn, metric="sqeuclidean")
        cost /= (cost.max() + 1e-12)
        plan = ot.sinkhorn(np.full(ih.size, 1.0 / ih.size),
                           np.full(ir.size, 1.0 / ir.size), cost, reg)   # (n_h, n_r)

        # Barycentric image of each robot point in human coords, snapped to the nearest human sample
        # within the segment (functional: several robot points may land on the same human point).
        image = (plan.T @ xh) / (plan.sum(0)[:, None] + 1e-12)            # (n_r, 3)
        _, nn = cKDTree(xh).query(image, k=1)
        smpl_idx[ir] = ih[nn]

    if (smpl_idx < 0).any():
        raise ValueError("some robot points were left unassigned by the OT coupling")
    return smpl_idx
