"""Per-segment entropic OT between the human source cloud and the G1 target cloud,
turning each soft coupling column into a single human-point source via a barycentric
image snapped to the nearest human sample within the same segment.

The map is keyed by G1 points: for each G1 surface point it returns the human point
that drives it. The assignment is functional, not injective -- two G1 points may share
a human point (the human field is simply read at the same body location twice). This
suits driving every G1 point from a corresponding human value without forcing the G1
cloud to be sparser than the human cloud.

The human cloud is already in the G1 world frame (human_source.to_g1_frame) and both
bodies are in a matching T-pose, so corresponding segments share their orientation:
the per-segment cost is just centre + isotropic scale + squared distance, which keeps
up/forward/left consistent (no end-to-end flips, no left<->right mirror).
"""
from __future__ import annotations

import numpy as np


def couple(src, tgt, reg: float = 0.05) -> np.ndarray:
    """Return human_idx (M,): for each G1 point, the human point that drives it.

    src: HumanSource (points, seg).  tgt: G1Surface (points_world, seg, link_idx,
    offset_local).  reg: Sinkhorn entropic regularisation on per-segment normalised
    coordinates.  M = tgt.points_world.shape[0]; each entry indexes into src.* and may
    repeat (functional, not injective).
    """
    import ot
    from scipy.spatial import cKDTree

    m = tgt.points_world.shape[0]
    human_idx = np.full(m, -1, dtype=np.int64)

    for s in np.unique(tgt.seg):
        ig = np.flatnonzero(tgt.seg == s)
        ih = np.flatnonzero(src.seg == s)
        if ih.size == 0:
            raise ValueError(f"segment {int(s)} has G1 points but no human source points")
        Xh = src.points[ih].astype(np.float64)
        Xg = tgt.points_world[ig].astype(np.float64)

        # Centre + scale each segment locally so the cost is about relative shape.
        ch, cg = Xh.mean(0), Xg.mean(0)
        sh = Xh.std() + 1e-8
        sg = Xg.std() + 1e-8
        Hn = (Xh - ch) / sh
        Gn = (Xg - cg) / sg

        # Squared-Euclidean cost, uniform marginals, entropic OT plan.
        M = ot.dist(Hn, Gn, metric="sqeuclidean")
        M /= (M.max() + 1e-12)
        a = np.full(ih.size, 1.0 / ih.size)
        b = np.full(ig.size, 1.0 / ig.size)
        plan = ot.sinkhorn(a, b, M, reg)                 # (n_h, n_g)

        # Barycentric image of each G1 point in human world coords, then snap to the
        # nearest human sample within the segment. Functional (not injective): several
        # G1 points may legitimately land on the same human point.
        col = plan.sum(0)[:, None] + 1e-12               # (n_g, 1)
        image = (plan.T @ Xh) / col                      # (n_g, 3)
        _, nn = cKDTree(Xh).query(image, k=1)
        human_idx[ig] = ih[nn]

    if (human_idx < 0).any():
        raise ValueError("some G1 points were left unassigned")
    return human_idx
