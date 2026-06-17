"""Reference/FK-dependent per-frame channels (tracking + per-link orientation).

These families need the retargeter's FK and the GMR-grounded per-link reference, both
provided by ``HoloNew.evaluation.reference_context.ReferenceContext``. The heavy FK runs
here (CLI side), not in the pure producers; the result is a flat dict of (T,) channels
the CLI injects through ``SignalContext.extra_channels``.
"""
from __future__ import annotations

import numpy as np


def tracking_channels(ref_ctx, qpos: np.ndarray) -> dict[str, np.ndarray]:
    """Per-link position (MPJPE, root-relative) + pelvis-frame orientation channels.

    ref_ctx: a ReferenceContext wrapping the retargeter. qpos: (T, nq) solved trajectory.
    Reference is the GMR-grounded per-link target (consistent with style/roots). Returns
    {tracking/mpjpe/<link>, tracking/mpjpe_root_rel/<link>, tracking/orient/<link>
    (tracked links only), tracking/base_track}, each length T' = min(T, reference T).
    """
    from HoloNew.evaluation.metrics.tracking import tracking_series
    from HoloNew.evaluation.metrics.style import style_series

    qpos = np.asarray(qpos)
    T = min(int(qpos.shape[0]), int(ref_ctx._gpos.shape[0]))
    rot_m, pos_m = ref_ctx.fk_links(qpos[:T])          # solved FK: (T,K,3,3), (T,K,3)
    rot_ref, pos_ref = ref_ctx.reference_RP(T)         # reference targets
    pelvis = ref_ctx.pelvis_idx

    ts = tracking_series(pos_m, pos_ref, pelvis,
                         base_xyz=qpos[:T, 0:3], ref_root_xyz=pos_ref[:, pelvis])
    ss = style_series(rot_m, pos_m, rot_ref, pos_ref, pelvis, ref_ctx.tracked)

    names = ref_ctx.body_order
    tracked = np.asarray(ref_ctx.tracked, dtype=bool)
    out: dict[str, np.ndarray] = {}
    for k, nm in enumerate(names):
        out[f"tracking/mpjpe/{nm}"] = ts["mpjpe"][:, k]
        out[f"tracking/mpjpe_root_rel/{nm}"] = ts["mpjpe_root_rel"][:, k]
        if tracked[k]:
            out[f"tracking/orient/{nm}"] = ss["orient"][:, k]
    out["tracking/base_track"] = ts["base_track"]
    return out
