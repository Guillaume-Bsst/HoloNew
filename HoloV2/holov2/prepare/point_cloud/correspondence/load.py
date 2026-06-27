"""Reads the cached human<->robot correspondence ``.npz`` into the V2 contracts (the reader side of
``build.py``, which writes it). Returns the ``CorrespondenceTable`` plus the embedded
``SurfaceSampling`` the human cloud must reuse to keep ``smpl_idx`` valid.

Field mapping (.npz -> contract): ``human_idx`` -> ``smpl_idx`` (each robot point's driving human
sample), ``link_idx`` / ``offset_local`` / ``link_names`` carry over directly, and ``tri_idx`` /
``bary`` are the canonical sampling. ``smpl_sampling_id`` is stamped to match the sampling so the
runner's binding assertion (``cloud.sampling_id == table.smpl_sampling_id``) holds.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ....contracts import CorrespondenceTable
from ..sampling import SurfaceSampling, sampling_id


def load_correspondence(path: Path) -> tuple[CorrespondenceTable, SurfaceSampling]:
    """Read ``corr_neutral.npz`` -> ``(CorrespondenceTable, SurfaceSampling)``. The sampling is the
    human cloud's canonical identity; the human cloud bakes against it and inherits its id."""
    d = np.load(Path(path), allow_pickle=False)
    sid = sampling_id(d["tri_idx"], d["bary"])
    sampling = SurfaceSampling(tri_idx=d["tri_idx"].astype(np.int64),
                               bary=d["bary"].astype(np.float32), sampling_id=sid)
    table = CorrespondenceTable(
        smpl_idx=d["human_idx"].astype(np.int64),
        link_idx=d["link_idx"].astype(np.int64),
        offset_local=d["offset_local"].astype(np.float32),
        link_names=tuple(str(x) for x in d["link_names"]),
        smpl_sampling_id=sid,
    )
    if int(table.smpl_idx.max()) >= sampling.n_points:
        raise ValueError(f"smpl_idx max {int(table.smpl_idx.max())} out of range for "
                         f"{sampling.n_points} human samples — cache is inconsistent")
    return table, sampling
