"""(De)serialisation of the human<->robot correspondence ``.npz`` — save AND load in one place, so
the writer and the reader of the format cannot drift apart. The asset is the pair
``(CorrespondenceTable, SurfaceSampling)``: the table maps each robot surface point to its driving
human sample, the sampling is the canonical ``(tri_idx, bary)`` the human cloud must reuse to keep
``smpl_idx`` valid.

Field mapping (contract <-> .npz): ``smpl_idx`` <-> ``human_idx`` (each robot point's driving human
sample), ``link_idx`` / ``offset_local`` / ``link_names`` carry over directly, and ``tri_idx`` /
``bary`` are the canonical sampling. ``smpl_sampling_id`` is not stored: it is recomputed from
``tri_idx`` / ``bary`` on load so it always matches the sampling and the runner's binding assertion
(``cloud.sampling_id == table.smpl_sampling_id``) holds.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ....contracts import CorrespondenceTable
from ..sampling import SurfaceSampling, sampling_id


def save_correspondence(asset: tuple[CorrespondenceTable, SurfaceSampling], path: Path) -> None:
    """Write the ``(CorrespondenceTable, SurfaceSampling)`` pair to ``path`` as a ``.npz`` (creating
    parent dirs). The schema is the one ``load_correspondence`` reads; ``smpl_sampling_id`` is left
    out (rederived from ``tri_idx`` / ``bary`` on load)."""
    table, sampling = asset
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(Path(path), human_idx=table.smpl_idx, link_idx=table.link_idx,
             offset_local=table.offset_local, link_names=np.array(table.link_names, dtype="<U64"),
             tri_idx=sampling.tri_idx, bary=sampling.bary)


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
