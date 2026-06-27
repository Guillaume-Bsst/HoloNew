"""Disk format of a ``PointCloud`` — one place, shared by the cloud builders.

The human and the object builders bake the SAME ``PointCloud`` contract, so its ``.npz``
serialisation lives here instead of being duplicated in each ``AssetBuilder.save``/``load``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ...contracts import PointCloud


def save_cloud(cloud: PointCloud, path: Path) -> None:
    """Write a ``PointCloud`` to ``path`` as a compact ``.npz`` (parts/weights/offsets + sampling id)."""
    np.savez(Path(path), parts=cloud.parts, weights=cloud.weights, offsets=cloud.offsets,
             sampling_id=np.str_(cloud.sampling_id))


def load_cloud(path: Path) -> PointCloud:
    """Read back a ``PointCloud`` saved by ``save_cloud`` (round-trips exactly)."""
    d = np.load(Path(path), allow_pickle=False)
    return PointCloud(parts=d["parts"], weights=d["weights"], offsets=d["offsets"],
                      sampling_id=str(d["sampling_id"]))
