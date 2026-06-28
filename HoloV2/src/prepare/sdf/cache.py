"""(De)serialisation of an ``SDF`` ``.npz`` — save AND load in one place, so the writer and the
reader of the format cannot drift apart. The asset is a signed-distance + witness grid; the grids are
large, hence ``np.savez_compressed``. ``SdfBuilder.save``/``load`` delegate here in one line.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..contracts import SDF


def save_sdf(sdf: SDF, path: Path) -> None:
    """Serialise an ``SDF`` to ``path`` (``np.savez_compressed`` — the grids are large), creating
    parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), grid=sdf.grid, witness=sdf.witness,
                        origin=sdf.origin, spacing=np.float64(sdf.spacing),
                        name=np.array(sdf.name))


def load_sdf(path: Path) -> SDF:
    """Inverse of ``save_sdf``: load an ``SDF`` from ``path``."""
    d = np.load(str(path), allow_pickle=False)
    return SDF(grid=d["grid"], witness=d["witness"], origin=d["origin"],
               spacing=float(d["spacing"]), name=str(d["name"]))
