"""(Dé)sérialisation .npz d'une ``GeodesicTable`` — save ET load au même endroit, pour que le writer
et le reader ne dérivent pas. ``geo`` est une matrice (P,P) f32 → ``np.savez_compressed``."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..contracts import GeodesicTable


def save_geo(table: GeodesicTable, path: Path) -> None:
    """Sérialise une ``GeodesicTable`` vers ``path`` (crée les dossiers parents)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), points=table.points, normals=table.normals, geo=table.geo,
                        name=np.array(table.name), sampling_id=np.array(table.sampling_id))


def load_geo(path: Path) -> GeodesicTable:
    """Inverse de ``save_geo``."""
    d = np.load(str(path), allow_pickle=False)
    return GeodesicTable(points=d["points"], normals=d["normals"], geo=d["geo"],
                         name=str(d["name"]), sampling_id=str(d["sampling_id"]))
