"""(Dé)sérialisation d'un ``SDF`` ``.npz`` — save ET load au même endroit, pour que le writer et le
reader du format ne s'éloignent pas. L'asset est une grille distance-signée + witness ; les grilles
sont grandes, d'où ``np.savez_compressed``. ``SdfBuilder.save``/``load`` délèguent ici en une ligne.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..contracts import SDF


def save_sdf(sdf: SDF, path: Path) -> None:
    """Sérialise un ``SDF`` vers ``path`` (``np.savez_compressed`` — les grilles sont grandes), crée
    les dossiers parents au besoin."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), grid=sdf.grid, witness=sdf.witness,
                        origin=sdf.origin, spacing=np.float64(sdf.spacing),
                        name=np.array(sdf.name))


def load_sdf(path: Path) -> SDF:
    """Inverse de ``save_sdf`` : charge un ``SDF`` depuis ``path``."""
    d = np.load(str(path), allow_pickle=False)
    return SDF(grid=d["grid"], witness=d["witness"], origin=d["origin"],
               spacing=float(d["spacing"]), name=str(d["name"]))
