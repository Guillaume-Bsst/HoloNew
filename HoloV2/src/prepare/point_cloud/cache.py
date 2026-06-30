"""(Dé)sérialisation d'un ``PointCloud`` — sauvegarde ET chargement au même endroit, partagé par les
2 builders de nuages. Comme les builders humain et objet créent le même contrat ``PointCloud``, son
aller-retour ``.npz`` réside ici au lieu d'être dupliqué dans chaque ``AssetBuilder.save``/``load``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..contracts import PointCloud


def save_cloud(cloud: PointCloud, path: Path) -> None:
    """Écrit un ``PointCloud`` dans ``path`` comme un ``.npz`` compact (parts/weights/offsets + sampling id),
    en créant les répertoires parents si nécessaire."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), parts=cloud.parts, weights=cloud.weights, offsets=cloud.offsets,
             sampling_id=np.str_(cloud.sampling_id))


def load_cloud(path: Path) -> PointCloud:
    """Relit un ``PointCloud`` sauvegardé par ``save_cloud`` (aller-retour exact)."""
    d = np.load(Path(path), allow_pickle=False)
    return PointCloud(parts=d["parts"], weights=d["weights"], offsets=d["offsets"],
                      sampling_id=str(d["sampling_id"]))
