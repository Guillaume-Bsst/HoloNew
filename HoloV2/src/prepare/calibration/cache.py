"""(Dé)sérialisation d'une ``Calibration`` ``.npz`` — save ET load au même endroit, pour que le
writer et le reader du format ne s'éloignent pas. L'asset est l'ANCRAGE per-(sujet, prise) : un
décalage sol humain, un décalage sol objet partagé et une frame racine (pas de stature — ça vit sur
``BodyModel``). ``CalibrationBuilder.save``/``load`` délèguent ici en une ligne.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..contracts import Calibration


def save_calibration(calib: Calibration, path: Path) -> None:
    """Sérialise une ``Calibration`` vers ``path`` (``np.savez``), crée les dossiers parents au besoin."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), human_offset=np.float64(calib.human_offset),
             object_offset=np.float64(calib.object_offset),
             root_frame=np.asarray(calib.root_frame, np.float64))


def load_calibration(path: Path) -> Calibration:
    """Inverse de ``save_calibration`` : charge une ``Calibration`` depuis ``path``."""
    d = np.load(str(path), allow_pickle=False)
    return Calibration(human_offset=float(d["human_offset"]), object_offset=float(d["object_offset"]),
                       root_frame=np.asarray(d["root_frame"], np.float64))
