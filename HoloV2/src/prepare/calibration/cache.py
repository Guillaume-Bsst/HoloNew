"""(De)serialisation of a ``Calibration`` ``.npz`` — save AND load in one place, so the writer and
the reader of the format cannot drift apart. The asset is the per-(subject, take) GROUNDING: a human
floor offset, a shared object floor offset and a root frame (no stature — that lives on the
``BodyModel``). ``CalibrationBuilder.save``/``load`` delegate here in one line.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..contracts import Calibration


def save_calibration(calib: Calibration, path: Path) -> None:
    """Serialise a ``Calibration`` to ``path`` (``np.savez``), creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), human_offset=np.float64(calib.human_offset),
             object_offset=np.float64(calib.object_offset),
             root_frame=np.asarray(calib.root_frame, np.float64))


def load_calibration(path: Path) -> Calibration:
    """Inverse of ``save_calibration``: load a ``Calibration`` from ``path``."""
    d = np.load(str(path), allow_pickle=False)
    return Calibration(human_offset=float(d["human_offset"]), object_offset=float(d["object_offset"]),
                       root_frame=np.asarray(d["root_frame"], np.float64))
