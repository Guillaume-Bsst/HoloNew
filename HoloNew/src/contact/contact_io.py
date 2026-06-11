"""Save/load per-frame contact channels (dict[str, ContactField]) as a numpy .npz.

Pure numpy — lets the bundled demo contact be loaded without coal or SMPL-X. Each
channel's four (T, ...)-stacked arrays are stored under prefixed keys.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .contact_field import ContactField


def save_contact_fields(path, fields: dict[str, ContactField]) -> None:
    arrays = {"channels": np.array(list(fields), dtype="<U32")}
    for name, f in fields.items():
        arrays[f"{name}/distance"] = f.distance
        arrays[f"{name}/direction"] = f.direction
        arrays[f"{name}/witness"] = f.witness
        arrays[f"{name}/active"] = f.active
    np.savez_compressed(str(Path(path)), **arrays)


def load_contact_fields(path) -> dict[str, ContactField]:
    d = np.load(str(Path(path)))
    out = {}
    for name in (str(c) for c in d["channels"]):
        out[name] = ContactField(
            distance=d[f"{name}/distance"], direction=d[f"{name}/direction"],
            witness=d[f"{name}/witness"], active=d[f"{name}/active"],
        )
    return out
