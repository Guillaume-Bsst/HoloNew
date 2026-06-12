"""Standalone viser app to visualize the per-frame contact field (data only).

Loads the bundled demo contact field and shows, for the selected frame, the human
contact witness points colored by their signed distance to the object (human_object
channel): red = penetrating/touching, blue = far (at the margin). Pure numpy (no
coal/SMPL-X). A frame slider scrubs time. Does not run any solve.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import viser

from HoloNew.src.test_socp.contact.contact_io import load_contact_fields
from HoloNew.src.test_socp.contact.constants import CONTACT_MARGIN_M


def _bundled() -> Path:
    return Path(__file__).resolve().parent.parent / "assets" / "contact" / "contact_sub3_largebox_003.npz"


def _color(dist: np.ndarray, margin: float) -> np.ndarray:
    t = np.clip(dist / margin, 0.0, 1.0)          # 0 (near) -> red, 1 (far) -> blue
    c = np.zeros((dist.shape[0], 3), np.uint8)
    c[:, 0] = ((1.0 - t) * 255).astype(np.uint8)
    c[:, 2] = (t * 255).astype(np.uint8)
    return c


def main() -> None:
    fields = load_contact_fields(_bundled())
    ch = fields["human_object"]
    T = ch.distance.shape[0]

    server = viser.ViserServer()
    server.scene.set_up_direction("+z")
    slider = server.gui.add_slider("Frame", min=0, max=T - 1, step=1, initial_value=0)

    def redraw(t: int) -> None:
        pts = ch.witness[t].astype(np.float32)
        colors = _color(ch.distance[t], CONTACT_MARGIN_M)
        server.scene.add_point_cloud("/contact", points=pts, colors=colors, point_size=0.01)

    @slider.on_update
    def _(_evt: viser.GuiEvent) -> None:
        redraw(int(slider.value))

    redraw(0)
    print("Contact viewer at http://localhost:8080 — Enter to exit ...")
    input("Enter to exit ...")


if __name__ == "__main__":
    main()
