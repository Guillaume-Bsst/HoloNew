"""Standalone viser app to visualize the human->G1 OT correspondence (data only).

Shows the G1 surface points colored by body segment (always available from the
bundled correspondence). If a SMPL-X model dir is available, also draws the human
rest surface colored consistently. Does not run any solve.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import viser
import yourdfpy
from viser.extras import ViserUrdf

from HoloNew.src.test_socp.correspondence.build_correspondence import load_correspondence
from HoloNew.src.test_socp.correspondence.constants import G1_29DOF_URDF, SMPLX_MODEL_DIR_DEFAULT
from HoloNew.src.test_socp.correspondence.segments import SEGMENTS
from HoloNew.src.test_socp.correspondence.viz import add_colored_points, segment_colors


def _bundled_corr() -> Path:
    return Path(__file__).resolve().parent.parent / "assets" / "correspondence" / "corr_neutral.npz"


def main() -> None:
    corr = load_correspondence(_bundled_corr())
    urdf = yourdfpy.URDF.load(G1_29DOF_URDF, load_meshes=True, build_scene_graph=True)
    urdf.update_cfg(corr.g1_rest_cfg)

    server = viser.ViserServer()
    server.scene.set_up_direction("+z")
    ViserUrdf(server, urdf_or_path=urdf, root_node_name="/g1")

    # Reconstruct world positions for each G1 correspondence point using FK.
    pts = np.zeros((corr.link_idx.shape[0], 3), np.float32)
    for li, link in enumerate(corr.link_names):
        T = np.asarray(urdf.get_transform(link))
        sel = corr.link_idx == li
        pts[sel] = corr.offset_local[sel] @ T[:3, :3].T + T[:3, 3]

    colors = segment_colors(corr.seg, len(SEGMENTS))
    add_colored_points(server, "/g1_corr_points", pts, colors, point_size=0.005)

    print("Correspondence viewer at http://localhost:8080 — Ctrl+C to exit")
    input("Enter to exit ...")


if __name__ == "__main__":
    main()
