"""Standalone viser app to visualize the human->G1 OT correspondence.

Identical rendering to the test_pipe transport viz: builds the correspondence live
(human point cloud + sampled G1 surface + OT coupling), then draws the human and
G1 clouds side by side. With ``--color transfer`` (default) a continuous field
over the human is painted onto the G1 through the correspondence, so the transport
is visible; ``--color segment`` uses one hue per body segment. Match lines connect
each G1 point to its human source. Needs the SMPL-X model dir (defaults to
SMPLX_MODEL_DIR_DEFAULT).
"""
from __future__ import annotations

import argparse

import numpy as np
import viser
import yourdfpy

from HoloNew.src.test_socp.correspondence.constants import (
    G1_29DOF_URDF,
    HUMAN_GRID_DENSITY,
    SMPLX_MODEL_DIR_DEFAULT,
)
from HoloNew.src.test_socp.correspondence.g1_surface import sample_g1_surface
from HoloNew.src.test_socp.correspondence.human_body import HumanBody
from HoloNew.src.test_socp.correspondence.human_source import build_human_source
from HoloNew.src.test_socp.correspondence.ot_couple import couple
from HoloNew.src.test_socp.correspondence.segments import SEGMENTS
from HoloNew.src.test_socp.correspondence.viz import continuous_colors, segment_colors


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualise the human->G1 OT correspondence.")
    ap.add_argument("--model-dir", default=SMPLX_MODEL_DIR_DEFAULT)
    ap.add_argument("--gender", default="neutral")
    ap.add_argument("--urdf", default=str(G1_29DOF_URDF))
    ap.add_argument("--human-density", type=float, default=HUMAN_GRID_DENSITY)
    ap.add_argument("--g1-density", type=float, default=3000.0)
    ap.add_argument("--reg", type=float, default=0.005)
    ap.add_argument("--lines", type=int, default=300, help="match lines to draw")
    ap.add_argument("--color", choices=("segment", "transfer"), default="transfer",
                    help="segment: one hue per segment; transfer: continuous human "
                         "field painted onto the G1 via the correspondence")
    args = ap.parse_args()

    body = HumanBody(args.model_dir, None, args.gender)
    src = build_human_source(body, args.human_density)
    urdf = yourdfpy.URDF.load(args.urdf, load_meshes=True, build_scene_graph=True)
    tgt = sample_g1_surface(urdf, args.g1_density)
    human_idx = couple(src, tgt, reg=args.reg)   # (M,) human point driving each G1 point

    # build_human_source already rotated the cloud into the G1 world frame (Z-up).
    human = src.points                            # (N,3) full human cloud
    g1 = tgt.points_world                         # (M,3) G1 surface cloud

    # Offset the human cloud sideways so both are visible side by side.
    shift = np.array([1.0, 0.0, 0.0], np.float32)
    if args.color == "segment":
        h_colors = segment_colors(src.seg, len(SEGMENTS))
        g_colors = segment_colors(tgt.seg, len(SEGMENTS))
    else:
        field = continuous_colors(human)          # smooth human field, per human point
        h_colors = field                          # human shown with its own field
        g_colors = field[human_idx]               # each G1 point reads its human's colour

    server = viser.ViserServer()
    server.scene.set_up_direction("+z")
    server.scene.add_point_cloud(
        "/human", points=(human + shift).astype(np.float32), colors=h_colors, point_size=0.006)
    # The real G1 cloud, coloured either by its own segment or by the human field it reads.
    server.scene.add_point_cloud(
        "/g1", points=g1.astype(np.float32), colors=g_colors, point_size=0.006)

    rng = np.random.default_rng(0)
    sel = rng.choice(len(human_idx), size=min(args.lines, len(human_idx)), replace=False)
    for k, j in enumerate(sel):
        server.scene.add_spline_catmull_rom(
            f"/match/{k}",
            positions=np.stack([g1[j], human[human_idx[j]] + shift]).astype(np.float32),
            color=tuple(int(c) for c in g_colors[j]), line_width=1.0)

    print("Correspondence viewer at http://localhost:8080 — Enter to exit")
    input("Enter to exit ...")


if __name__ == "__main__":
    main()
