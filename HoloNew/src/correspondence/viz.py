"""Pure viser drawing helpers for the correspondence visualiser.

Ported from test_pipe_retargeting/transport/viz.py — only the two stateless
colour-generation utilities and a thin point-cloud wrapper are kept; all
SharedData, runner state and test_pipe-specific imports have been removed.
"""
from __future__ import annotations

import numpy as np


def segment_colors(seg: np.ndarray, n_segments: int) -> np.ndarray:
    """Return (N, 3) uint8 color per point, one distinct hue per segment index.

    Args:
        seg: (N,) integer segment indices.
        n_segments: total number of segments (palette size).

    Returns:
        (N, 3) uint8 array.
    """
    import colorsys

    palette = np.array(
        [
            [int(255 * c) for c in colorsys.hsv_to_rgb(i / n_segments, 0.65, 0.95)]
            for i in range(n_segments)
        ],
        dtype=np.uint8,
    )
    return palette[np.asarray(seg)]


def continuous_colors(points: np.ndarray) -> np.ndarray:
    """Return (N, 3) uint8 color from each point's world XYZ, min-max normalised to RGB.

    A smooth global gradient: transferred onto the G1 via the correspondence, a good
    map repaints the same gradient anatomically; flips/tears show as colour breaks.

    Args:
        points: (N, 3) float array of 3-D positions.

    Returns:
        (N, 3) uint8 array.
    """
    p = np.asarray(points, np.float64)
    lo, hi = p.min(0), p.max(0)
    return (255 * (p - lo) / (hi - lo + 1e-9)).astype(np.uint8)


def add_colored_points(
    server,
    name: str,
    points: np.ndarray,
    colors: np.ndarray,
    point_size: float = 0.005,
) -> object:
    """Add a colored point cloud to a viser scene.

    Args:
        server: viser.ViserServer instance.
        name: scene-graph path for the point cloud node.
        points: (N, 3) float32 world positions.
        colors: (N, 3) uint8 RGB colors.
        point_size: radius in metres (default 0.005).

    Returns:
        The viser PointCloudHandle.
    """
    return server.scene.add_point_cloud(
        name,
        points=np.asarray(points, np.float32),
        colors=colors,
        point_size=point_size,
    )
