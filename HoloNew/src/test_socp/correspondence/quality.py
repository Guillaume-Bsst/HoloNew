"""Quality metrics for the G1->human OT correspondence, computed without any
ground-truth pairing. For each segment we measure properties a good map must have:

- axial monotonicity: walking proximal->distal on the G1 limb walks
  proximal->distal on the human limb (catches flips and axis scrambling);
- neighbour preservation: G1 points that are close map to human points that are
  close (catches tears/folds in the map);
- coverage: how many distinct human points are used (catches collapse, i.e. many
  G1 points piling onto one human point).

Both clouds are taken in their rest-pose world frame: tgt.points_world (G1) and the
image points src.points[human_idx] returned by couple().
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .segments import SEGMENTS


@dataclass(frozen=True)
class SegmentQuality:
    segment: str
    n_points: int               # G1 points in the segment
    axial_monotonicity: float   # Spearman in [-1, 1]; ~+1 good, <0 flipped
    neighbour_preservation: float  # Spearman in [-1, 1]; ~+1 smooth map
    coverage: float             # distinct human points used / G1 points, in [0, 1]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation, NaN-safe for degenerate (constant) inputs."""
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def _oriented_axis_coord(X: np.ndarray, body_center: np.ndarray) -> np.ndarray:
    """Coordinate of each point along the cloud's principal axis, oriented so the
    far end (away from the body centre) is positive. Gives a sign-meaningful
    proximal->distal coordinate so a flip shows up as anti-correlation."""
    c = X.mean(0)
    Xc = X - c
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    axis = vt[0]
    far = X[np.linalg.norm(X - body_center, axis=1).argmax()]
    if axis @ (far - c) < 0:
        axis = -axis
    return Xc @ axis


def _neighbour_preservation(H: np.ndarray, G: np.ndarray, k: int, rng) -> float:
    """Spearman corr between human and image distances over each point's k human
    nearest neighbours. High = locally distance-preserving (smooth, no folds)."""
    from scipy.spatial import cKDTree

    n = len(H)
    if n < k + 2:
        return float("nan")
    _, nn = cKDTree(H).query(H, k=k + 1)        # (n, k+1), col 0 is self
    src_d, img_d = [], []
    for i in range(n):
        for j in nn[i, 1:]:
            src_d.append(np.linalg.norm(H[i] - H[j]))
            img_d.append(np.linalg.norm(G[i] - G[j]))
    return _spearman(np.asarray(src_d), np.asarray(img_d))


def segment_quality(
    g1_points: np.ndarray, image_points: np.ndarray, seg: np.ndarray,
    human_idx: np.ndarray, k: int = 8, seed: int = 0,
) -> list[SegmentQuality]:
    """Per-segment quality of the correspondence. g1_points[j] is a G1 surface point,
    image_points[j] the rest-pose world position of the human point chosen for it, and
    human_idx[j] its index into the human cloud (for the coverage count)."""
    rng = np.random.default_rng(seed)
    body_center_g = g1_points.mean(0)
    body_center_h = image_points.mean(0)

    out: list[SegmentQuality] = []
    for si, name in enumerate(SEGMENTS):
        m = np.flatnonzero(seg == si)
        if m.size == 0:
            out.append(SegmentQuality(name, 0, float("nan"), float("nan"), float("nan")))
            continue
        G = g1_points[m].astype(np.float64)
        H = image_points[m].astype(np.float64)
        cg = _oriented_axis_coord(G, body_center_g)
        ch = _oriented_axis_coord(H, body_center_h)
        out.append(SegmentQuality(
            segment=name,
            n_points=int(m.size),
            axial_monotonicity=_spearman(cg, ch),
            neighbour_preservation=_neighbour_preservation(G, H, k, rng),
            coverage=len(np.unique(human_idx[m])) / m.size,
        ))
    return out


# Heuristic alert thresholds (a metric below this is suspect, worth a visual check).
THRESHOLDS = {"axial_monotonicity": 0.7, "neighbour_preservation": 0.5, "coverage": 0.3}


def format_report(rows: list[SegmentQuality]) -> str:
    """Human-readable table; flags any metric below its alert threshold with '!'."""
    lines = [f"{'segment':15s} {'n':>5s} {'axial':>7s} {'neigh':>7s} {'cover':>6s}"]
    for r in rows:
        def cell(v: float, key: str) -> str:
            if np.isnan(v):
                return "   -  "
            flag = "!" if v < THRESHOLDS[key] else " "
            return f"{v:6.2f}{flag}"
        lines.append(
            f"{r.segment:15s} {r.n_points:5d} "
            f"{cell(r.axial_monotonicity, 'axial_monotonicity')} "
            f"{cell(r.neighbour_preservation, 'neighbour_preservation')} "
            f"{cell(r.coverage, 'coverage')}"
        )
    return "\n".join(lines)


def main() -> None:
    import argparse

    import yourdfpy

    from .constants import G1_29DOF_URDF, HUMAN_GRID_DENSITY
    from .human_body import HumanBody

    from .g1_surface import sample_g1_surface
    from .human_source import build_human_source
    from .ot_couple import couple

    ap = argparse.ArgumentParser(description="Report human->G1 OT correspondence quality.")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--gender", default="neutral")
    ap.add_argument("--urdf", default=str(G1_29DOF_URDF))
    ap.add_argument("--human-density", type=float, default=HUMAN_GRID_DENSITY)
    ap.add_argument("--g1-density", type=float, default=50000.0)
    ap.add_argument("--reg", type=float, default=0.005)
    args = ap.parse_args()

    body = HumanBody(args.model_dir, None, args.gender)
    src = build_human_source(body, args.human_density)
    urdf = yourdfpy.URDF.load(args.urdf, load_meshes=True, build_scene_graph=True)
    tgt = sample_g1_surface(urdf, args.g1_density)
    human_idx = couple(src, tgt, reg=args.reg)
    image = src.points[human_idx]
    rows = segment_quality(tgt.points_world, image, tgt.seg, human_idx)
    print(format_report(rows))


if __name__ == "__main__":
    main()
