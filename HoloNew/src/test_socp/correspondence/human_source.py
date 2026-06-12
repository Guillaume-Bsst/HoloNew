"""The human source cloud: the stable-identity surface samples (PointCloudCache)
evaluated on the rest-pose mesh, each tagged with its body segment. This is the
exact set the contact field is indexed by, so the saved correspondence is keyed
1:1 with online contact signals.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .segments import point_segments

# SMPL-X rest pose is Y-up facing +Z; the G1 (URDF) world is Z-up facing +X. The OT
# couples the human against the G1 in the G1 world frame, so the human cloud is
# rotated into it. Without this the per-segment axis anchors (which reference world
# up / forward) disagree between the two bodies and central segments map mirrored
# left<->right (e.g. left torso -> right G1 torso). Apply as points @ R.T.
SMPLX_TO_G1_FRAME = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])


def to_g1_frame(points: np.ndarray) -> np.ndarray:
    """Rotate SMPL-X-frame points (N,3) into the G1 world frame (Z-up, facing +X)."""
    return np.asarray(points) @ SMPLX_TO_G1_FRAME.T


@dataclass(frozen=True)
class HumanSource:
    points: np.ndarray    # (N, 3) rest-pose surface samples, G1 world frame
    tri_idx: np.ndarray   # (N,)   triangle index per sample (cache identity)
    bary: np.ndarray      # (N, 3) barycentric weights per sample (cache identity)
    seg: np.ndarray       # (N,)   segment index (into segments.SEGMENTS)


def build_human_source(body, density: float) -> HumanSource:
    """Sample the rest-pose SMPL-X surface at `density` pts/m² and label segments."""
    cache = body.build_point_cloud_cache(density)
    rest = body.rest_verts()
    tri = rest[body.faces[cache.tri_idx]]               # (N, 3, 3)
    points = np.einsum("nij,ni->nj", tri, cache.bary)
    points = to_g1_frame(points).astype(np.float32)     # SMPL-X -> G1 world frame
    lbs = body.model.lbs_weights.detach().cpu().numpy()  # (V, 55)
    seg = point_segments(lbs, body.faces, cache.tri_idx, cache.bary)
    return HumanSource(points=points, tri_idx=cache.tri_idx, bary=cache.bary, seg=seg)
