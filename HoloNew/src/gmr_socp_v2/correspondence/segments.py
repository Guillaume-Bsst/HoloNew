"""The 14 body segments shared by the human (SMPL-X) and the G1, plus the folds
that assign each SMPL-X joint and each G1 link to one segment, and a helper that
labels each human surface sample by its dominant skinning joint's segment.

Per-segment optimal transport keeps hand->hand, foot->foot: a human point can only
be coupled to G1 points sharing its segment.
"""
from __future__ import annotations

import numpy as np

SEGMENTS: list[str] = [
    "pelvis", "torso", "head",
    "left_thigh", "left_shank", "left_foot",
    "right_thigh", "right_shank", "right_foot",
    "left_upperarm", "left_forearm", "left_hand",
    "right_upperarm", "right_forearm", "right_hand",
]
_SEG_IDX: dict[str, int] = {s: i for i, s in enumerate(SEGMENTS)}


def _side_segment(name: str, side: str) -> str | None:
    """Segment for a G1 link known to belong to `side` ('left'/'right')."""
    # Quick check: does the link name contain the side prefix or indicator?
    # G1 uses 'left_'/'right_' or sometimes '_L_'/_R_' in constraints.
    side_match = (f"{side}_" in name or f"_{side[0].upper()}_" in name or name.startswith(f"{side[0].upper()}_"))
    if not side_match:
        return None

    # One anatomical rule, applied identically to both bodies (no per-side tweaks):
    #   hip -> thigh | knee -> shank | ankle, foot -> foot
    #   shoulder -> upperarm | elbow, wrist -> forearm | hand, fingers -> hand
    if "hip" in name:
        return f"{side}_thigh"
    if "knee" in name:
        return f"{side}_shank"
    if "ankle" in name or "foot" in name:
        return f"{side}_foot"
    if "shoulder" in name:
        return f"{side}_upperarm"
    if "elbow" in name or "wrist" in name:
        return f"{side}_forearm"
    if any(k in name for k in ("hand", "palm", "thumb", "index", "middle", "pinky", "finger")):
        return f"{side}_hand"
    return None


def g1_link_to_segment(link_name: str) -> str:
    """Map a G1 URDF link name to one of SEGMENTS (waist/torso/head -> torso/head)."""
    if link_name in ("pelvis", "pelvis_contour_link"):
        return "pelvis"
    if "head" in link_name:
        return "head"
    for side in ("left", "right"):
        seg = _side_segment(link_name, side)
        if seg is not None:
            return seg
    return "torso"


# SMPL-X joint index -> segment, using the SAME anatomical rule as the G1 links
# (ankle -> foot, wrist -> forearm), so segment X means the same body part on both
# bodies. Body joints 0..21 explicit; jaw/eyes 22..24 -> head; hands 25..54 below.
_SMPLX_BODY: dict[int, str] = {
    0: "pelvis",
    1: "left_thigh", 2: "right_thigh",
    3: "torso", 6: "torso", 9: "torso", 12: "torso", 13: "torso", 14: "torso",
    15: "head",                          # head
    4: "left_shank", 5: "right_shank",   # knees
    7: "left_foot", 10: "left_foot",     # ankle + toes -> foot
    8: "right_foot", 11: "right_foot",   # ankle + toes -> foot
    16: "left_upperarm", 17: "right_upperarm",
    18: "left_forearm", 19: "right_forearm",   # elbows
    20: "left_forearm", 21: "right_forearm",   # wrists -> forearm
    22: "head", 23: "head", 24: "head",        # jaw/eyes -> head
}
SMPLX_JOINT_TO_SEGMENT: dict[int, str] = dict(_SMPLX_BODY)
for _j in range(25, 40):
    SMPLX_JOINT_TO_SEGMENT[_j] = "left_hand"
for _j in range(40, 55):
    SMPLX_JOINT_TO_SEGMENT[_j] = "right_hand"


def _vertex_segment_indices(lbs_weights: np.ndarray) -> np.ndarray:
    """(V,) int segment index per SMPL-X vertex, from its dominant skinning joint."""
    dom_joint = np.asarray(lbs_weights).argmax(axis=1)  # (V,)
    return np.array([_SEG_IDX[SMPLX_JOINT_TO_SEGMENT[int(j)]] for j in dom_joint], dtype=np.int64)


def point_segments(
    lbs_weights: np.ndarray, faces: np.ndarray,
    tri_idx: np.ndarray, bary: np.ndarray,
) -> np.ndarray:
    """(N,) int segment index per surface sample.

    Each sample takes the segment of its triangle vertex with the largest
    barycentric weight (the body location it sits closest to).
    """
    vseg = _vertex_segment_indices(lbs_weights)          # (V,)
    tri_v = np.asarray(faces)[np.asarray(tri_idx)]        # (N, 3) vertex ids
    dom_corner = np.asarray(bary).argmax(axis=1)          # (N,) in {0,1,2}
    dom_vertex = tri_v[np.arange(len(tri_idx)), dom_corner]
    return vseg[dom_vertex].astype(np.int64)
