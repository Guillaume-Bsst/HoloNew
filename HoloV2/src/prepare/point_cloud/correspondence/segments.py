"""Les 15 segments corporels partagés par l'humain (SMPL-X) et n'importe quel robot humanoides,
plus les cartes qui assignent chaque articulation SMPL-X et chaque lien du robot à un segment, et
un assistant étiqueté chaque échantillon de surface humain par le segment de son articulation de
skinning dominante.

Le transport optimal par segment garde main→main, pied→pied : un point humain ne peut être associé
qu'à des points du robot partageant son segment. Une règle anatomique unique est appliquée
identiquement aux deux corps (pas d'ajustements par côté), donc le segment X signifie la même
partie du corps sur chacun.
"""
from __future__ import annotations

import numpy as np

SEGMENTS: tuple[str, ...] = (
    "pelvis", "torso", "head",
    "left_thigh", "left_shank", "left_foot",
    "right_thigh", "right_shank", "right_foot",
    "left_upperarm", "left_forearm", "left_hand",
    "right_upperarm", "right_forearm", "right_hand",
)
_SEG_IDX: dict[str, int] = {s: i for i, s in enumerate(SEGMENTS)}


def _side_segment(name: str, side: str) -> str | None:
    """Segment pour un lien du robot connu pour appartenir au ``side`` ('left'/'right'), ou None
    sinon. Règle anatomique : hip->thigh | knee->shank | ankle/foot->foot ; shoulder->upperarm |
    elbow/wrist->forearm | hand/fingers->hand."""
    if not (f"{side}_" in name or f"_{side[0].upper()}_" in name or name.startswith(f"{side[0].upper()}_")):
        return None
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


def link_to_segment(link_name: str) -> str:
    """Mappe un nom de lien URDF du robot à l'un des ``SEGMENTS`` via des heuristiques de noms
    anatomiques (hip/knee/ankle/shoulder/elbow/wrist…) ; waist/torso/head/unknown par défaut à torso."""
    if link_name in ("pelvis", "pelvis_contour_link"):
        return "pelvis"
    if "head" in link_name:
        return "head"
    for side in ("left", "right"):
        seg = _side_segment(link_name, side)
        if seg is not None:
            return seg
    return "torso"


# Index d'articulation SMPL-X → segment, en utilisant la MÊME règle anatomique que les liens du
# robot (ankle → foot, wrist → forearm). Articulations du corps 0..21 explicites ; jaw/eyes 22..24 →
# head ; hands 25..54 ci-dessous.
_SMPLX_BODY: dict[int, str] = {
    0: "pelvis",
    1: "left_thigh", 2: "right_thigh",
    3: "torso", 6: "torso", 9: "torso", 12: "torso", 13: "torso", 14: "torso",
    15: "head",
    4: "left_shank", 5: "right_shank",
    7: "left_foot", 10: "left_foot",
    8: "right_foot", 11: "right_foot",
    16: "left_upperarm", 17: "right_upperarm",
    18: "left_forearm", 19: "right_forearm",
    20: "left_forearm", 21: "right_forearm",
    22: "head", 23: "head", 24: "head",
}
SMPLX_JOINT_TO_SEGMENT: dict[int, str] = dict(_SMPLX_BODY)
for _j in range(25, 40):
    SMPLX_JOINT_TO_SEGMENT[_j] = "left_hand"
for _j in range(40, 55):
    SMPLX_JOINT_TO_SEGMENT[_j] = "right_hand"


def seg_index(name: str) -> int:
    """Indice du segment ``name`` dans ``SEGMENTS``."""
    return _SEG_IDX[name]


def _vertex_segment_indices(lbs_weights: np.ndarray) -> np.ndarray:
    """(V,) indice int de segment par sommet SMPL-X, à partir de son articulation de skinning dominante."""
    dom_joint = np.asarray(lbs_weights).argmax(axis=1)               # (V,)
    return np.array([_SEG_IDX[SMPLX_JOINT_TO_SEGMENT[int(j)]] for j in dom_joint], dtype=np.int64)


def point_segments(lbs_weights: np.ndarray, faces: np.ndarray,
                   tri_idx: np.ndarray, bary: np.ndarray) -> np.ndarray:
    """(N,) indice int de segment par échantillon de surface : le segment du sommet de son triangle
    avec le poids barycentrique le plus grand (la localisation du corps auquel il est le plus proche)."""
    vseg = _vertex_segment_indices(lbs_weights)                     # (V,)
    tri_v = np.asarray(faces)[np.asarray(tri_idx)]                  # (N, 3) ids de vertex
    dom_corner = np.asarray(bary).argmax(axis=1)                    # (N,) in {0,1,2}
    dom_vertex = tri_v[np.arange(len(tri_idx)), dom_corner]
    return vseg[dom_vertex].astype(np.int64)
