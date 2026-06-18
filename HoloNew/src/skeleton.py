# src/skeleton.py
"""52-joint SMPLH skeleton topology and colours for the stage viewer.

Indices are positions into SMPLH_DEMO_JOINTS (intermimic MuJoCo order), so the
same bone/joint tables drive any (T, 52, 3) source-skeleton frame.
"""
from __future__ import annotations

import numpy as np

# Bones connecting the 18 body joints (+ the 4 right-arm joints at 33-36).
BODY_BONES: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (12, 13),
    (11, 14), (14, 15), (15, 16), (16, 17),
    (11, 33), (33, 34), (34, 35), (35, 36),
]

# SMPL-X 22-joint body topology (pelvis-rooted kinematic tree, in
# SMPLX_BODY_JOINT_NAMES order). The smplx data path's source skeleton carries
# only these 22 body joints (no fingers), so its "Original" stage is drawn with
# this bone list instead of the 52-joint SMPLH BODY_BONES above. Each pair is
# (parent, child); the root pelvis (0) contributes no bone.
SMPLX_BODY_BONES: list[tuple[int, int]] = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8),
    (6, 9), (7, 10), (8, 11), (9, 12), (9, 13), (9, 14), (12, 15),
    (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21),
]

# Finger bones: wrist -> 5 finger roots, then each finger's two distal links,
# for both hands (left wrist = 17, right wrist = 36).
FINGER_BONES: list[tuple[int, int]] = (
    [(17, 18), (17, 21), (17, 24), (17, 27), (17, 30)]
    + [(18, 19), (19, 20), (21, 22), (22, 23), (24, 25), (25, 26),
       (27, 28), (28, 29), (30, 31), (31, 32)]
    + [(36, 37), (36, 40), (36, 43), (36, 46), (36, 49)]
    + [(37, 38), (38, 39), (40, 41), (41, 42), (43, 44), (44, 45),
       (46, 47), (47, 48), (49, 50), (50, 51)]
)

BODY_JOINT_INDICES: list[int] = list(range(18)) + [33, 34, 35, 36]
FINGER_JOINT_INDICES: list[int] = list(range(18, 33)) + list(range(37, 52))

COLOR_BODY = np.array([70, 130, 220], dtype=np.uint8)
COLOR_FINGER = np.array([120, 170, 230], dtype=np.uint8)
COLOR_GHOST_BODY = np.array([150, 180, 220], dtype=np.uint8)
COLOR_GHOST_FINGER = np.array([185, 205, 235], dtype=np.uint8)
COLOR_STAGE = np.array([230, 140, 30], dtype=np.uint8)
COLOR_GHOST_STAGE = np.array([240, 200, 150], dtype=np.uint8)

# Child -> parent in the 52-joint SMPLH tree, read off the bone lists (each bone
# is a (parent, child) pair). Used to connect an arbitrary joint subset into a
# skeleton: a mapped stage keeps only some joints, so each is linked to its
# nearest *present* ancestor.
_PARENT: dict[int, int] = {child: parent for parent, child in BODY_BONES + FINGER_BONES}


def bones_for_subset(indices: list[int]) -> list[tuple[int, int]]:
    """Bone pairs (as positions into ``indices``) connecting each selected joint
    to its nearest present ancestor in the 52-joint SMPLH tree.

    ``indices`` are 52-joint indices in the stage's own joint order; the returned
    pairs index into that stage array, so a mapped/preprocessing stage renders as
    a skeleton without carrying any topology of its own. The root (no ancestor in
    the subset, e.g. the pelvis) contributes no bone.
    """
    pos = {idx: i for i, idx in enumerate(indices)}
    bones: list[tuple[int, int]] = []
    for idx in indices:
        ancestor = _PARENT.get(idx)
        while ancestor is not None and ancestor not in pos:
            ancestor = _PARENT.get(ancestor)
        if ancestor is not None:
            bones.append((pos[ancestor], pos[idx]))
    return bones
