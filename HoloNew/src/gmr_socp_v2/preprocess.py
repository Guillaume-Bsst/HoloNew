"""GMR pre-IK pipeline reimplemented as pure functions.

Ported verbatim from test_pipe's solver/gmr/preprocess.py (credit: General Motion
Retargeting, YanjieZe/GMR — only the configuration tables are sourced from GMR).

human_data is a dict: body_name -> (pos (3,) float, quat_wxyz (4,) float).
Operations (scale / offset / ground) are trivial and implemented independently;
only the configuration tables in tables.py are sourced from GMR.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from .tables import (
    GROUND_HEIGHT,
    HUMAN_BODY_TO_IDX,
    HUMAN_HEIGHT_ASSUMPTION,
    HUMAN_ROOT_NAME,
    HUMAN_SCALE_TABLE,
    IK_MATCH_TABLE1,
    MAPPED_BODY_NAMES,
)

HumanData = dict[str, tuple[np.ndarray, np.ndarray]]  # {body: (pos, quat_wxyz)}


def scale(human_data: HumanData, ratio: float) -> HumanData:
    """Scale each body in pelvis-local frame by HUMAN_SCALE_TABLE[body] * ratio."""
    root_pos, root_quat = human_data[HUMAN_ROOT_NAME]
    scaled_root = HUMAN_SCALE_TABLE[HUMAN_ROOT_NAME] * ratio * root_pos
    out: HumanData = {HUMAN_ROOT_NAME: (scaled_root, root_quat)}
    for name, (pos, quat) in human_data.items():
        if name == HUMAN_ROOT_NAME:
            continue
        s = HUMAN_SCALE_TABLE[name] * ratio
        out[name] = ((pos - root_pos) * s + scaled_root, quat)
    return out


def _offset_lookup() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """human_body -> (pos_offset (3,), rot_offset_wxyz (4,)) from IK_MATCH_TABLE1."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for _frame, (body, _pw, _rw, pos_off, rot_off) in IK_MATCH_TABLE1.items():
        out[body] = (np.asarray(pos_off, dtype=float), np.asarray(rot_off, dtype=float))
    return out


def offset(human_data: HumanData) -> HumanData:
    """Apply per-body rotation offset (quat compose) then position offset
    (rotated into the updated body frame). Mirrors GMR.offset_human_data."""
    table = _offset_lookup()
    ground = GROUND_HEIGHT * np.array([0.0, 0.0, 1.0])
    out: HumanData = {}
    for name, (pos, quat) in human_data.items():
        pos_off, rot_off = table[name]
        rot_off_R = R.from_quat(rot_off, scalar_first=True)
        updated = R.from_quat(quat, scalar_first=True) * rot_off_R
        updated_quat = updated.as_quat(scalar_first=True)
        global_off = updated.apply(pos_off - ground)
        out[name] = (pos + global_off, updated_quat)
    return out


def apply_ground(human_data: HumanData, ground_offset: float = 0.0) -> HumanData:
    """Mirror GMR.apply_ground_offset: subtract a fixed z offset (default 0 => no-op)."""
    shift = np.array([0.0, 0.0, ground_offset])
    return {n: (p - shift, q) for n, (p, q) in human_data.items()}


def build_human_data(positions: np.ndarray, quats_wxyz: np.ndarray) -> HumanData:
    """positions (52,3), quats_wxyz (52,4) -> {body: (pos, quat_wxyz)} for mapped bodies."""
    return {
        name: (positions[idx].astype(float), quats_wxyz[idx].astype(float))
        for name, idx in HUMAN_BODY_TO_IDX.items()
    }


def compute_stages(
    positions: np.ndarray,
    quats_wxyz: np.ndarray,
    human_height: float = HUMAN_HEIGHT_ASSUMPTION,
    anchor_root_xy: bool = True,
) -> dict[str, dict[str, np.ndarray]]:
    """positions (T,52,3), quats_wxyz (T,52,4) ->
    {stage: {'pos': (T,B,3), 'quat': (T,B,4)}} for stages mapped/scaled/offset/ground.
    Mirrors GMR.update_targets order: scale -> offset -> apply_ground_offset.

    The 'ground' stage applies GMR's offline floor correction: a single ground offset
    (the lowest mapped-body z over the WHOLE sequence, on the offset-stage targets) is
    subtracted from every frame so the sequence's lowest body rests on the floor.

    When `anchor_root_xy` is True, each stage is translated in XY so its pelvis
    coincides with the raw pelvis (GMR scales the absolute root toward the origin; this
    restores the raw root XY while keeping the Z floor-drop). Set False for the literal
    GMR-scaled positions.
    """
    T = positions.shape[0]
    ratio = human_height / HUMAN_HEIGHT_ASSUMPTION
    names = MAPPED_BODY_NAMES
    B = len(names)
    stage_names = ("mapped", "scaled", "offset", "ground")
    out = {
        s: {"pos": np.empty((T, B, 3), np.float32), "quat": np.empty((T, B, 4), np.float32)}
        for s in stage_names
    }

    mapped_src_indices = [HUMAN_BODY_TO_IDX[name] for name in names]
    pos_mapped = positions[:, mapped_src_indices, :].astype(np.float32)    # (T, B, 3)
    quats_mapped = quats_wxyz[:, mapped_src_indices, :].astype(np.float32)  # (T, B, 4)

    for t in range(T):
        hd = {name: (pos_mapped[t, i], quats_mapped[t, i]) for i, name in enumerate(names)}
        sd = scale(hd, ratio)
        od = offset(sd)
        for stage, data in (("mapped", hd), ("scaled", sd), ("offset", od)):
            for bi, n in enumerate(names):
                p, q = data[n]
                out[stage]["pos"][t, bi] = p
                out[stage]["quat"][t, bi] = q

    ground_offset = float(out["offset"]["pos"][:, :, 2].min())
    out["ground"]["pos"][:] = out["offset"]["pos"]
    out["ground"]["pos"][:, :, 2] -= ground_offset
    out["ground"]["quat"][:] = out["offset"]["quat"]

    if anchor_root_xy:
        pelvis_bi = names.index(HUMAN_ROOT_NAME)
        raw_pelvis_xy = positions[:, HUMAN_BODY_TO_IDX[HUMAN_ROOT_NAME], :2]  # (T, 2)
        for s in stage_names:
            stage_pelvis_xy = out[s]["pos"][:, pelvis_bi, :2]                # (T, 2)
            shift = (raw_pelvis_xy - stage_pelvis_xy)[:, None, :]            # (T, 1, 2)
            out[s]["pos"][:, :, :2] += shift
    return out
