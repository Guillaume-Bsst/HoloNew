"""Subject SMPL-X shape lookup for OMOMO sequences.

The OMOMO_new ``.pt`` motion files carry joint positions and quaternions but not
the SMPL-X shape (betas) or gender. Those live in the original OMOMO release, in
``data/{train,test}_diffusion_manip_seq_joints24.p``, keyed by ``seq_name`` (e.g.
``sub3_largebox_003``). Without them the human mesh falls back to the neutral mean
shape; loading them lets HumanBody pose the subject's actual body.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_human_metadata(omomo_dir: Path, seq: str) -> tuple[np.ndarray | None, str]:
    """Load betas (16,) and gender for ``seq`` from the original OMOMO .p files.

    Searches the train split first, then the test split, returning on the first
    match. Returns ``(None, "neutral")`` when the sequence (or the files) are not
    found, so callers degrade to the neutral mean shape.
    """
    import joblib

    p_files = [
        omomo_dir / "data" / "train_diffusion_manip_seq_joints24.p",
        omomo_dir / "data" / "test_diffusion_manip_seq_joints24.p",
    ]
    for p_file in p_files:
        if not p_file.exists():
            continue
        data = joblib.load(str(p_file))
        for idx in data:
            if data[idx].get("seq_name", "") == seq:
                betas = np.asarray(data[idx]["betas"]).flatten()
                gender = str(data[idx].get("gender", "neutral"))
                return betas, gender
    return None, "neutral"


def load_object_scale(omomo_dir: Path, seq: str) -> float | None:
    """Per-sequence object scale for ``seq`` from the original OMOMO .p files.

    OMOMO stores a canonical (unit) object mesh in ``captured_objects`` and a per-frame
    ``obj_scale`` that resizes it for each sequence (near-constant within a sequence, so
    the mean is returned). Returns ``None`` when the sequence (or the files) are absent,
    so callers keep the mesh at native size.
    """
    import joblib

    p_files = [
        omomo_dir / "data" / "train_diffusion_manip_seq_joints24.p",
        omomo_dir / "data" / "test_diffusion_manip_seq_joints24.p",
    ]
    for p_file in p_files:
        if not p_file.exists():
            continue
        data = joblib.load(str(p_file))
        for idx in data:
            if data[idx].get("seq_name", "") == seq and "obj_scale" in data[idx]:
                return float(np.asarray(data[idx]["obj_scale"]).mean())
    return None
