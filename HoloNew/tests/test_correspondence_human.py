import os
import numpy as np
import pytest
from HoloNew.src.correspondence.constants import SMPLX_MODEL_DIR_DEFAULT

needs_smplx = pytest.mark.skipif(not os.path.isdir(SMPLX_MODEL_DIR_DEFAULT),
                                 reason="SMPL-X model dir not present")

@needs_smplx
def test_human_source_builds():
    from HoloNew.src.correspondence.human_body import HumanBody
    from HoloNew.src.correspondence.human_source import build_human_source
    body = HumanBody(SMPLX_MODEL_DIR_DEFAULT, None, "neutral")
    src = build_human_source(body, density=500.0)   # low density = fast
    N = src.points.shape[0]
    assert N > 0 and src.points.shape == (N, 3)
    assert src.seg.shape == (N,) and src.seg.min() >= 0 and src.seg.max() <= 14
    assert src.tri_idx.shape == (N,) and src.bary.shape == (N, 3)
