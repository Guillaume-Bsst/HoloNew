import os
import numpy as np
import pytest
from HoloNew.src.test_socp.correspondence.constants import SMPLX_MODEL_DIR_DEFAULT, G1_29DOF_URDF
from HoloNew.src.test_socp.correspondence.build_correspondence import (
    CorrespondenceTable, save_correspondence, load_correspondence,
)

def test_save_load_roundtrip(tmp_path):
    t = CorrespondenceTable(
        link_idx=np.array([0, 1]), offset_local=np.zeros((2, 3), np.float32),
        link_names=["pelvis", "torso_link"], human_idx=np.array([3, 4]),
        tri_idx=np.array([7, 8]), bary=np.full((2, 3), 1 / 3, np.float32),
        density=2000.0, gender="neutral", betas=np.zeros(0, np.float32),
        g1_rest_cfg=np.zeros(5), seg=np.array([0, 1]),
    )
    p = tmp_path / "c.npz"
    save_correspondence(p, t)
    r = load_correspondence(p)
    assert r.link_names == ["pelvis", "torso_link"]
    np.testing.assert_array_equal(r.human_idx, [3, 4])

needs_smplx = pytest.mark.skipif(not os.path.isdir(SMPLX_MODEL_DIR_DEFAULT),
                                 reason="SMPL-X model dir not present")

@needs_smplx
def test_build_table_neutral():
    from HoloNew.src.test_socp.correspondence.build_correspondence import build_table
    t = build_table(SMPLX_MODEL_DIR_DEFAULT, "neutral", None, G1_29DOF_URDF,
                    human_density=500.0, g1_density=300.0, reg=0.005)
    M = t.link_idx.shape[0]
    assert M > 0
    assert t.human_idx.min() >= 0
    assert t.link_idx.min() >= 0 and t.link_idx.max() < len(t.link_names)
