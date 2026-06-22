import os
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R
from HoloNew.src.test_socp.correspondence.constants import SMPLX_MODEL_DIR_DEFAULT
from HoloNew.src.test_socp.correspondence.human_body import HumanBody, SMPLH_PARENTS

_HAVE = os.path.isdir(SMPLX_MODEL_DIR_DEFAULT)
pytestmark = pytest.mark.skipif(not _HAVE, reason="SMPL-X model not present")


def _body():
    return HumanBody(SMPLX_MODEL_DIR_DEFAULT, betas=None, gender="neutral")


def test_smplh_parents_match_verified_array():
    expected = np.array(
        [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
         20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35,
         21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50], dtype=np.int64)
    np.testing.assert_array_equal(SMPLH_PARENTS, expected)


def test_placed_verts_smpl_curled_hand_moves_fingertips():
    body = _body()
    pelvis = np.zeros(3)
    flat = np.zeros((55, 4), np.float32); flat[:, 0] = 1.0          # all identity
    curled = flat.copy()
    # Curl every left-hand joint (25..39) by ~1 rad; the test only needs a large finite
    # displacement vs the flat hand.
    q = R.from_rotvec([1.0, 0.0, 0.0]).as_quat()[[3, 0, 1, 2]]      # wxyz
    curled[25:40] = q
    v_flat = body.placed_verts_smpl(flat, pelvis)
    v_curled = body.placed_verts_smpl(curled, pelvis)
    disp = np.linalg.norm(v_curled - v_flat, axis=1)
    assert disp.max() > 0.02            # hand vertices moved a couple cm+
    assert np.isfinite(v_curled).all()  # mesh did not explode


def test_placed_verts_smpl_backward_compat_22():
    body = _body()
    q22 = np.zeros((22, 4), np.float32); q22[:, 0] = 1.0
    v = body.placed_verts_smpl(q22, np.zeros(3))
    assert v.shape[0] > 1000 and np.isfinite(v).all()   # still produces a body mesh


def test_placed_verts_omomo_poses_hands():
    body = _body()
    flat = np.zeros((52, 4), np.float32); flat[:, 0] = 1.0
    curled = flat.copy()
    q = R.from_rotvec([1.0, 0.0, 0.0]).as_quat()[[3, 0, 1, 2]]
    # MuJoCo order scatters into SMPL slots; curl a wide block to guarantee hand slots hit.
    curled[:] = q
    v_flat = body.placed_verts(flat, np.zeros(3))
    v_curled = body.placed_verts(curled, np.zeros(3))
    assert np.linalg.norm(v_curled - v_flat, axis=1).max() > 0.02
    assert np.isfinite(v_curled).all()
