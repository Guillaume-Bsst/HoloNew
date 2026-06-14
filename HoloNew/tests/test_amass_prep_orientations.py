"""Validate the global-orientation FK added to prep_amass_smplx_for_rt.

A serial chain rotating about z by per-joint angles must compose to cumulative
angles down the tree: global angle[k] = sum of local angles from root to k.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from HoloNew.data_utils.prep_amass_smplx_for_rt import (
    compute_global_joint_orientations, _SMPLX_BODY_PARENTS,
)


def test_serial_chain_composes_angles():
    # 4-joint serial chain: parents 0<-1<-2<-3, each a local z-rotation.
    parents = np.array([-1, 0, 1, 2], dtype=np.int64)
    angles = np.array([0.1, 0.2, -0.3, 0.4])          # local z angles
    aa = np.zeros((1, 4, 3))
    aa[0, :, 2] = angles                              # axis-angle about z
    q = compute_global_joint_orientations(aa, parents)  # (1, 4, 4) wxyz
    # Recover each joint's global z-angle from the quaternion.
    cum = np.cumsum(angles)                           # expected global angles
    for k in range(4):
        wxyz = q[0, k]
        R = Rotation.from_quat(wxyz[[1, 2, 3, 0]]).as_matrix()
        ang = np.arctan2(R[1, 0], R[0, 0])           # z-rotation angle
        assert abs(ang - cum[k]) < 1e-6, f"joint {k}: {ang} != {cum[k]}"


def test_branching_tree_matches_manual_compose():
    # SMPL-X body parents: verify a couple of branch joints compose correctly.
    parents = _SMPLX_BODY_PARENTS
    rng = np.random.default_rng(0)
    aa = rng.standard_normal((2, 22, 3)) * 0.3
    q = compute_global_joint_orientations(aa, parents)
    local_R = Rotation.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(2, 22, 3, 3)
    # left_shoulder (16): parent chain 16<-13(left_collar)<-9<-6<-3<-0.
    for t in range(2):
        chain = [0, 3, 6, 9, 13, 16]
        Rg = np.eye(3)
        for j in chain:
            Rg = Rg @ local_R[t, j]
        got = Rotation.from_quat(q[t, 16][[1, 2, 3, 0]]).as_matrix()
        assert np.allclose(got, Rg, atol=1e-9), f"frame {t}: shoulder global mismatch"


def test_output_is_unit_wxyz():
    parents = _SMPLX_BODY_PARENTS
    rng = np.random.default_rng(1)
    aa = rng.standard_normal((3, 22, 3)) * 0.5
    q = compute_global_joint_orientations(aa, parents)
    assert q.shape == (3, 22, 4)
    norms = np.linalg.norm(q, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-9), "quaternions must be unit"
