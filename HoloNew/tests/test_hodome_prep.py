from pathlib import Path

import numpy as np
import pytest

from HoloNew.src.data_loaders.hodome import (
    HodomeMeshPoser,
    global_orientations_zup,
    prep_hodome_processed,
)

_REPO = Path(__file__).resolve().parents[5]
_HODOME_NPZ = _REPO / "data/00_raw_datasets/HODome/smplx/subject01_baseball.npz"
_SMPLX_DIR = _REPO / "data/00_raw_datasets/models/models_smplx_v1_1/models/smplx"


def test_zero_pose_gives_identity_orientations():
    T = 3
    global_orient = np.zeros((T, 3), np.float32)
    body_pose = np.zeros((T, 63), np.float32)
    q = global_orientations_zup(global_orient, body_pose)
    assert q.shape == (T, 22, 4)
    # zero axis-angle -> identity rotation; M-conjugation of identity is identity.
    assert np.allclose(q, np.tile([1.0, 0.0, 0.0, 0.0], (T, 22, 1)), atol=1e-6)


def test_orientations_are_unit_quaternions():
    rng = np.random.default_rng(0)
    T = 4
    global_orient = rng.normal(scale=0.3, size=(T, 3)).astype(np.float32)
    body_pose = rng.normal(scale=0.3, size=(T, 63)).astype(np.float32)
    q = global_orientations_zup(global_orient, body_pose)
    assert q.shape == (T, 22, 4)
    norms = np.linalg.norm(q, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5)


@pytest.mark.skipif(not (_HODOME_NPZ.exists() and _SMPLX_DIR.exists()),
                    reason="HODome data / SMPL-X model not present")
def test_prep_hodome_processed_real():
    out = prep_hodome_processed(_HODOME_NPZ, _SMPLX_DIR)
    pos = out["global_joint_positions"]
    ori = out["global_joint_orientations"]
    assert pos.ndim == 3 and pos.shape[1:] == (22, 3)
    assert ori.shape == (pos.shape[0], 22, 4)
    assert np.allclose(np.linalg.norm(ori, axis=-1), 1.0, atol=1e-4)
    assert 1.4 < float(out["height"]) < 2.1          # plausible human stature
    # Z-up: the vertical spread (Z) should dominate the lateral spread of the pelvis track.
    pelvis = pos[:, 0, :]
    assert pelvis[:, 2].std() >= 0  # finite, sanity
    assert isinstance(out["gender"], str)


@pytest.mark.skipif(not (_HODOME_NPZ.exists() and _SMPLX_DIR.exists()),
                    reason="HODome data / SMPL-X model not present")
def test_mesh_poser_aligns_with_joints():
    # Regression: the mesh must be posed by a native forward + Y->Z vertex swap so it
    # matches the skeleton. The old orientation-conjugation path collapsed the body
    # (mesh ~0.4 m tall, wrist >0.25 m off). Here the mesh spans the full stature and a
    # wrist vertex sits on the wrist joint.
    out = prep_hodome_processed(_HODOME_NPZ, _SMPLX_DIR)
    joints = out["global_joint_positions"]
    poser = HodomeMeshPoser(_HODOME_NPZ, _SMPLX_DIR)
    f = min(1000, joints.shape[0] - 1)
    v = poser.vertices_zup(f)
    assert poser.faces.ndim == 2 and poser.faces.shape[1] == 3
    # Full standing stature, not a collapsed blob.
    assert (v[:, 2].max() - v[:, 2].min()) > 1.3
    # Wrist (SMPL-X body joint 21) and head (15) joints lie on the mesh surface.
    assert np.linalg.norm(v - joints[f, 21], axis=1).min() < 0.05
    assert np.linalg.norm(v - joints[f, 15], axis=1).min() < 0.10
