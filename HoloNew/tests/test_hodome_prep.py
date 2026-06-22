from pathlib import Path

import numpy as np
import pytest

from HoloNew.src.data_loaders.hodome import (
    _YUP_TO_ZUP,
    HodomeMeshPoser,
    global_orientations_zup,
    prep_hodome_processed,
)


def test_yup_to_zup_is_a_proper_rotation():
    # A bare y<->z axis swap is a reflection (det -1) that mirrors the subject and
    # renders the SMPL mesh inside-out. The transform must be a proper rotation.
    Q = _YUP_TO_ZUP
    assert np.allclose(Q @ Q.T, np.eye(3), atol=1e-9)            # orthonormal
    assert np.isclose(np.linalg.det(Q), 1.0)                     # rotation, not reflection
    assert np.allclose(Q @ np.array([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0])  # Y-up -> Z-up

_REPO = Path(__file__).resolve().parents[5]
_HODOME_NPZ = _REPO / "data/00_raw_datasets/HODome/smplx/subject01_baseball.npz"
_SMPLX_DIR = _REPO / "data/00_raw_datasets/models/models_smplx_v1_1/models/smplx"


def test_zero_pose_gives_world_rotation_orientations():
    from scipy.spatial.transform import Rotation as R
    T = 3
    global_orient = np.zeros((T, 3), np.float32)
    body_pose = np.zeros((T, 63), np.float32)
    q = global_orientations_zup(global_orient, body_pose)
    assert q.shape == (T, 55, 4)
    # Zero axis-angle -> every global rotation is identity in the native Y-up frame.
    # Expressing that rest pose in the physically Y->Z rotated scene left-multiplies Q,
    # so every joint orientation is Q itself (NOT identity: the body now stands up Z-up).
    q_Q = R.from_matrix(_YUP_TO_ZUP).as_quat()[[3, 0, 1, 2]]   # xyzw -> wxyz
    assert np.allclose(q, np.tile(q_Q, (T, 55, 1)), atol=1e-6)


def test_orientations_are_unit_quaternions():
    rng = np.random.default_rng(0)
    T = 4
    global_orient = rng.normal(scale=0.3, size=(T, 3)).astype(np.float32)
    body_pose = rng.normal(scale=0.3, size=(T, 63)).astype(np.float32)
    q = global_orientations_zup(global_orient, body_pose)
    assert q.shape == (T, 55, 4)
    norms = np.linalg.norm(q, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5)


@pytest.mark.skipif(not (_HODOME_NPZ.exists() and _SMPLX_DIR.exists()),
                    reason="HODome data / SMPL-X model not present")
def test_prep_hodome_processed_real():
    out = prep_hodome_processed(_HODOME_NPZ, _SMPLX_DIR)
    pos = out["global_joint_positions"]
    ori = out["global_joint_orientations"]
    assert pos.ndim == 3 and pos.shape[1:] == (22, 3)
    assert ori.shape == (pos.shape[0], 55, 4)    # body + face + both MANO hands
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
    # Right-side-out: most face normals point away from the body centroid (a reflection
    # would invert the winding and push this below 0.5 -> inside-out render).
    import trimesh
    faces = poser.faces.astype(np.int64)
    mesh = trimesh.Trimesh(vertices=v, faces=faces, process=False)
    outward = ((mesh.face_normals * (v[faces].mean(1) - v.mean(0))).sum(1) > 0).mean()
    assert outward > 0.55


def test_global_orientations_zup_is_world_rotation_not_conjugation():
    # global_joint_positions are built as a NATIVE Y-up forward then a rigid world
    # rotation Q=_YUP_TO_ZUP applied to the points (hodome_fk: joints @ Q.T). For the
    # per-joint orientations to describe that SAME physically-rotated scene, a frame is
    # transformed by LEFT-multiplying Q (R -> Q R), NOT by conjugation (R -> Q R Q^T).
    # Conjugation rotates the joint-LOCAL axes too, so it mis-articulates every limb
    # relative to the joints/mesh (the "collapsed body" the mesh poser docstring warns
    # of). Two invariants of a rigid world rotation pin this down:
    #   * root orientation is left-multiplied by Q;
    #   * relative-to-parent (articulation) rotations are UNCHANGED.
    from scipy.spatial.transform import Rotation as R
    from HoloNew.data_utils.prep_amass_smplx_for_rt import (
        _SMPLX_BODY_PARENTS, compute_global_joint_orientations,
    )
    rng = np.random.default_rng(1)
    go = rng.normal(scale=0.4, size=(1, 3))
    bp = rng.normal(scale=0.4, size=(1, 63))
    aa = np.concatenate([go.reshape(1, 1, 3), bp.reshape(1, 21, 3)], axis=1)   # (1,22,3)
    q_yup = compute_global_joint_orientations(aa, _SMPLX_BODY_PARENTS)[0]      # (22,4) wxyz
    q_zup = global_orientations_zup(go, bp)[0]                                 # (22,4) wxyz
    Ry = R.from_quat(q_yup[:, [1, 2, 3, 0]]).as_matrix()                       # wxyz -> xyzw
    Rz = R.from_quat(q_zup[:, [1, 2, 3, 0]]).as_matrix()
    Q = _YUP_TO_ZUP
    parents = _SMPLX_BODY_PARENTS
    assert np.allclose(Rz[0], Q @ Ry[0], atol=1e-6)                            # root: left-mult
    for j in range(1, 22):                                                    # children: rel unchanged
        p = parents[j]
        rel_y = Ry[p].T @ Ry[j]
        rel_z = Rz[p].T @ Rz[j]
        assert np.allclose(rel_z, rel_y, atol=1e-6), f"joint {j} articulation changed"


@pytest.mark.skipif(not (_HODOME_NPZ.exists() and _SMPLX_DIR.exists()),
                    reason="HODome data / SMPL-X model not present")
def test_orientations_reproduce_joint_positions():
    # End-to-end consistency: re-posing the SMPL-X body from the stored
    # global_joint_orientations (exactly how the contact probe's placed_verts_smpl does:
    # global -> relative-to-parent local rotations -> SMPL-X forward) must reproduce the
    # stored global_joint_positions. Conjugated orientations mis-articulate the body, so
    # the re-posed joints drift far from the positions; the correct left-multiply matches.
    import torch
    import smplx
    from scipy.spatial.transform import Rotation as R
    from HoloNew.data_utils.prep_amass_smplx_for_rt import _SMPLX_BODY_PARENTS
    out = prep_hodome_processed(_HODOME_NPZ, _SMPLX_DIR)
    ori, pos = out["global_joint_orientations"], out["global_joint_positions"]
    betas = np.asarray(out["betas"], np.float32).reshape(1, -1)
    model = smplx.SMPLX(model_path=str(_SMPLX_DIR), gender=str(out["gender"]), ext="npz",
                        num_betas=betas.shape[-1], use_pca=False)
    parents = _SMPLX_BODY_PARENTS
    f = min(500, pos.shape[0] - 1)
    grot = R.from_quat(ori[f][:22, [1, 2, 3, 0]]).as_matrix()                 # (22,3,3) global body
    rel = np.matmul(np.transpose(grot[parents], (0, 2, 1)), grot)             # parent^T child
    rel[parents == -1] = grot[parents == -1]                                  # root keeps global
    go = torch.from_numpy(R.from_matrix(rel[0]).as_rotvec()).float().view(1, 3)
    bp = torch.from_numpy(R.from_matrix(rel[1:22]).as_rotvec()).float().view(1, -1)
    with torch.no_grad():
        o = model(global_orient=go, body_pose=bp,
                  betas=torch.from_numpy(betas), return_verts=True)
    j = o.joints[0, :22].numpy()
    j = j - j[0] + pos[f, 0]                                                   # align pelvis (drop transl)
    assert np.linalg.norm(j - pos[f], axis=1).max() < 0.02                    # <2 cm everywhere
