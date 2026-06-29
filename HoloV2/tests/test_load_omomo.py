"""OMOMO loader test (skips if data / SMPL-X model absent). Validates that the reconstructed
local SmplParams reproduce the .pt's global joint positions -- body AND both hands -- through the
BodyModel FK, and that the object (mesh + per-frame pose) resolves from the OMOMO release."""
from pathlib import Path

import numpy as np
import pytest

from src.prepare.contracts import RobotSpec, SceneSpec
from datapaths import OMOMO_NEW, OMOMO as _ROOT, SMPLX_MODELS as _SMPLX

_PT = OMOMO_NEW / "sub16_largetable_008.pt"


def _spec() -> SceneSpec:
    return SceneSpec(
        dataset="omomo", motion_path=_PT, dataset_root=_ROOT, smpl_model_dir=_SMPLX,
        robot=RobotSpec(name="g1", urdf_path=Path("g1.urdf"), link_names=("a",), dof=29, height=1.3))


@pytest.mark.skipif(not (_PT.exists() and _SMPLX.is_dir()), reason="OMOMO data / SMPL-X model absent")
def test_omomo_reconstruction_matches_body_and_hands():
    from src.prepare.load import load
    from src.prepare.load.datasets.omomo import _SMPL_2_MUJOCO, _load_pt
    from src.prepare.load.smpl import build_body_model

    raw = load(_spec())
    T = raw.n_frames
    assert raw.source_format == "omomo" and raw.is_parametric
    assert raw.joint_pos.shape == (T, 22, 3)
    p = raw.smpl_params
    assert p.body_pose.shape == (T, 63)
    assert p.left_hand_pose.shape == (T, 45) and p.right_hand_pose.shape == (T, 45)
    # OMOMO has no finger capture: the fingers inherit the wrist orientation, so the reconstructed
    # (parent-relative) hand pose is rigid (~0). This is DERIVED from the .pt, not assumed -- the
    # hand FK parity below then confirms the flat SMPL-X hand matches the .pt's rigid hand joints.
    assert np.abs(p.left_hand_pose).max() < 1e-5 and np.abs(p.right_hand_pose).max() < 1e-5

    # Object resolved: one (T,7) pose + a mesh path on disk.
    assert len(raw.object_poses_raw) == 1 and raw.object_poses_raw[0].shape == (T, 7)
    assert raw.object_mesh_paths[0].exists()

    # FK parity: the reconstructed locals must reproduce the .pt joints in the SMPL-X layout --
    # body 0-21, left hand 25-39 (== SMPL-H 22-36), right hand 40-54 (== SMPL-H 37-51).
    joints_mj, _, _ = _load_pt(_PT)
    pos52 = np.empty_like(joints_mj); pos52[:, _SMPL_2_MUJOCO] = joints_mj   # SMPL-H 52 order
    body = build_body_model(p, _SMPLX)
    body_err, lh_err, rh_err = [], [], []
    for t in (0, T // 2, T - 1):
        fk = body.bone_transforms(p, t)[1]                                  # (55,3) Z-up
        body_err.append(np.abs(fk[:22] - pos52[t, :22]).max())
        lh_err.append(np.abs(fk[25:40] - pos52[t, 22:37]).max())
        rh_err.append(np.abs(fk[40:55] - pos52[t, 37:52]).max())
    assert max(body_err) < 1e-3, f"body reconstruction off by {max(body_err):.4f} m"
    assert max(lh_err) < 1e-3 and max(rh_err) < 1e-3, "hand reconstruction off (MANO order?)"
    # The demo joints are exactly the SMPL-order body positions.
    assert np.abs(raw.joint_pos - pos52[:, :22]).max() < 1e-5
