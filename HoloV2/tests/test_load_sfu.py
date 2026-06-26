"""SFU loader test (skips if demo data / SMPL-X model absent). Validates that the reconstructed
local SmplParams reproduce SFU's global joint positions through the BodyModel FK."""
from pathlib import Path

import numpy as np
import pytest

from holov2.contracts import RobotSpec, SceneSpec

_NPZ = Path("/home/vboxuser/Documents/wbt_rl/modules/01_retargeting/HoloNew/HoloNew/demo_data/"
            "SFU/0005_2FeetJump001.npz")
_SMPLX = Path("/home/vboxuser/Documents/wbt_rl/data/00_raw_datasets/models/models_smplx_v1_1/models/smplx")


@pytest.mark.skipif(not (_NPZ.exists() and _SMPLX.is_dir()), reason="SFU data / SMPL-X model absent")
def test_sfu_reconstruction_matches_positions():
    from holov2.prepare.load import load
    from holov2.prepare.load.smpl import build_body_model

    spec = SceneSpec(
        dataset="sfu", motion_path=_NPZ,
        robot=RobotSpec(name="g1", urdf_path=Path("g1.urdf"), link_names=("a",), dof=29, height=1.3),
        smpl_model_dir=_SMPLX,
    )
    raw = load(spec)
    T = raw.n_frames
    assert raw.source_format == "sfu" and raw.is_parametric
    assert raw.joint_pos.shape == (T, 22, 3)
    p = raw.smpl_params
    assert p.body_pose.shape == (T, 63) and p.left_hand_pose.shape == (T, 45)
    assert raw.object_poses_raw == ()

    # The reconstructed local params must reproduce SFU's global positions through the FK.
    body = build_body_model(p, _SMPLX)
    errs = []
    for t in (0, T // 2, T - 1):
        pos_fk = body.bone_transforms(p, t)[1][:22]
        errs.append(np.abs(pos_fk - raw.joint_pos[t]).max())
    assert max(errs) < 1e-2, f"reconstruction error too large: {max(errs):.4f} m (wrong quat order?)"
