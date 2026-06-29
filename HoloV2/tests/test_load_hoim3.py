"""HOI-M3 loader test (skips if data / SMPL-X model / deftrafo absent). Validates the EasyMocap
SMPL -> SMPL-X conversion (placement + shape transfer) and multi-object resolution."""
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from src.prepare.contracts import RobotSpec, SceneSpec
from datapaths import HOIM3, SMPLX_MODELS as _SMPLX, SMPL2SMPLX as _DEFTRAFO

_HUMAN = HOIM3 / "office_data05_human.npz"
_YUP_TO_ZUP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def _spec() -> SceneSpec:
    return SceneSpec(
        dataset="hoim3", motion_path=_HUMAN, smpl_model_dir=_SMPLX,
        robot=RobotSpec(name="g1", urdf_path=Path("g1.urdf"), link_names=("a",), dof=29, height=1.3))


@pytest.mark.skipif(not (_HUMAN.exists() and _SMPLX.is_dir() and _DEFTRAFO.exists()),
                    reason="HOI-M3 data / SMPL-X model / deftrafo absent")
def test_hoim3_conversion_and_objects():
    from src.prepare.load import load

    raw = load(_spec())
    T = raw.n_frames
    assert raw.source_format == "hoim3" and raw.is_parametric and raw.fps == 60.0
    p = raw.smpl_params
    assert p.model_type == "smplx" and p.betas.shape == (16,) and np.all(np.isfinite(p.betas))
    assert raw.joint_pos.shape == (T, 22, 3)
    assert p.body_pose.shape == (T, 63)

    # Multi-object: at least one object resolved with a (T,7) pose and an on-disk mesh.
    assert len(raw.object_poses_raw) >= 1
    assert raw.object_poses_raw[0].shape == (T, 7) and raw.object_mesh_paths[0].exists()

    # Placement parity vs the GROUND-TRUTH EasyMocap convention: EasyMocap places the body as
    # R(Rh) @ J_canonical + Th about the SMPL pelvis, so the SMPL-X demo pelvis (Z-up) must equal
    # Q @ (Th + R @ J0_smpl). J0_smpl is the SMPL rest pelvis (not SMPL-X: their heights differ
    # ~14cm). Recompute from the raw npz for the retargeted person and check a few frames.
    from src.prepare.load.smpl2smplx import smpl_rest_pelvis
    hd = np.load(str(_HUMAN), allow_pickle=True)
    sp = hd["smpl_params"]
    tid = int(np.asarray(sp[0][0]["id"]))
    smplh_npz = _SMPLX.parents[2] / "smplh" / str(hd["gender"]) / "model.npz"
    betas10 = np.asarray(sp[0][0]["shapes"], np.float64).reshape(-1)[:10]
    j0_smpl = smpl_rest_pelvis(betas10, smplh_npz)
    for t in (0, T // 2, T - 1):
        ent = next(e for e in sp[t] if int(np.asarray(e["id"])) == tid)
        Rh = np.asarray(ent["Rh"], np.float64).reshape(3)
        Th = np.asarray(ent["Th"], np.float64).reshape(3)
        expect = _YUP_TO_ZUP @ (Th + R.from_rotvec(Rh).as_matrix() @ j0_smpl)
        assert np.abs(raw.joint_pos[t, 0] - expect).max() < 1e-4, "EasyMocap->SMPL-X placement off"

    # Body is upright in Z-up: head (joint 15) clearly above the pelvis (joint 0).
    head_above = (raw.joint_pos[:, 15, 2] - raw.joint_pos[:, 0, 2])
    assert np.median(head_above) > 0.3, "body not upright (Z-up convention?)"
