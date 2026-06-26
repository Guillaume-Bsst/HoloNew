"""Integration test for the HODome loader. Skips when the HODome release or the SMPL-X model
is not available locally (machine-specific data)."""
from pathlib import Path

import numpy as np
import pytest

from holov2.contracts import RobotSpec, SceneSpec

_DATA = Path("/home/vboxuser/Documents/wbt_rl/data/00_raw_datasets")
_HODOME = _DATA / "HODome"
_SMPLX = _DATA / "models" / "models_smplx_v1_1" / "models" / "smplx"


def _pick_sequence() -> Path | None:
    """A sequence present in BOTH smplx/ and object/ (so it has an object), else None."""
    smplx_dir, obj_dir = _HODOME / "smplx", _HODOME / "object"
    if not (smplx_dir.is_dir() and obj_dir.is_dir() and _SMPLX.is_dir()):
        return None
    shared = {p.stem for p in smplx_dir.glob("*.npz")} & {p.stem for p in obj_dir.glob("*.npz")}
    return smplx_dir / f"{sorted(shared)[0]}.npz" if shared else None


_SEQ = _pick_sequence()


@pytest.mark.skipif(_SEQ is None, reason="HODome data / SMPL-X model not available")
def test_hodome_load_contract():
    from holov2.prepare.load import load  # lazy: imports + registers the hodome loader

    spec = SceneSpec(
        dataset="hodome", motion_path=_SEQ,
        robot=RobotSpec(name="g1", urdf_path=Path("g1.urdf"), link_names=("a",), dof=29, height=1.3),
        smpl_model_dir=_SMPLX,
    )
    raw = load(spec)
    T = raw.n_frames

    assert raw.is_parametric and T > 0
    assert raw.source_format == "hodome"
    assert raw.joint_pos.shape == (T, 22, 3)
    assert len(raw.joint_names) == 22
    p = raw.smpl_params
    assert p.model_type == "smplx"
    assert p.body_pose.shape == (T, 63)
    assert p.left_hand_pose.shape == (T, 45) and p.right_hand_pose.shape == (T, 45)
    # one object: poses (T,7) pos-first, mesh resolved on disk
    assert len(raw.object_poses_raw) == 1 and len(raw.object_mesh_paths) == 1
    assert raw.object_poses_raw[0].shape == (T, 7)
    assert raw.object_mesh_paths[0].exists()
    # Z-up sanity: the body spans a plausible vertical (standing-ish) extent
    assert np.ptp(raw.joint_pos[:, :, 2]) > 0.3
