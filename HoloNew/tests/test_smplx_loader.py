"""TEST-SOCP can ingest AMASS SMPL-X clips (data_format='smplx', robot_only).

The processed npz (data_utils/prep_amass_smplx_for_rt) carries 22 body joints +
world orientations; load_smplx_to_smplh_layout remaps them into the SMPLH slot
layout the GMR tables expect. Validated on the SFU 0005_2FeetJump001 clip — a real
jump with a flight phase — so the centroidal (W^c ballistic / W^L) work has a clip.
"""
from pathlib import Path
import numpy as np
import pytest
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.src.test_socp.targets import (
    load_smplx_to_smplh_layout, SMPLX_BODY_JOINT_NAMES)
from HoloNew.src.test_socp.tables import MAPPED_BODY_NAMES, HUMAN_BODY_TO_IDX

_CLIP = Path(__file__).resolve().parent.parent / "demo_data" / "SFU" / "0005_2FeetJump001.npz"


def test_smplx_remap_places_mapped_joints():
    if not _CLIP.exists():
        pytest.skip("SFU clip not present")
    raw, quat, height = load_smplx_to_smplh_layout(_CLIP, MAPPED_BODY_NAMES, HUMAN_BODY_TO_IDX)
    sidx = {n: i for i, n in enumerate(SMPLX_BODY_JOINT_NAMES)}
    d = np.load(_CLIP)
    pos = d["global_joint_positions"]
    # Every mapped body must land at its SMPLH slot with the right SMPL-X data.
    for name in MAPPED_BODY_NAMES:
        np.testing.assert_allclose(raw[:, HUMAN_BODY_TO_IDX[name]], pos[:, sidx[name]], atol=1e-6)
    # Quaternions are unit.
    used = [HUMAN_BODY_TO_IDX[n] for n in MAPPED_BODY_NAMES]
    norms = np.linalg.norm(quat[:, used], axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_smplx_robot_only_runs_and_jumps():
    if not _CLIP.exists():
        pytest.skip("SFU clip not present")
    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="0005_2FeetJump001",
        data_format="smplx", data_path=_CLIP.parent,
        retargeter=TestSocpRetargeterConfig()))
    res = rt.retarget(max_frames=120)
    assert np.all(np.isfinite(res.qpos)), "non-finite qpos on smplx jump clip"
    qz = res.qpos[:120, 2]
    # The clip is a jump: the solved pelvis must rise substantially (flight phase).
    assert (qz.max() - qz.min()) > 0.2, f"no jump detected: pelvis z range {qz.max()-qz.min():.3f}"
    assert qz.min() > 0.3 and qz.max() < 1.4, f"pelvis z out of sane range [{qz.min():.3f},{qz.max():.3f}]"
