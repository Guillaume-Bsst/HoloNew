"""Parity: HoloNew's GMR-SOCP vs test_pipe's GMR (mink velocity IK).

The two solve the same GMR body-tracking problem with different optimizers
(HoloNew = conic SOCP reusing holosoma's framework; test_pipe = mink differential
IK), so the results are NOT identical — this test QUANTIFIES the difference rather
than asserting equality. It runs HoloNew GMR-SOCP v1 live and compares it to a
frozen mink reference.

Observed on sub3_largebox_003 (for reference, the asserts below are looser):
    base-pos RMSE ~0.40 m   base-quat mean|dot| ~0.995   joints mean|diff| ~0.19 rad
The base-position gap mostly reflects different pre-IK grounding/scaling between
the two pipelines; orientations track closely.

Reference: tests/golden/gmr_mink_qpos.npz -> qpos from test_pipe's
`compute_gmr_stage`, run in the `tpretargeting` conda env. Regenerate:
    <tpretargeting-python> -c "
    import numpy as np
    from test_pipe_retargeting.human.motion import load_pt
    from test_pipe_retargeting.solver.gmr.preprocess import compute_stages
    from test_pipe_retargeting.solver.gmr.stage import compute_gmr_stage
    from test_pipe_retargeting.constants import G1_29DOF_MJCF
    j,_,q = load_pt('<sub3.pt>'); s = compute_stages(j,q)
    g = compute_gmr_stage(s['ground'], mjcf_path=str(G1_29DOF_MJCF))
    np.savez('<HoloNew>/tests/golden/gmr_mink_qpos.npz', qpos=np.asarray(g['qpos']))"
"""
from pathlib import Path

import numpy as np
import pytest

_MINK = Path(__file__).parent / "golden" / "gmr_mink_qpos.npz"


@pytest.mark.skipif(not _MINK.exists(), reason="mink GMR reference not present")
def test_gmr_socp_is_close_to_mink():
    from HoloNew.examples.robot_retarget import RetargetingConfig
    from HoloNew.src.gmr_socp.gmr_socp_v1 import GmrSocpRetargeterV1

    mink = np.load(_MINK)["qpos"]
    rt = GmrSocpRetargeterV1.from_config(
        RetargetingConfig(task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    holonew = rt.retarget().qpos

    T = min(len(mink), len(holonew))
    mink, holonew = mink[:T], holonew[:T]
    assert mink.shape == holonew.shape, (mink.shape, holonew.shape)
    assert np.isfinite(holonew).all()

    base_pos_rmse = float(np.sqrt(np.mean(np.sum((mink[:, :3] - holonew[:, :3]) ** 2, axis=1))))
    q1 = mink[:, 3:7] / np.linalg.norm(mink[:, 3:7], axis=1, keepdims=True)
    q2 = holonew[:, 3:7] / np.linalg.norm(holonew[:, 3:7], axis=1, keepdims=True)
    base_quat_dot = float(np.mean(np.abs(np.sum(q1 * q2, axis=1))))
    joints_mean = float(np.mean(np.abs(mink[:, 7:] - holonew[:, 7:])))
    print(f"\nGMR-SOCP vs mink: base-pos RMSE={base_pos_rmse:.3f} m | "
          f"base-quat mean|dot|={base_quat_dot:.3f} | joints mean|diff|={joints_mean:.3f} rad")

    # Loose sanity bounds: same solution family, not bit-identical.
    assert base_quat_dot > 0.95, base_quat_dot          # orientations broadly aligned
    assert base_pos_rmse < 1.0, base_pos_rmse           # base within ~1 m
    assert joints_mean < 0.6, joints_mean               # mean joint diff < ~34 deg
