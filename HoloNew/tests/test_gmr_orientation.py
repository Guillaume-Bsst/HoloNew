"""Empirical convergence test for GMR-SOCP v1 orientation tracking (Task 4).

Verifies that a pure orientation target causes the angular error to reduce
by at least 50% over 15 solver iterations.  The specific rotation convention
in the objective is validated empirically by this test — do NOT lower the
threshold.
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def test_orientation_error_decreases():
    from HoloNew.config_types.retargeting import RetargetingConfig
    from HoloNew.src.gmr_socp.gmr_socp_v1 import GmrSocpRetargeterV1

    rt = GmrSocpRetargeterV1.from_config(
        RetargetingConfig(task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh")
    )

    frame = "left_elbow_link"
    body = rt.robot_link_names[frame]
    q = np.copy(rt.q_init_full)

    R_cur0 = rt.body_rotation(q, body)
    R_tgt = R_cur0 @ Rotation.from_euler("z", 30, degrees=True).as_matrix()
    p_tgt = rt.body_position(q, body)

    # Orientation-only target (w_p=0, w_r=10)
    targets = {frame: (p_tgt, R_tgt, 0.0, 10.0)}

    errs = []
    for _ in range(15):
        q, _ = rt.solve_single_iteration(q, q[rt.q_a_indices], q, targets)
        R_cur = rt.body_rotation(q, body)
        errs.append(np.linalg.norm(Rotation.from_matrix(R_cur.T @ R_tgt).as_rotvec()))

    print(f"Orientation errors: {errs}")
    assert errs[-1] < errs[0] * 0.5, (
        f"Expected error to halve over 15 iters, got: {errs}"
    )
