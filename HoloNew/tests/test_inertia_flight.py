"""#7 — W^c governs flight: contacts place the stance, ballistic W^c places the
flight. Validated on the SFU 2FeetJump clip (data_format=smplx, inertia_mode).

Mechanism (observed): the SMPL-X contact probe detects floor contacts on the feet
during stance and NONE during flight; the floor D/X anchor the stance, and a weak
W^c governs the airborne phase. W^c is acceleration-only, so lambda_c is the lever
on the flight height: lambda_c=1e-5 overshoots (apex ~2.27 m vs ref ~1.14 m),
lambda_c=1e-3 controls it (~1.41 m). This test runs the validated lambda_c=1e-3.
"""
from pathlib import Path
import numpy as np
import pytest
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.src.test_socp.interaction import (
    robot_control_points, query_entities, frame_references, _activation)

_CLIP = Path(__file__).resolve().parent.parent / "demo_data" / "SFU" / "0005_2FeetJump001.npz"


def _floor_active(rt, qpos, t):
    L = rt.smplx_ground_probe.margin
    M = rt.correspondence.link_idx.shape[0]
    q_pin = rt.pin.qpos_mj_to_q_pin(qpos[t, :36])
    P = robot_control_points(rt, q_pin)
    _, fflr = query_entities(rt, P, None, margin=L)
    _, _, d_flr_ref, _, _ = frame_references(rt, t)
    return int(sum(1 for i in range(M)
                   if _activation(float(d_flr_ref[i]), L) > 0 and bool(fflr.active[i])))


def test_inertia_jump_contacts_and_ballistic():
    if not _CLIP.exists():
        pytest.skip("jump clip not present")
    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="0005_2FeetJump001",
        data_format="smplx", data_path=_CLIP.parent,
        retargeter=TestSocpRetargeterConfig(inertia_mode=True, lambda_c=1e-3)))
    assert rt.smplx_ground_probe is not None and rt.smplx_ground_probe.smpl_order
    res = rt.retarget(max_frames=100)
    assert np.all(np.isfinite(res.qpos)), "non-finite qpos on the jump clip"

    # Contacts: feet anchored in stance (t=5), none at the flight apex (t=82).
    stance = _floor_active(rt, res.qpos, 5)
    flight = _floor_active(rt, res.qpos, 82)
    assert stance > 100, f"feet not anchored in stance: {stance} active"
    assert flight == 0, f"unexpected floor contact at the flight apex: {flight} active"

    # The robot jumps, and W^c (lambda_c=1e-3) keeps the flight height controlled
    # (vs the ~2.27 m overshoot at the weak default lambda_c=1e-5).
    qz = res.qpos[:100, 2]
    assert (qz.max() - qz.min()) > 0.3, f"no jump: pelvis z range {qz.max()-qz.min():.3f}"
    assert qz.max() < 1.8, f"flight height uncontrolled: pelvis z max {qz.max():.3f}"
