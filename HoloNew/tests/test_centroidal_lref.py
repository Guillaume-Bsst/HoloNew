"""Reference angular-momentum tracking (W^L L_ref) — Brick 4 extension.

reference_orbital_angular_momentum builds a lumped L_ref from a body trajectory;
build_lumped_L_block builds the matching current L (linearized in dqa). The two use
the same masses so the tracked residual is consistent. Validated synthetically and,
in test_centroidal_lref_jump, on a real aerial SFU clip.
"""
import numpy as np
import pytest


def test_reference_orbital_L_synthetic_rotation():
    # Two equal masses at +/-x, rotating in the xy-plane at angular speed w about z.
    # CoM is the origin, so L_z = sum m_k r^2 w.
    T, dt, w, r = 40, 0.02, 3.0, 1.0
    m = np.array([2.0, 2.0])
    th = w * dt * np.arange(T)
    p = np.zeros((T, 2, 3))
    p[:, 0, 0], p[:, 0, 1] = r * np.cos(th), r * np.sin(th)
    p[:, 1, 0], p[:, 1, 1] = -r * np.cos(th), -r * np.sin(th)
    from HoloNew.src.test_socp.centroidal import reference_orbital_angular_momentum
    L = reference_orbital_angular_momentum(p, m, dt)
    assert np.all(L[2:, 2] > 0), "L_z must be positive for ccw rotation"
    expected = m.sum() * r * r * w
    np.testing.assert_allclose(L[3:, 2].mean(), expected, rtol=0.05)
    assert np.allclose(L[2:, :2], 0, atol=1e-6), "in-plane rotation -> only L_z"


def test_lumped_L_block_matches_numpy():
    from HoloNew.examples.robot_retarget import RetargetingConfig
    from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
    from HoloNew.src.test_socp.centroidal import (
        mapped_frame_masses_and_names, build_lumped_L_block)
    from HoloNew.src.test_socp.interaction import _skew

    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    frames, masses = mapped_frame_masses_and_names(rt)
    assert len(frames) == 14 and masses.shape == (14,)
    assert masses.min() > 0

    rng = np.random.default_rng(0)
    q_pin = rt.pin.qpos_mj_to_q_pin(rt.q_init_full[:36])
    q_prev = rt.pin.qpos_mj_to_q_pin(rt.q_init_full[:36])
    L_ref_t = rng.standard_normal(3)
    dqa_val = rng.standard_normal(rt.nv_a) * 0.01
    lam, dt = 2.0, 1.0 / 30.0

    blocks = build_lumped_L_block(rt, q_pin, q_prev, frames, masses, L_ref_t, lam, dt)
    assert len(blocks) == 1
    b = blocks[0]
    block_val = float(np.sum((b.A @ dqa_val + b.c) ** 2))

    # Independent numpy ground truth (moment arms fixed at q_pin).
    M = masses.sum()
    p0 = np.array([rt.pin.body_position(q_pin, f) for f in frames])
    pprev = np.array([rt.pin.body_position(q_prev, f) for f in frames])
    Js = [rt.pin.frame_translational_jacobian(q_pin, f)[:, rt.v_a_indices] for f in frames]
    c0 = (masses[:, None] * p0).sum(0) / M
    cprev = (masses[:, None] * pprev).sum(0) / M
    Jc = sum(masses[k] * Js[k] for k in range(14)) / M
    L = np.zeros(3)
    for k in range(14):
        arm = p0[k] - c0
        vk = (p0[k] - pprev[k]) / dt + (Js[k] @ dqa_val) / dt
        cd = (c0 - cprev) / dt + (Jc @ dqa_val) / dt
        L += masses[k] * np.cross(arm, vk - cd)
    gt = lam * float(np.sum((L - L_ref_t) ** 2))
    np.testing.assert_allclose(block_val, gt, rtol=1e-9)


def test_lref_tracking_reproduces_cartwheel_spin():
    """On a real aerial clip, W^L tracking makes the solved angular momentum follow
    L_ref. Without it Style ignores the spin (corr ~ -0.03); with it the robot
    reproduces the reference momentum (corr > 0.8)."""
    from pathlib import Path
    from HoloNew.examples.robot_retarget import RetargetingConfig
    from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
    from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
    from HoloNew.src.test_socp.centroidal import (
        mapped_frame_masses_and_names, reference_orbital_angular_momentum)

    clip = Path(__file__).resolve().parent.parent / "demo_data" / "SFU" / "0007_Cartwheel001.npz"
    if not clip.exists():
        pytest.skip("cartwheel clip not present")
    N, dt = 50, 1.0 / 30.0
    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="0007_Cartwheel001",
        data_format="smplx", data_path=clip.parent,
        retargeter=TestSocpRetargeterConfig(activate_wl_track=True, lambda_l_track=5.0)))
    frames, masses = mapped_frame_masses_and_names(rt)
    Lref = reference_orbital_angular_momentum(rt.gmr_floor["pos"], masses, dt)[:N]
    res = rt.retarget(max_frames=N)
    assert np.all(np.isfinite(res.qpos))

    M = masses.sum()
    P = np.zeros((N, len(frames), 3))
    for t in range(N):
        qp = rt.pin.qpos_mj_to_q_pin(res.qpos[t, :36])
        for k, f in enumerate(frames):
            P[t, k] = rt.pin.body_position(qp, f)
    c = (masses[None, :, None] * P).sum(1) / M
    v = np.zeros_like(P); v[1:] = (P[1:] - P[:-1]) / dt
    cd = np.zeros_like(c); cd[1:] = (c[1:] - c[:-1]) / dt
    Lsol = np.zeros((N, 3))
    for k in range(len(frames)):
        Lsol += masses[k] * np.cross(P[:, k] - c, v[:, k] - cd)
    corr = (np.sum(Lsol[3:] * Lref[3:]) /
            (np.linalg.norm(Lsol[3:]) * np.linalg.norm(Lref[3:]) + 1e-9))
    assert corr > 0.8, f"L_ref tracking failed: solved-vs-ref momentum corr={corr:.2f}"
