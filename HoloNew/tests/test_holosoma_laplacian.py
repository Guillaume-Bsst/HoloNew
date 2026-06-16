"""W^lap port (native Holosoma interaction-mesh Laplacian deformation)."""
import numpy as np

from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter


def _rt(**kw):
    return TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh",
        retargeter=TestSocpRetargeterConfig(**kw)))


def test_laplacian_jl_matches_finite_difference():
    """J_L = kron(L, I3) @ J_V must match the FD of L @ vertices(q) (the linearization
    is exact in L since uniform weights make L adjacency-only; only FK is linearized)."""
    from HoloNew.src.test_socp.laplacian import laplacian_pieces, _mesh_frames
    rt = _rt()
    pin = rt.pin
    q = pin.qpos_mj_to_q_pin(rt.q_init_full[:36])
    dqa = np.random.RandomState(0).randn(rt.nv_a) * 1e-6
    v = np.zeros(pin.model.nv)
    v[rt.v_a_indices] = dqa
    q2 = pin.integrate(q, v)

    L, _target, verts0, J_V = laplacian_pieces(rt, q, 0)
    J_L = np.kron(L, np.eye(3)) @ J_V
    lap0 = (L @ verts0).reshape(-1)
    frames = _mesh_frames(rt)
    verts2 = np.array([pin.body_position(q2, f) for f in frames])
    lap2 = (L @ verts2).reshape(-1)
    assert np.max(np.abs((lap2 - lap0) - J_L @ dqa)) < 1e-9


def test_laplacian_runs_finite():
    q = _rt(activate_wlap=True, lambda_lap=10.0).retarget(max_frames=8).qpos
    assert np.all(np.isfinite(q)), "non-finite qpos with W^lap on"
