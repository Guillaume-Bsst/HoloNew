"""build_style : formes + linéarisation A·δv + c vs FD du résidu réel (pos linéaire ; rot = so3_log)."""
import numpy as np

from src.solve.config import SolveConfig
from src.solve.terms.style import build_style
from src.solve.terms._ops import quat_to_rot, so3_log
from src.targets.contracts import StyleEval, StyleTargets


def _rand_rot(rng):
    a = rng.standard_normal(3); a /= np.linalg.norm(a); th = rng.uniform(0.2, 2.0)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def _rot_to_quat(R):
    w = np.sqrt(max(0.0, 1 + np.trace(R))) / 2
    x = (R[2, 1] - R[1, 2]) / (4 * w); y = (R[0, 2] - R[2, 0]) / (4 * w); z = (R[1, 0] - R[0, 1]) / (4 * w)
    return np.array([w, x, y, z])


def _make(rng, L=3, nv=9, with_rot=True):
    pos = rng.standard_normal((L, 3)); rot = np.stack([_rand_rot(rng) for _ in range(L)])
    jp = rng.standard_normal((L, 3, nv)); jr = rng.standard_normal((L, 3, nv))
    names = tuple(f"link{i}" for i in range(L))
    ev = StyleEval(position=pos, rotation=rot, jac_pos=jp, jac_rot=jr, link_names=names)
    tgt_pos = rng.standard_normal((L, 3))
    ori = np.stack([_rot_to_quat(_rand_rot(rng)) for _ in range(L)]) if with_rot else None
    tgt = StyleTargets(link_names=names, position=tgt_pos, orientation=ori)
    return ev, tgt


def test_spos_shapes_and_linear():
    rng = np.random.default_rng(0); ev, tgt = _make(rng)
    cfg = SolveConfig()
    blocks = {b.name: b for b in build_style(ev, tgt, cfg)}
    sp = blocks["S-pos"]
    L, nv = 3, 9
    assert sp.A.shape == (L * 3, nv) and sp.c.shape == (L * 3,) and sp.A_obj is None
    # residual r(v) = w·((pos + jp·v) − tgt) is linear -> A·v + c == r(v) exactly.
    v = rng.standard_normal(nv)
    r = cfg.w_pos * ((ev.position + np.einsum("lij,j->li", ev.jac_pos, v)) - tgt.position).reshape(-1)
    assert np.allclose(sp.A @ v + sp.c, r)
    assert np.allclose(sp.c, (cfg.w_pos * (ev.position - tgt.position)).reshape(-1))


def test_srot_linearization_vs_fd():
    rng = np.random.default_rng(1); ev, tgt = _make(rng)
    cfg = SolveConfig()
    sr = {b.name: b for b in build_style(ev, tgt, cfg)}["S-rot"]
    L, nv = 3, 9
    assert sr.A.shape == (L * 3, nv) and sr.c.shape == (L * 3,)
    R_ref = quat_to_rot(tgt.orientation)
    assert np.allclose(sr.c, (cfg.w_rot * so3_log(R_ref, ev.rotation)).reshape(-1))
    # FD: perturb world orientation by jac_rot·v (R_cur(v) = exp([jac_rot·v]x) R_cur). Check A·v matches
    # the first-order change of so3_log -> validates A = w_rot·jac_rot (the world-frame convention).
    v = rng.standard_normal(nv) * 1e-5
    R_pert = np.empty_like(ev.rotation)
    for l in range(L):
        wl = ev.jac_rot[l] @ v
        K = np.array([[0, -wl[2], wl[1]], [wl[2], 0, -wl[0]], [-wl[1], wl[0], 0]])
        R_pert[l] = (np.eye(3) + K) @ ev.rotation[l]
    c_pert = (cfg.w_rot * so3_log(R_ref, R_pert)).reshape(-1)
    assert np.allclose(sr.A @ v, c_pert - sr.c, atol=1e-6)


def test_position_only_skips_srot():
    rng = np.random.default_rng(2); ev, tgt = _make(rng, with_rot=False)
    names = [b.name for b in build_style(ev, tgt, SolveConfig())]
    assert names == ["S-pos"]
