"""build_contact : C-D linéarisation vs FD de la distance ; C-X assemblage vs geo_value_grad + FD
(champ géodésique synthétique LINÉAIRE -> gradient exact). FLAG repris : C-X mappe le gradient
object-local -> monde via world_normal(R_i, .) ; même `value` pour c et le gradient (signe cohérent)."""
import numpy as np

from src.solve.config import SolveConfig
from src.solve.terms.contact import build_contact
from src.solve.terms._ops import GeoField, dist_jac, world_normal
from src.targets.contracts import (ContactEval, MultiChannelField, RobotInteractionTargets)
from src.prepare.contracts import GeodesicTable
from src.targets.interaction.geodesic import geo_value_grad, nearest_index


def _mcf(C, M, rng, active):
    return MultiChannelField(
        distance=rng.standard_normal((C, M)),
        direction=rng.standard_normal((C, M, 3)),
        witness=rng.standard_normal((C, M, 3)),
        active=active, channels=tuple(["ground"] + [f"obj{i}" for i in range(C - 1)]))


def _eval(C, M, nv, rng, active):
    return ContactEval(
        field=_mcf(C, M, rng, active),
        point_jac=rng.standard_normal((M, 3, nv)),
        probe_jac_obj=rng.standard_normal((C, M, 3, 6)),
        env=())


def test_cd_linearization_vs_distance_fd():
    rng = np.random.default_rng(0)
    C, M, nv = 1, 4, 9                              # ground channel only (R_i = I), n_obj=0
    active = np.ones((C, M), bool)
    ev = _eval(C, M, nv, rng, active)
    ref = RobotInteractionTargets(field=_mcf(C, M, rng, active))
    geo = GeoField(tables=(None,), rot=np.eye(3)[None], pos=np.zeros((1, 3)), object_idx=(-1,))
    cfg = SolveConfig(contact_d_ref_scale=0.0)     # disable falloff -> weight = w_cd on active rows
    cd = {b.name: b for b in build_contact(ev, ref, geo, cfg)}["C-D"]
    assert cd.A.shape == (M, nv) and cd.c.shape == (M,) and cd.A_obj is None
    # per-row: c = w·(d_cur − d_ref) ; A·v = w·(g_world·point_jac)·v  (linear distance model)
    # gradient de distance signée = sign(d)·direction (pas la direction brute — voir test de pénétration)
    g_world = world_normal(np.eye(3), np.sign(ev.field.distance[0])[:, None] * ev.field.direction[0])
    assert np.allclose(cd.c, cfg.w_cd * (ev.field.distance[0] - ref.field.distance[0]))
    v = rng.standard_normal(nv)
    assert np.allclose(cd.A @ v, cfg.w_cd * (dist_jac(g_world, ev.point_jac) @ v))


def test_cd_jacobian_uses_signed_sdf_gradient_for_penetration():
    """Régression : pour un point PÉNÉTRANT (distance < 0), le Jacobien C-D doit utiliser le VRAI
    gradient de distance signée ``sign(distance)·direction`` (pointe HORS de la surface -> restaurateur),
    PAS le vecteur géométrique brut surface->point (``direction``) qui s'inverse sous la surface et
    enfoncerait le contact plus profond. Porté de V1 ``interaction.py`` : ``g = sign(d0)·n0``."""
    rng = np.random.default_rng(7)
    C, M, nv = 1, 3, 9                                   # sol seul (R_i = I), n_obj = 0
    active = np.ones((C, M), bool)
    field = MultiChannelField(
        distance=np.array([[-0.02, -0.05, 0.03]]),       # deux points sous le sol, un au-dessus
        # direction = surface->point : pointe vers le BAS (−z) là où le point pénètre (comme le vrai SDF)
        direction=np.array([[[0, 0, -1.0], [0, 0, -1.0], [0, 0, 1.0]]]),
        witness=np.zeros((C, M, 3)), active=active, channels=("ground",))
    point_jac = np.tile(np.eye(3, nv), (M, 1, 1))        # (M,3,nv) : colonnes 0..2 = xyz monde du point
    ev = ContactEval(field=field, point_jac=point_jac,
                     probe_jac_obj=np.zeros((C, M, 3, 6)), env=())
    ref = RobotInteractionTargets(field=_mcf(C, M, rng, active))
    geo = GeoField(tables=(None,), rot=np.eye(3)[None], pos=np.zeros((1, 3)), object_idx=(-1,))
    cfg = SolveConfig(contact_d_ref_scale=0.0)
    cd = {b.name: b for b in build_contact(ev, ref, geo, cfg)}["C-D"]
    grad = np.sign(field.distance[0])[:, None] * field.direction[0]   # vrai gradient == +z partout
    assert np.allclose(cd.A, cfg.w_cd * dist_jac(grad, point_jac))
    # ∂distance/∂(point_z) doit être POSITIF pour CHAQUE point (monter => distance augmente => restaurateur)
    assert np.all(cd.A[:, 2] > 0)


def test_cd_object_channel_couples_dxi():
    rng = np.random.default_rng(1)
    C, M, nv = 2, 3, 9                              # ground + 1 object -> n_obj = 1
    active = np.zeros((C, M), bool); active[1, :] = True   # only object-channel pairs active
    ev = _eval(C, M, nv, rng, active)
    ref = RobotInteractionTargets(field=_mcf(C, M, rng, active))
    R1 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])        # object world rotation (90° z)
    geo = GeoField(tables=(None, None), rot=np.stack([np.eye(3), R1]),
                   pos=np.zeros((2, 3)), object_idx=(-1, 0))
    cfg = SolveConfig(contact_d_ref_scale=0.0)
    cd = {b.name: b for b in build_contact(ev, ref, geo, cfg)}["C-D"]
    assert cd.A_obj is not None and cd.A_obj.shape == (M, 6)
    # gradient SDF local signé du canal 1 (sign(d)·direction) — pas la direction brute
    g1 = np.sign(ev.field.distance[1])[:, None] * ev.field.direction[1]
    # object side uses the LOCAL signed gradient against probe_jac_obj of channel 1
    expect = cfg.w_cd * dist_jac(g1, ev.probe_jac_obj[1])
    assert np.allclose(cd.A_obj, expect)
    # robot side uses the WORLD-mapped gradient world_normal(R1, g1)
    expect_A = cfg.w_cd * dist_jac(world_normal(R1, g1), ev.point_jac)
    assert np.allclose(cd.A, expect_A)


def test_cx_assembly_and_fd_on_linear_field():
    rng = np.random.default_rng(2)
    C, M, nv = 1, 2, 9
    active = np.ones((C, M), bool)
    ev = _eval(C, M, nv, rng, active)
    ref = RobotInteractionTargets(field=_mcf(C, M, rng, active))
    # synthetic LINEAR geodesic field f(p) = a·p over a small point set -> geo_value_grad exact.
    P = 30
    pts = rng.standard_normal((P, 3)).astype(np.float32)
    a = np.array([0.7, -0.3, 0.4])
    geo_rows = (pts.astype(np.float64) @ a).astype(np.float32)
    table = GeodesicTable(points=pts, normals=np.tile([0, 0, 1.0], (P, 1)).astype(np.float32),
                          geo=np.tile(geo_rows, (P, 1)), name="ground")
    geo = GeoField(tables=(table,), rot=np.eye(3)[None], pos=np.zeros((1, 3)), object_idx=(-1,))
    cfg = SolveConfig(contact_d_ref_scale=0.0)
    cx = {b.name: b for b in build_contact(ev, ref, geo, cfg)}["C-X"]
    assert cx.A.shape == (M, nv) and cx.c.shape == (M,)
    # recompute the expected (value, grad) the builder must use
    src = nearest_index(table.points, ref.field.witness[0])       # source from the reference witness
    val, grad = geo_value_grad(table, src, ev.field.witness[0])    # query = current witness
    assert np.allclose(cx.c, cfg.w_cx * val, atol=1e-3)
    # ground channel: world_normal(I, grad) == grad ; A = w·geo_chain(grad, point_jac)
    v = rng.standard_normal(nv)
    A_expect = cfg.w_cx * np.einsum("mi,mij->mj", grad, ev.point_jac)
    assert np.allclose(cx.A @ v, A_expect @ v, atol=2e-2)         # loose: geodesic grad approx (Assumption 1)


def test_no_active_pairs_returns_empty():
    rng = np.random.default_rng(3)
    C, M, nv = 1, 3, 9
    active = np.zeros((C, M), bool)
    ev = _eval(C, M, nv, rng, active)
    ref = RobotInteractionTargets(field=_mcf(C, M, rng, active))
    geo = GeoField(tables=(None,), rot=np.eye(3)[None], pos=np.zeros((1, 3)), object_idx=(-1,))
    assert build_contact(ev, ref, geo, SolveConfig()) == []
