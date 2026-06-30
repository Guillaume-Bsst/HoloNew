"""build_object : CO-D (A=0, A_obj via cloud_jac_self/probe_jac_obj) + O (ancre I, c=se3_log_world).
nv passé via point_jac (M,3,nv). CO-X différée (pas de `geo` dans la signature) — non émise."""
import numpy as np

from src.solve.config import SolveConfig
from src.solve.terms.object import build_object
from src.solve.terms._ops import dist_jac, world_normal, se3_log_world
from src.targets.contracts import (ContactEval, ContactEnvEval, MultiChannelField,
                                    EnvironmentInteractionTargets, RobotInteractionTargets)


def _mcf(C, P, rng, active, names):
    return MultiChannelField(distance=rng.standard_normal((C, P)),
                             direction=rng.standard_normal((C, P, 3)),
                             witness=rng.standard_normal((C, P, 3)), active=active, channels=names)


def test_co_d_self_and_O(monkeypatch=None):
    rng = np.random.default_rng(0)
    N, C, P, nv, M = 1, 2, 3, 9, 1            # 1 object => channels (ground, obj0); diagonal inactive
    names = ("ground", "obj0")
    act = np.zeros((C, P), bool); act[0, :] = True       # object cloud vs GROUND active (self-channel off)
    env_field = _mcf(C, P, rng, act, names)
    env = ContactEnvEval(field=env_field, cloud_jac_self=rng.standard_normal((P, 3, 6)),
                         probe_jac_obj=rng.standard_normal((C, P, 3, 6)))
    ev = ContactEval(field=_mcf(C, M, rng, np.zeros((C, M), bool), names),
                     point_jac=rng.standard_normal((M, 3, nv)),
                     probe_jac_obj=rng.standard_normal((C, M, 3, 6)), env=(env,))
    refs = EnvironmentInteractionTargets(per_object=(_mcf(C, P, rng, act, names),))
    R0 = np.eye(3)[None]; p0 = np.zeros((1, 3))
    cfg = SolveConfig()
    blocks = {b.name: b for b in build_object(ev, refs, R0, p0, cfg)}
    cod = blocks["CO-D"]
    assert cod.A.shape == (P, nv) and np.allclose(cod.A, 0.0)
    assert cod.A_obj.shape == (P, N * 6)
    # ground channel: world frame = I, self term = w·dist_jac(dir, cloud_jac_self) in object 0 slot
    dir0 = env_field.direction[0]                         # ground -> world == local
    expect = cfg.w_cod * dist_jac(world_normal(np.eye(3), dir0), env.cloud_jac_self)
    assert np.allclose(cod.A_obj, expect)
    assert np.allclose(cod.c, cfg.w_cod * (env_field.distance[0] - refs.per_object[0].distance[0]))
    # O term: anchor block = w_obj·I, c = w_obj·se3_log_world(ref,cur) = 0 at the linearisation point
    o = blocks["O"]
    assert o.A.shape == (N * 6, nv) and np.allclose(o.A, 0.0)
    assert np.allclose(o.A_obj, cfg.w_obj * np.eye(N * 6))
    assert np.allclose(o.c, 0.0)


def test_object_object_coupling_two_objects():
    rng = np.random.default_rng(1)
    N, C, P, nv, M = 2, 3, 2, 9, 1           # channels (ground, obj0, obj1)
    names = ("ground", "obj0", "obj1")
    # object 0's cloud vs channel 2 (object 1) active -> couples δξ0 (self) AND δξ1 (probe)
    act = np.zeros((C, P), bool); act[2, :] = True
    env0_field = _mcf(C, P, rng, act, names)
    env0 = ContactEnvEval(field=env0_field, cloud_jac_self=rng.standard_normal((P, 3, 6)),
                          probe_jac_obj=rng.standard_normal((C, P, 3, 6)))
    env1 = ContactEnvEval(field=_mcf(C, P, rng, np.zeros((C, P), bool), names),
                          cloud_jac_self=rng.standard_normal((P, 3, 6)),
                          probe_jac_obj=rng.standard_normal((C, P, 3, 6)))
    ev = ContactEval(field=_mcf(C, M, rng, np.zeros((C, M), bool), names),
                     point_jac=rng.standard_normal((M, 3, nv)),
                     probe_jac_obj=rng.standard_normal((C, M, 3, 6)), env=(env0, env1))
    refs = EnvironmentInteractionTargets(per_object=(env0_field, env1.field))
    R1 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    object_rot = np.stack([np.eye(3), R1]); object_pos = np.zeros((2, 3))
    cfg = SolveConfig()
    cod = {b.name: b for b in build_object(ev, refs, object_rot, object_pos, cfg)}["CO-D"]
    assert cod.A_obj.shape == (P, N * 6)
    dir_local = env0_field.direction[2]
    # self contribution (object 0, world frame = R1 of channel 2) -> slot 0
    self_blk = cfg.w_cod * dist_jac(world_normal(R1, dir_local), env0.cloud_jac_self)
    assert np.allclose(cod.A_obj[:, 0:6], self_blk)
    # cross contribution (object 1 of channel 2, local dir) -> slot 1
    cross_blk = cfg.w_cod * dist_jac(dir_local, env0.probe_jac_obj[2])
    assert np.allclose(cod.A_obj[:, 6:12], cross_blk)


def test_O_nonzero_when_pose_drifts():
    # se3_log_world is the engine of c; check it is non-trivial when cur != ref (the general formula).
    rng = np.random.default_rng(2)
    R_ref = np.eye(3)[None]; p_ref = np.zeros((1, 3))
    R_cur = np.array([[[1, 0, 0], [0, 0, -1], [0, 1, 0.0]]]); p_cur = np.array([[0.1, 0.0, 0.0]])
    e = se3_log_world(R_ref, p_ref, R_cur, p_cur)
    assert e.shape == (1, 6) and not np.allclose(e, 0.0)
