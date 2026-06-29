"""Les types d'éval q-dépendants (StyleEval / ContactEval / ContactEnvEval) : construction valide +
rejet des formes incohérentes au __post_init__ (cohérent avec MultiChannelField)."""
import numpy as np
import pytest

from src.targets.contracts import (ContactEnvEval, ContactEval, MultiChannelField, StyleEval)


def _mcf(c: int, p: int) -> MultiChannelField:
    return MultiChannelField(
        distance=np.zeros((c, p)), direction=np.zeros((c, p, 3)),
        witness=np.zeros((c, p, 3)), active=np.zeros((c, p), bool),
        channels=tuple(f"ch{i}" for i in range(c)))


def test_style_eval_construction_and_validation():
    L, nv = 4, 35
    se = StyleEval(position=np.zeros((L, 3)), rotation=np.zeros((L, 3, 3)),
                   jac_pos=np.zeros((L, 3, nv)), jac_rot=np.zeros((L, 3, nv)),
                   link_names=tuple(f"l{i}" for i in range(L)))
    assert se.position.shape == (L, 3) and se.jac_rot.shape == (L, 3, nv)

    with pytest.raises(ValueError):                                # L mismatch on jac_pos
        StyleEval(position=np.zeros((L, 3)), rotation=np.zeros((L, 3, 3)),
                  jac_pos=np.zeros((L + 1, 3, nv)), jac_rot=np.zeros((L, 3, nv)),
                  link_names=tuple(f"l{i}" for i in range(L)))
    with pytest.raises(ValueError):                               # nv mismatch jac_rot vs jac_pos
        StyleEval(position=np.zeros((L, 3)), rotation=np.zeros((L, 3, 3)),
                  jac_pos=np.zeros((L, 3, nv)), jac_rot=np.zeros((L, 3, nv + 1)),
                  link_names=tuple(f"l{i}" for i in range(L)))


def test_contact_eval_construction_and_validation():
    C, M, nv = 2, 5, 35
    env = (ContactEnvEval(field=_mcf(C, 3), cloud_jac_self=np.zeros((3, 3, 6)),
                          probe_jac_obj=np.zeros((C, 3, 3, 6))),)
    ce = ContactEval(field=_mcf(C, M), point_jac=np.zeros((M, 3, nv)),
                     probe_jac_obj=np.zeros((C, M, 3, 6)), env=env)
    assert ce.point_jac.shape == (M, 3, nv) and ce.probe_jac_obj.shape == (C, M, 3, 6)

    with pytest.raises(ValueError):                               # M mismatch on point_jac
        ContactEval(field=_mcf(C, M), point_jac=np.zeros((M + 1, 3, nv)),
                    probe_jac_obj=np.zeros((C, M, 3, 6)), env=env)
    with pytest.raises(ValueError):                               # probe_jac_obj last dim != 6
        ContactEval(field=_mcf(C, M), point_jac=np.zeros((M, 3, nv)),
                    probe_jac_obj=np.zeros((C, M, 3, 5)), env=env)


def test_contact_env_eval_validation():
    C, P = 2, 4
    ok = ContactEnvEval(field=_mcf(C, P), cloud_jac_self=np.zeros((P, 3, 6)),
                        probe_jac_obj=np.zeros((C, P, 3, 6)))
    assert ok.cloud_jac_self.shape == (P, 3, 6)

    with pytest.raises(ValueError):                               # P mismatch on cloud_jac_self
        ContactEnvEval(field=_mcf(C, P), cloud_jac_self=np.zeros((P + 1, 3, 6)),
                       probe_jac_obj=np.zeros((C, P, 3, 6)))
    with pytest.raises(ValueError):                               # probe_jac_obj channels mismatch
        ContactEnvEval(field=_mcf(C, P), cloud_jac_self=np.zeros((P, 3, 6)),
                       probe_jac_obj=np.zeros((C + 1, P, 3, 6)))
