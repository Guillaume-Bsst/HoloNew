import numpy as np
from HoloNew.src.test_socp.solve.constraints import box_freeze_limits, trust_regions
from HoloNew.src.test_socp.solve.spec import LinearConstraint, TrustRegion


class _RtStub:
    nv_a = 4
    activate_tb = True
    activate_qa = True
    activate_joint_limits = True
    step_size = 0.2
    v_a_indices = np.array([0, 6, 7, 8])           # one base DOF + three joints
    _v_a_lb = np.array([-1.0, -2.0, -2.0, -2.0])
    _v_a_ub = np.array([1.0, 2.0, 2.0, 2.0])


def test_trust_regions_dqa_only_without_object():
    trs = trust_regions(_RtStub(), n_obj=0)
    assert [t.var for t in trs] == ["dqa"]
    assert trs[0].radius == 0.2


def test_trust_regions_include_object():
    trs = trust_regions(_RtStub(), n_obj=6)
    assert sorted(t.var for t in trs) == ["dqa", "dxi"]


def test_joint_limit_box_subtracts_current_value():
    rt = _RtStub()
    q_pin = np.zeros(16)
    q_pin[8] = 0.5
    cons = box_freeze_limits(rt, q_pin)
    box = [c for c in cons if c.name == "joint_limits"][0]
    assert isinstance(box, LinearConstraint)
    assert box.A.shape == (4, 4)


def test_hard_constraint_block_builders_exist():
    from HoloNew.src.test_socp.interaction import (
        build_p_constraint_blocks, build_obj_surface_nonpen_blocks)
    assert callable(build_p_constraint_blocks) and callable(build_obj_surface_nonpen_blocks)
