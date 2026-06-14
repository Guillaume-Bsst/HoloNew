"""Floor-only interaction path: query_entities works when object_sdf is None."""
import numpy as np
from HoloNew.src.test_socp.contact.contact_field import ContactField, inactive_field


def test_inactive_field_shapes_and_flags():
    f = inactive_field(5, margin=0.1)
    assert isinstance(f, ContactField)
    assert f.distance.shape == (5,) and np.all(f.distance >= 0.1)
    assert f.direction.shape == (5, 3) and np.all(f.direction == 0)
    assert f.witness.shape == (5, 3) and np.all(f.witness == 0)
    assert f.active.shape == (5,) and not f.active.any()


import pytest
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.src.test_socp.interaction import query_entities, robot_control_points


def test_query_entities_floor_only():
    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="object_interaction", task_name="sub3_largebox_003", data_format="smplh"))
    if rt.correspondence is None:
        pytest.skip("assets not present")
    rt.object_sdf = None
    q_pin = rt.pin.qpos_mj_to_q_pin(rt.q_init_full[:36])
    P = robot_control_points(rt, q_pin)
    fobj, fflr = query_entities(rt, P, rt._obj_poses_raw[0], margin=0.1)
    assert not np.asarray(fobj.active).any(), "object channel must be inactive with no SDF"
    assert np.all(np.isfinite(np.asarray(fflr.distance))), "floor field must be finite"


def test_ground_probe_floor_only_runs():
    import os, pytest
    from HoloNew.src.test_socp.contact.smplx_field import build_smplx_ground_probe
    from HoloNew.src.test_socp.contact.constants import CONTACT_MARGIN_M, OMOMO_DIR_DEFAULT
    # Use the SAME import paths that from_config uses for these two constants:
    from HoloNew.src.test_socp.correspondence.constants import SMPLX_MODEL_DIR_DEFAULT, HUMAN_GRID_DENSITY
    if not os.path.isdir(OMOMO_DIR_DEFAULT):
        pytest.skip("OMOMO data not present")
    probe = build_smplx_ground_probe(
        "sub3_largebox_003", OMOMO_DIR_DEFAULT, SMPLX_MODEL_DIR_DEFAULT,
        object_sdf=None, obj_poses=None, margin=CONTACT_MARGIN_M,
        density=HUMAN_GRID_DENSITY)
    pf = probe(0, np.zeros((52, 4)), np.zeros(3))
    assert pf.points.ndim == 2 and pf.points.shape[1] == 3
    assert np.all(np.isfinite(pf.points))
    assert not np.asarray(pf.field.active).any()
