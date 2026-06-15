"""Object<->floor contact (paper's object-environment pair) for inertia mode.

In inertia mode the OBJECT, like the body, is placed by contacts — not a
positional target: object surface points carried by T_obj query the floor field
and resist breaking the near-floor contact (vanishing when the object is lifted,
so it is then placed by object<->robot contact + ballistic W^o). The bundle sets
lambda_o_pos=0 (drop the anchor) and lambda_obj_floor>0.

Validated 2026-06-14 on sub3_largebox_003 (30 frames): with the floor term and NO
anchor the solved object stays near its reference (pos err ~16 mm mean, ~92 mm max
during the carry phase) vs ~270 mm drift with neither anchor nor floor term.
"""
import numpy as np
import cvxpy as cp
import pytest
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.tests.paper_placement import PAPER_PLACEMENT


def _make(**kw):
    return TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="object_interaction", task_name="sub3_largebox_003",
        data_format="smplh", retargeter=TestSocpRetargeterConfig(**kw)))


def test_inertia_bundle_object_placed_by_contacts():
    """Inertia bundle drops the object anchor and turns on object<->floor."""
    rt = _make(**PAPER_PLACEMENT)
    if rt.correspondence is None:
        pytest.skip("assets not present")
    assert rt.lambda_o_pos == 0.0, "inertia mode must drop the object position anchor"
    assert rt.lambda_obj_floor > 0.0, "inertia mode must enable object<->floor"
    assert rt.object_surface_local is not None and rt.object_surface_local.shape[1] == 3


def test_object_floor_term_assembles_finite():
    """build_object_floor_terms returns finite cvxpy terms at a test dxi."""
    from HoloNew.src.test_socp.movable import build_object_floor_terms
    rt = _make(**PAPER_PLACEMENT)
    if rt.object_surface_local is None:
        pytest.skip("object surface not present")
    dxi = cp.Variable(6)
    dxi.value = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.02])
    obj_pose = rt._obj_poses_raw[0]
    terms = build_object_floor_terms(rt, dxi, obj_pose, lambda_of=5.0, margin=0.1)
    assert isinstance(terms, list)
    # The largebox rests on the floor at frame 0, so some bottom points are active.
    assert len(terms) == 2, "expected D + X object-floor terms with active points"
    for t in terms:
        assert np.isfinite(float(t.value))


def test_object_stays_near_reference_via_floor_contact():
    """With the floor term and NO anchor, the solved object stays near reference."""
    rt = _make(**PAPER_PLACEMENT)
    if rt.correspondence is None:
        pytest.skip("assets not present")
    res = rt.retarget(max_frames=30)
    assert np.all(np.isfinite(res.qpos))
    solved = rt._obj_solved_poses[:30]
    ref = rt._obj_poses_raw[:30]
    perr = [np.linalg.norm(s[4:7] - r[4:7]) for s, r in zip(solved, ref)]
    # Contacts keep the object near reference: far tighter than the ~0.27 m drift
    # with neither anchor nor floor term. Generous ceiling (carry phase deviates).
    assert np.mean(perr) < 0.10, f"object drifted: mean pos err={np.mean(perr):.3f} m"
    assert np.max(perr) < 0.20, f"object spiked: max pos err={np.max(perr):.3f} m"
