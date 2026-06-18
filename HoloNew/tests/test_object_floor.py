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
    assert rt.lambda_d_obj > 0.0 and rt.lambda_x_obj > 0.0, "inertia mode must enable object<->floor D/X"
    assert rt.object_surface_local is not None and rt.object_surface_local.shape[1] == 3


class _StubRT:
    """Minimal rt for the object<->floor D/X unit test (only object_surface_local is read)."""
    def __init__(self, pts):
        self.object_surface_local = pts
        self.nv_a = 0


def test_object_floor_d_x_pull_displaced_box_toward_reference():
    """D/X must drive a displaced warm-start box BACK to the reference floor contact.

    Regression for the missing target: without the reference target the residual is
    ``A @ dxi`` (minimised at dxi=0), so the box never aligns no matter the weight. With
    the target it must push the contact down to the reference floor height AND pull the
    tangential footprint back to the reference."""
    from HoloNew.src.test_socp.movable import build_object_floor_blocks
    # Box bottom face (object-local), symmetric so rotation does not dominate the fit.
    pts = np.array([[0.1, 0.1, -0.1], [0.1, -0.1, -0.1],
                    [-0.1, 0.1, -0.1], [-0.1, -0.1, -0.1]], float)
    rt = _StubRT(pts)
    # Reference: box resting on the floor (bottom at z=0), centred at x=0.
    obj_pose_ref = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1], float)
    # Warm-start: floating 5 cm (bottom z=0.05, in-band) and shifted +5 cm in x.
    obj_pose = np.array([1.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.15], float)
    blocks = build_object_floor_blocks(rt, obj_pose, lambda_d_obj=10.0,
                                       lambda_x_obj=10.0, margin=0.1,
                                       obj_pose_ref=obj_pose_ref)
    assert len(blocks) == 2
    # Unconstrained least-squares: find dxi that minimises sum ||A_obj @ dxi + c||^2
    A_stack = np.vstack([b.A_obj for b in blocks])
    c_stack = np.concatenate([b.c for b in blocks])
    v, _, _, _ = np.linalg.lstsq(A_stack, -c_stack, rcond=None)
    assert v[2] < -0.02, f"D did not pull the box down to the floor: dxi_z={v[2]:.3f}"
    assert v[0] < -0.02, f"X did not pull the box back in x: dxi_x={v[0]:.3f}"


def test_object_floor_weight_independent_of_inactive_point_count():
    """The per-contact D weight must depend on the CONTACT PATCH (active points), not on
    how much of the object is sampled away from the floor. Adding far/inactive surface
    points (top, sides) must not dilute the floor-contact term — otherwise lambda_d_obj is
    not comparable to the robot's lambda_d (which normalises per carrier link)."""
    from HoloNew.src.test_socp.movable import build_object_floor_blocks
    bottom = np.array([[0.1, 0.1, -0.1], [0.1, -0.1, -0.1],
                       [-0.1, 0.1, -0.1], [-0.1, -0.1, -0.1]], float)
    far = np.array([[0.1, 0.1, 1.0], [-0.1, -0.1, 1.0]], float)   # high -> inactive
    obj_pose_ref = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10], float)  # bottom on floor
    obj_pose = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15], float)      # bottom at z=0.05

    def d_block_cost_at_zero(pts):
        rt = _StubRT(pts)
        blocks = build_object_floor_blocks(rt, obj_pose, lambda_d_obj=10.0,
                                           lambda_x_obj=0.0, margin=0.10,
                                           obj_pose_ref=obj_pose_ref)
        return float(sum(np.sum(b.c ** 2) for b in blocks))  # cost at dxi=0: ||c||^2

    v_few = d_block_cost_at_zero(bottom)
    v_many = d_block_cost_at_zero(np.vstack([bottom, far]))
    assert v_few > 0
    assert abs(v_few - v_many) < 1e-9, (
        f"inactive points leak into the contact weight: {v_few} vs {v_many}")


def test_object_floor_blocks_assemble_finite():
    """build_object_floor_blocks returns finite blocks at a test obj_pose."""
    from HoloNew.src.test_socp.movable import build_object_floor_blocks
    rt = _make(**PAPER_PLACEMENT)
    if rt.object_surface_local is None:
        pytest.skip("object surface not present")
    obj_pose = rt._obj_poses_raw[0]
    blocks = build_object_floor_blocks(rt, obj_pose, lambda_d_obj=5.0, lambda_x_obj=5.0, margin=0.1)
    assert isinstance(blocks, list)
    # The largebox rests on the floor at frame 0, so some bottom points are active.
    assert len(blocks) == 2, "expected D + X object-floor blocks with active points"
    for b in blocks:
        assert np.all(np.isfinite(b.A_obj)) and np.all(np.isfinite(b.c))


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
