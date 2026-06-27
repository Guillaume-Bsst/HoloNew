"""Unit test for the per-segment OT coupling: every robot point must be driven by a human sample in
its OWN segment (the property that keeps hand->hand, foot->foot), and all points get assigned."""
import numpy as np

from src.prepare.point_cloud.correspondence.ot_couple import couple


def test_couple_assigns_within_segment():
    rng = np.random.default_rng(0)
    # segment 0 clustered near origin, segment 1 near (10,0,0) — well separated.
    human_pts = np.vstack([rng.normal(0, 0.1, (20, 3)), rng.normal([10, 0, 0], 0.1, (20, 3))])
    human_seg = np.array([0] * 20 + [1] * 20)
    robot_pts = np.vstack([rng.normal(0, 0.1, (15, 3)), rng.normal([10, 0, 0], 0.1, (15, 3))])
    robot_seg = np.array([0] * 15 + [1] * 15)

    smpl_idx = couple(human_pts, human_seg, robot_pts, robot_seg, reg=0.05)
    assert smpl_idx.shape == (30,)
    assert (smpl_idx >= 0).all() and int(smpl_idx.max()) < 40       # all assigned, in range
    assert (human_seg[smpl_idx] == robot_seg).all()                # never crosses a segment


def test_couple_raises_when_segment_has_no_human():
    human_pts = np.zeros((5, 3))
    human_seg = np.zeros(5, dtype=np.int64)                         # only segment 0 on the human side
    robot_pts = np.ones((3, 3))
    robot_seg = np.ones(3, dtype=np.int64)                          # segment 1 has no human source
    try:
        couple(human_pts, human_seg, robot_pts, robot_seg, reg=0.05)
        assert False, "expected ValueError for an unmatched segment"
    except ValueError:
        pass
