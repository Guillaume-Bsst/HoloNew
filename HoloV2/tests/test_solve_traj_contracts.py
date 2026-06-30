"""Contrats d'orchestration solve : construction de FrameEval / FrameInfo / SolveTrajectory + rejet
des formes incohérentes au __post_init__ de SolveTrajectory."""
import types

import numpy as np
import pytest

from src.solve.contracts import FrameEval, FrameInfo, SolveTrajectory


def test_frame_eval_is_a_plain_container():
    fe = FrameEval(style=types.SimpleNamespace(position=np.zeros((1, 3))),
                   contact=types.SimpleNamespace(point_jac=np.zeros((1, 3, 6))))
    assert fe.style.position.shape == (1, 3)
    assert fe.contact.point_jac.shape == (1, 3, 6)


def test_frame_info_fields():
    fi = FrameInfo(n_iters=3, status="optimal", cost=1.5, cost_by_term={"S-pos": 1.0, "C-D": 0.5})
    assert fi.n_iters == 3 and fi.status == "optimal" and fi.cost == 1.5
    assert fi.cost_by_term["S-pos"] == 1.0


def test_solve_trajectory_valid_with_objects():
    T, nq, N = 2, 8, 1
    info = (FrameInfo(1, "optimal", 0.0, {}), FrameInfo(2, "optimal", 0.0, {}))
    traj = SolveTrajectory(qpos=np.zeros((T, nq)), object_poses=np.zeros((T, N, 7)), info=info)
    assert traj.qpos.shape == (T, nq) and traj.object_poses.shape == (T, N, 7)
    assert traj.n_frames == T


def test_solve_trajectory_valid_no_objects():
    T, nq = 3, 5
    info = tuple(FrameInfo(1, "optimal", 0.0, {}) for _ in range(T))
    traj = SolveTrajectory(qpos=np.zeros((T, nq)), object_poses=np.zeros((T, 0, 7)), info=info)
    assert traj.object_poses.shape == (T, 0, 7)


def test_solve_trajectory_bad_object_pose_width_raises():
    with pytest.raises(ValueError):                       # last dim must be 7
        SolveTrajectory(qpos=np.zeros((2, 5)), object_poses=np.zeros((2, 1, 6)),
                        info=(FrameInfo(1, "optimal", 0.0, {}), FrameInfo(1, "optimal", 0.0, {})))


def test_solve_trajectory_info_length_mismatch_raises():
    with pytest.raises(ValueError):                       # len(info) must equal T
        SolveTrajectory(qpos=np.zeros((3, 5)), object_poses=np.zeros((3, 0, 7)),
                        info=(FrameInfo(1, "optimal", 0.0, {}),))


def test_solve_trajectory_object_frames_mismatch_raises():
    with pytest.raises(ValueError):                       # object_poses T axis must equal qpos T axis
        SolveTrajectory(qpos=np.zeros((2, 5)), object_poses=np.zeros((3, 0, 7)),
                        info=(FrameInfo(1, "optimal", 0.0, {}), FrameInfo(1, "optimal", 0.0, {})))
