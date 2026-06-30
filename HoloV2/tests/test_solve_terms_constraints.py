"""build_constraints : box limites articulaires sur δv[6:] (linéarisée au neutre) + TrustRegion box
par-DOF depuis la config. Robot STUB (pas de pinocchio)."""
import numpy as np

from src.solve.config import SolveConfig
from src.solve.terms.constraints import build_constraints


class _StubRobot:
    nv = 9          # 6 (free-flyer) + 3 joints
    dof = 3
    def neutral(self):
        # q = [pos(3), quat xyzw(4), joints(3)] -> joints at indices 7:10
        return np.array([0, 0, 0, 0, 0, 0, 1, 0.1, 0.0, -0.1], float)
    def joint_pos_limits(self):
        return np.array([-1.0, -2.0, -3.0]), np.array([1.0, 2.0, 3.0])


def test_joint_limits_constraint():
    robot = _StubRobot()
    cfg = SolveConfig()
    cons, _ = build_constraints(robot, cfg)
    jl = [c for c in cons if c.name == "joint_limits"][0]
    assert jl.A.shape == (3, 9) and jl.A_obj is None
    # selection picks v[6:9]
    S = np.zeros((3, 9)); S[0, 6] = S[1, 7] = S[2, 8] = 1.0
    assert np.allclose(jl.A, S)
    q0 = robot.neutral()[7:]
    assert np.allclose(jl.lb, np.array([-1.0, -2.0, -3.0]) - q0)
    assert np.allclose(jl.ub, np.array([1.0, 2.0, 3.0]) - q0)


def test_trust_region_box():
    robot = _StubRobot()
    cfg = SolveConfig(tr_base_pos=0.05, tr_base_rot=0.1, tr_joints=0.2)
    _, trs = build_constraints(robot, cfg)
    assert len(trs) == 1
    tr = trs[0]
    assert tr.var == "dv" and tr.norm == -1 and tr.radius.shape == (9,)
    assert np.allclose(tr.radius[:3], 0.05)
    assert np.allclose(tr.radius[3:6], 0.1)
    assert np.allclose(tr.radius[6:9], 0.2)
