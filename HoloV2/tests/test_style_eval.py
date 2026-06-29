"""style_eval : FK + jacobiennes géométriques des links suivis (config-free). Gardé par le G1 URDF
(le moteur pinocchio de Plan 1). position/rotation == FK direct ; jac_pos/jac_rot vs différences
finies via robot.integrate (jac_rot = vitesse angulaire monde, FD par rotvec de la rotation relative)."""
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from src.prepare.contracts import RobotSpec
from src.prepare.load.robot import build_robot_model
from src.targets.config import style_table
from src.targets.contracts import StyleEval
from src.targets.style.eval import style_eval

_URDF = Path(__file__).resolve().parent.parent / "models" / "g1" / "g1_29dof.urdf"
_SKIP = pytest.mark.skipif(not _URDF.exists(), reason="G1 URDF absent")


def _robot():
    return build_robot_model(RobotSpec(name="g1", urdf_path=_URDF, link_names=(), dof=29, height=1.3))


@_SKIP
def test_style_eval_matches_fk_and_shapes():
    robot = _robot()
    link_names = tuple(style_table("g1").keys())               # tracked links = style recipe order
    q = robot.integrate(robot.neutral(), 0.1 * np.random.default_rng(0).standard_normal(robot.nv))

    se = style_eval(robot, q, link_names)
    L, nv = len(link_names), robot.nv
    assert isinstance(se, StyleEval)
    assert se.position.shape == (L, 3) and se.rotation.shape == (L, 3, 3)
    assert se.jac_pos.shape == (L, 3, nv) and se.jac_rot.shape == (L, 3, nv)
    assert se.link_names == link_names

    rot_all, pos_all = robot.link_transforms(q)                # FK direct
    idx = [robot.link_names.index(n) for n in link_names]
    assert np.allclose(se.position, pos_all[idx], atol=1e-9)
    assert np.allclose(se.rotation, rot_all[idx], atol=1e-9)


@_SKIP
def test_style_eval_jacobians_match_finite_differences():
    robot = _robot()
    link_names = tuple(style_table("g1").keys())
    rng = np.random.default_rng(1)
    q = robot.integrate(robot.neutral(), 0.1 * rng.standard_normal(robot.nv))

    se = style_eval(robot, q, link_names)
    L, nv, eps = len(link_names), robot.nv, 1e-6
    for k in range(nv):
        v = np.zeros(nv); v[k] = eps
        se_p = style_eval(robot, robot.integrate(q, v), link_names)
        se_m = style_eval(robot, robot.integrate(q, -v), link_names)
        fd_pos = (se_p.position - se_m.position) / (2 * eps)   # (L, 3) ∂pos/∂v_k
        assert np.allclose(se.jac_pos[:, :, k], fd_pos, atol=1e-4)
        for i in range(L):
            dR = se_p.rotation[i] @ se_m.rotation[i].T          # relative rotation in WORLD frame
            omega = R.from_matrix(dR).as_rotvec() / (2 * eps)   # (3,) ∂ω/∂v_k
            assert np.allclose(se.jac_rot[i, :, k], omega, atol=1e-4), (link_names[i], k)
