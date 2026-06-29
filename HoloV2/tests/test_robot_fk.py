"""Robot kinematics (PinRobot): free-flyer FK + the robot-keyed correspondence rest pose.
Skips when the G1 URDF is not present."""
from pathlib import Path

import numpy as np
import pytest

from src.prepare.contracts import RobotSpec
from src.prepare.load.robot import build_robot_model, correspondence_rest_angles

_URDF = Path(__file__).resolve().parent.parent / "models" / "g1" / "g1_29dof.urdf"


def test_correspondence_rest_angles_is_robot_keyed():
    assert "left_elbow_joint" in correspondence_rest_angles("g1")   # G1 defined
    with pytest.raises(ValueError):
        correspondence_rest_angles("no_such_robot")                 # unknown robot -> clear error


@pytest.mark.skipif(not _URDF.exists(), reason="G1 URDF not available")
def test_pin_robot_fk_shapes_and_neutral():
    robot = build_robot_model(RobotSpec(name="g1", urdf_path=_URDF, link_names=(), dof=29, height=1.3))
    assert robot.dof == 29
    assert robot.nv == 6 + 29 and robot.nq == 7 + 29
    assert "pelvis" in robot.link_names
    n = len(robot.link_names)

    q0 = robot.neutral()
    assert q0.shape == (robot.nq,)
    assert np.isclose(np.linalg.norm(q0[3:7]), 1.0)                 # unit base quaternion

    rot, pos = robot.rest_transforms()
    assert rot.shape == (n, 3, 3) and pos.shape == (n, 3)
    assert np.allclose(rot[0] @ rot[0].T, np.eye(3), atol=1e-6)     # orthonormal

    # bend a joint -> some link relocates
    q = q0.copy()
    q[7] += 0.8
    _, pos1 = robot.link_transforms(q)
    assert not np.allclose(pos, pos1)


@pytest.mark.skipif(not _URDF.exists(), reason="G1 URDF not available")
def test_link_jacobians_match_finite_differences():
    robot = build_robot_model(RobotSpec(name="g1", urdf_path=_URDF, link_names=(), dof=29, height=1.3))
    rng = np.random.default_rng(1)
    q = robot.integrate(robot.neutral(), 0.1 * rng.standard_normal(robot.nv))   # random valid config

    rot, pos, jac_lin, jac_ang = robot.link_jacobians(q)
    n = len(robot.link_names)
    assert jac_lin.shape == (n, 3, robot.nv) and jac_ang.shape == (n, 3, robot.nv)
    assert np.allclose(pos, robot.link_transforms(q)[1], atol=1e-9)   # transforms agree with FK

    # finite-difference the WORLD position of a few links along each tangent direction.
    eps = 1e-6
    test_links = [robot.link_names.index(n) for n in ("left_elbow_link", "pelvis")
                  if n in robot.link_names]
    assert test_links, "no FD test link resolved"
    for k in range(robot.nv):
        v = np.zeros(robot.nv); v[k] = eps
        pos_p = robot.link_transforms(robot.integrate(q, v))[1]
        pos_m = robot.link_transforms(robot.integrate(q, -v))[1]
        fd = (pos_p - pos_m) / (2 * eps)                             # (L, 3) d pos / d v_k
        for i in test_links:
            assert np.allclose(jac_lin[i, :, k], fd[i], atol=1e-4), (robot.link_names[i], k)


@pytest.mark.skipif(not _URDF.exists(), reason="G1 URDF not available")
def test_pin_fk_parity_vs_yourdfpy_base_relative():
    # At the neutral free-flyer base (identity), pinocchio WORLD transforms == yourdfpy base-relative
    # transforms (same URDF kinematics). Compare a few links at a random actuated config.
    import yourdfpy
    robot = build_robot_model(RobotSpec(name="g1", urdf_path=_URDF, link_names=(), dof=29, height=1.3))
    urdf = yourdfpy.URDF.load(str(_URDF), load_meshes=False, build_scene_graph=True)

    rng = np.random.default_rng(0)
    angles = rng.uniform(-0.3, 0.3, size=29)
    cfg = {name: float(a) for name, a in zip(urdf.actuated_joint_names, angles)}
    urdf.update_cfg(np.array([cfg[n] for n in urdf.actuated_joint_names]))

    # set actuated joints in pinocchio order via the public mapping (Task 2 helper)
    q = robot.config_from_angles(cfg)
    rot, pos = robot.link_transforms(q)

    assert any(n in robot.link_names for n in ("left_elbow_link", "right_wrist_yaw_link", "left_knee_link")), \
        "no parity link found in robot.link_names — check URDF link names"
    for name in ("left_elbow_link", "right_wrist_yaw_link", "left_knee_link"):
        if name not in robot.link_names:
            continue
        i = robot.link_names.index(name)
        T = np.asarray(urdf.get_transform(name))                    # base-relative (base at origin)
        assert np.allclose(pos[i], T[:3, 3], atol=1e-5), name
        assert np.allclose(rot[i], T[:3, :3], atol=1e-5), name
