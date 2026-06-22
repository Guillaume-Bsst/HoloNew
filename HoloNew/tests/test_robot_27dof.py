"""G1 27-DOF support: a robot name like 'g1_27dof' must parse to (type=g1, dof=27),
and the hardcoded g1 joint-limit / cost overrides (defined at 29-DOF indices) must be
remapped to 27-DOF — waist_roll (joint 13) and waist_pitch (joint 14) are removed, so
every higher joint shifts down by 2. A stale 29-DOF index would overflow the 27-DOF
model (nv=33) and crash the solve."""
from HoloNew.config_types.robot import RobotConfig, parse_robot_name


def test_parse_robot_name():
    assert parse_robot_name("g1") == ("g1", None)
    assert parse_robot_name("g1_27dof") == ("g1", 27)
    assert parse_robot_name("g1_29dof") == ("g1", 29)
    assert parse_robot_name("t1") == ("t1", None)
    assert parse_robot_name("G1_27dof") == ("g1", 27)   # case-insensitive


def test_27dof_resolves_dof_and_model():
    c = RobotConfig(robot_type="g1", robot_dof=27)
    assert c.ROBOT_DOF == 27
    assert c.ROBOT_URDF_FILE == "models/g1/g1_27dof.urdf"


def test_robotconfig_accepts_dof_suffixed_name():
    # A dof-suffixed name passed as robot_type auto-splits, so every construction site
    # (view_stages, builder, robot_retarget) supports '--robot g1_27dof' unchanged.
    c = RobotConfig(robot_type="g1_27dof")
    assert c.robot_type == "g1" and c.ROBOT_DOF == 27
    assert c.ROBOT_URDF_FILE == "models/g1/g1_27dof.urdf"
    # Bare name keeps the type default.
    assert RobotConfig(robot_type="g1").ROBOT_DOF == 29


def test_motiondataconfig_accepts_dof_suffixed_name():
    from HoloNew.config_types.data_type import MotionDataConfig
    m = MotionDataConfig(data_format="smplx", robot_type="g1_27dof")
    assert m.robot_type == "g1"   # joint-mapping lookup is keyed by the bare type


def test_29dof_manual_tables_unchanged():
    c = RobotConfig(robot_type="g1")  # default dof 29
    assert c.MANUAL_COST == {"19": 0.2, "20": 0.2}
    # waist_roll/pitch (qpos 20/21) and both 3-DOF wrists present at 29-DOF indices.
    assert "20" in c.MANUAL_LB and "21" in c.MANUAL_LB
    assert {"33", "34", "35"} <= set(c.MANUAL_LB)


def test_27dof_manual_tables_remapped():
    c = RobotConfig(robot_type="g1", robot_dof=27)
    nv = 6 + 27  # tangent size; LB/UB apply at qpos k -> tangent k-1, must stay < nv

    # waist_roll/pitch dropped; higher qpos indices shifted -2; base quat (3-6) kept.
    assert c.MANUAL_LB == {"3": -1.0, "4": -1.0, "5": -1.0, "6": -1.0,
                           "24": -0.1, "25": -0.1, "26": -0.05,
                           "31": -0.1, "32": -0.1, "33": -0.05}
    assert c.MANUAL_UB == {"3": 1.0, "4": 1.0, "5": 1.0, "6": 1.0,
                           "23": 1.4, "24": 0.2, "25": 0.3, "26": 0.05,
                           "30": 1.4, "31": 0.2, "32": 0.3, "33": 0.05}
    # MANUAL_COST is in actuated-joint space (drop joints 13/14, shift >14 by -2).
    assert c.MANUAL_COST == {"17": 0.2, "18": 0.2}

    # No override overflows the 27-DOF model.
    for k in list(c.MANUAL_LB) + list(c.MANUAL_UB):
        if int(k) >= 7:
            assert int(k) - 1 < nv
    for k in c.MANUAL_COST:
        assert int(k) < 27
