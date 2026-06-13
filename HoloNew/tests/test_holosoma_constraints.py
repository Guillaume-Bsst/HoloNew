"""Tests for holosoma-style optional constraint helpers (self-collision, non-penetration).

TDD: these tests are written before the helper methods exist so they
initially fail, then pass once the helpers are copied into the solvers.
"""
def test_self_collision_solves():
    from HoloNew.examples.robot_retarget import RetargetingConfig
    from HoloNew.src.test_socp.test_socp import TestSocpRetargeter

    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    # Turn the flag on post-construction for a quick unit check.
    rt.activate_self_collision = True
    rt._self_collision_enabled = len(rt._self_collision_geom_pairs) > 0 or True
    # Compute constraints for frame 0 at the init config — must not raise.
    Js, phis = rt._compute_self_collision_constraints(0)
    # Default config has no geom pairs -> empty dicts; this is a no-crash smoke check.
    assert isinstance(Js, dict) and isinstance(phis, dict)


def test_ground_non_penetration_solves():
    import numpy as np
    from HoloNew.examples.robot_retarget import RetargetingConfig
    from HoloNew.src.test_socp.test_socp import TestSocpRetargeter

    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    rt.activate_obj_non_penetration = True
    # q_init_full is set by from_config; it is a valid full qpos of length nq.
    Js, phis = rt._update_jacobians_and_phis_from_q(np.copy(rt.q_init_full))
    assert isinstance(phis, dict)  # robot<->ground pairs within threshold, or empty
