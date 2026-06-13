"""Tests for holosoma-style optional constraint helpers (self-collision, non-penetration).

TDD: these tests are written before the helper methods exist so they
initially fail, then pass once the helpers are copied into the solvers.
"""


def test_gmr_socp_config_defaults_off():
    from HoloNew.src.gmr_socp.config import GmrSocpRetargeterConfig
    c = GmrSocpRetargeterConfig()
    assert c.activate_obj_non_penetration is False
    assert c.activate_foot_sticking is False
    assert c.activate_self_collision is False


def test_test_socp_config_defaults_off():
    from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
    c = TestSocpRetargeterConfig()
    assert c.activate_obj_non_penetration is False
    assert c.activate_foot_sticking is False
    assert c.activate_self_collision is False


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


def test_object_name_set_before_model_load_gmr():
    """Verify object_name is correctly set and has_dynamic_object defaults False
    for a robot_only (ground) task — the default-off / parity path."""
    from HoloNew.examples.robot_retarget import RetargetingConfig
    from HoloNew.src.gmr_socp.gmr_socp import GmrSocpRetargeter

    rt = GmrSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    # Default (flag off, ground): plain xml loaded -> no dynamic object DOFs.
    assert rt.object_name == "ground"
    assert rt.has_dynamic_object is False
    assert rt._obj_poses_mj is None
    # Jacobians/phis computable on the plain model (ground collision geometry present).
    import numpy as np
    Js, phis = rt._update_jacobians_and_phis_from_q(np.copy(rt.q_init_full))
    assert isinstance(Js, dict) and isinstance(phis, dict)


def test_object_name_set_before_model_load_test_socp():
    """Mirror of the gmr test for the TestSocpRetargeter."""
    from HoloNew.examples.robot_retarget import RetargetingConfig
    from HoloNew.src.test_socp.test_socp import TestSocpRetargeter

    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    assert rt.object_name == "ground"
    assert rt.has_dynamic_object is False
    assert rt._obj_poses_mj is None
    import numpy as np
    Js, phis = rt._update_jacobians_and_phis_from_q(np.copy(rt.q_init_full))
    assert isinstance(Js, dict) and isinstance(phis, dict)


def test_foot_sticking_sequence_built():
    from HoloNew.examples.robot_retarget import RetargetingConfig
    from HoloNew.src.test_socp.test_socp import TestSocpRetargeter

    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    assert isinstance(rt.foot_sticking_sequences, list) and len(rt.foot_sticking_sequences) > 0


def test_gmr_socp_constraints_on_smoke():
    """ON smoke: GMR-SOCP with ground non-penetration + foot-sticking enabled.

    Enables both constraints via GmrSocpRetargeterConfig (the proper opt-in path,
    not post-construction mutation) on a robot_only g1 task.  The g1 xml contains
    a ground plane geom, so ground non-penetration fires; g1 FOOT_STICKING_LINKS
    contain left/right ankle names, so foot-sticking fires.

    Asserts the full retarget() completes without error and returns a sane qpos
    (correct column count, all finite values).  This proves the enabled constraint
    path is feasible end-to-end on the demo clip.
    """
    import numpy as np
    from HoloNew.config_types.retargeting import RetargetingConfig
    from HoloNew.src.gmr_socp.config import GmrSocpRetargeterConfig
    from HoloNew.src.gmr_socp.gmr_socp import GmrSocpRetargeter

    cfg = RetargetingConfig(
        task_type="robot_only",
        task_name="sub3_largebox_003",
        data_format="smplh",
        retargeter=GmrSocpRetargeterConfig(
            activate_obj_non_penetration=True,
            activate_foot_sticking=True,
        ),
    )
    rt = GmrSocpRetargeter.from_config(cfg)

    # Confirm the flags were wired through from_config (not silently dropped).
    assert rt.activate_obj_non_penetration is True
    assert rt.activate_foot_sticking is True

    res = rt.retarget()

    # Sane output: correct joint dimension and no NaN/Inf anywhere.
    assert res.qpos.shape[1] >= 36, res.qpos.shape
    assert np.isfinite(res.qpos).all(), "NaN or Inf in qpos with constraints ON"
