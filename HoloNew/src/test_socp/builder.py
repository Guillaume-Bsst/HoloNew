"""Construction of TestSocpRetargeter from a RetargetingConfig.

build_from_config loads the robot model, motion data (OMOMO .pt or AMASS smplx),
the contact assets, and applies the config (including the inertia-mode bundle).
Kept out of test_socp.py so the solver file stays focused on the SQP itself.
"""
from __future__ import annotations

import numpy as np


def build_from_config(cls, cfg) -> "TestSocpRetargeter":
    """Build a TestSocpRetargeter and populate its motion inputs.

    Loads motion data directly from the .pt file without going through
    holosoma's preprocess_motion_data or initialize_robot_pose, since the
    GMR retarget uses only compute_stages' 'ground' output and the base
    init is fully overridden by the ground pelvis position and orientation.

    Args:
        cfg: RetargetingConfig instance (task_type must be "robot_only",
            data_format must be "smplh" or None).

    Returns:
        Configured TestSocpRetargeter ready to call .retarget().
    """
    from HoloNew.config_types.data_type import MotionDataConfig
    from HoloNew.config_types.robot import RobotConfig
    from HoloNew.examples.robot_retarget import (
        DEFAULT_DATA_FORMATS,
        build_retargeter_kwargs_from_config,
        create_task_constants,
    )
    from HoloNew.src.holosoma.preprocess import ground_to_floor
    from .preprocess import compute_stages
    from .tables import HUMAN_ROOT_NAME, MAPPED_BODY_NAMES
    from .targets import load_pt_joints, load_pt_quaternions

    task_type = cfg.task_type
    data_format = cfg.data_format or DEFAULT_DATA_FORMATS[task_type]

    # Ensure robot / motion configs are consistent
    if cfg.robot_config.robot_type != cfg.robot:
        cfg.robot_config = RobotConfig(robot_type=cfg.robot)
    if (cfg.motion_data_config.robot_type != cfg.robot
            or cfg.motion_data_config.data_format != data_format):
        cfg.motion_data_config = MotionDataConfig(
            data_format=data_format, robot_type=cfg.robot
        )

    constants = create_task_constants(
        robot_config=cfg.robot_config,
        motion_data_config=cfg.motion_data_config,
        task_config=cfg.task_config,
        task_type=task_type,
    )

    # Build retargeter kwargs and construct the retargeter
    kwargs = build_retargeter_kwargs_from_config(
        cfg.retargeter, constants, object_urdf_path=None, task_type=task_type
    )
    # Holosoma-style constraints are OPT-IN for TEST-SOCP and default OFF
    # (holosoma's RetargeterConfig defaults them ON). Take the activate_*
    # flags from a TEST-SOCP-specific config: honor one if the caller passed it,
    # else force OFF so the default solve is unchanged.
    from .config import TestSocpRetargeterConfig
    sc = cfg.retargeter if isinstance(cfg.retargeter, TestSocpRetargeterConfig) else TestSocpRetargeterConfig()
    kwargs["activate_obj_non_penetration"] = sc.activate_obj_non_penetration
    kwargs["activate_foot_sticking"] = sc.activate_foot_sticking
    kwargs["activate_self_collision"] = sc.activate_self_collision
    # Surface ε flat: feed self_collision_margin into the companion config's tolerance
    # (the flat field wins). build_retargeter_kwargs_from_config set kwargs["self_collision"]
    # from the inherited RetargeterConfig; SelfCollisionConfig is frozen, so replace().
    if kwargs.get("self_collision") is not None:
        import dataclasses
        kwargs["self_collision"] = dataclasses.replace(
            kwargs["self_collision"], tolerance=sc.self_collision_margin)
    # Each cost term has its own activate_* switch (config §3): the switch alone decides,
    # so resolve the EFFECTIVE weight here (tuned value when on, 0 when off) and feed the
    # solver, whose gating stays the simple "weight > 0". One flag per weight.
    kwargs["L_interaction"] = sc.L_interaction
    kwargs["L_floor"] = sc.L_floor
    kwargs["L_object"] = sc.L_object
    kwargs["sdf_resolution"] = sc.sdf_resolution
    kwargs["lambda_d"] = sc.lambda_d if sc.activate_wd else 0.0
    kwargs["lambda_x"] = sc.lambda_x if sc.activate_wx else 0.0
    kwargs["lambda_p"] = sc.lambda_p if sc.activate_wp else 0.0
    kwargs["sigma_v"] = sc.sigma_v
    kwargs["lambda_r"] = sc.lambda_r if sc.activate_wr else 0.0
    kwargs["sigma_qddot"] = sc.sigma_qddot
    kwargs["sigma_Vdot"] = sc.sigma_Vdot
    kwargs["sigma_R"] = sc.sigma_R
    kwargs["sigma_a"] = sc.sigma_a
    kwargs["sigma_L"] = sc.sigma_L
    kwargs["sigma_ao"] = sc.sigma_ao
    kwargs["sigma_omega"] = sc.sigma_omega
    kwargs["lambda_smooth"] = sc.lambda_smooth if sc.activate_smooth else 0.0
    kwargs["lambda_qdiag"] = sc.lambda_qdiag if sc.activate_qdiag else 0.0
    kwargs["lambda_nominal"] = sc.lambda_nominal if sc.activate_nominal else 0.0
    kwargs["nominal_tau"] = sc.nominal_tau
    kwargs["activate_pos_tracking"] = sc.activate_pos_tracking
    kwargs["activate_rot_tracking"] = sc.activate_rot_tracking
    kwargs["lambda_pos"] = sc.lambda_pos
    kwargs["sigma_p"] = sc.sigma_p
    kwargs["lambda_rot"] = sc.lambda_rot
    kwargs["sigma_rot"] = sc.sigma_rot
    kwargs["lambda_ws"] = sc.lambda_ws if sc.activate_ws else 0.0
    kwargs["style_weights"] = sc.style_weights  # None -> solver uses STYLE_WEIGHT_TABLE
    # Centroidal: one switch per term; the solver's master activate_centroidal is the OR.
    kwargs["activate_centroidal"] = (sc.activate_wc or sc.activate_wc_pos
                                     or sc.activate_wl or sc.activate_wcv)
    kwargs["lambda_c"] = sc.lambda_c if sc.activate_wc else 0.0
    kwargs["lambda_c_pos"] = sc.lambda_c_pos if sc.activate_wc_pos else 0.0
    kwargs["lambda_l"] = sc.lambda_l if sc.activate_wl else 0.0
    kwargs["lambda_cv"] = sc.lambda_cv if sc.activate_wcv else 0.0
    kwargs["sigma_cv"] = sc.sigma_cv
    kwargs["activate_wl_track"] = sc.activate_wl_track
    kwargs["lambda_l_track"] = sc.lambda_l_track
    # §1 Variables: q_a / T_B are free (frozen when their flag is False); activate_tm (§1)
    # makes the object a variable. The three W^o weights are each switched independently.
    kwargs["activate_qa"] = sc.activate_qa
    kwargs["activate_tb"] = sc.activate_tb
    kwargs["activate_tm"] = sc.activate_tm
    kwargs["lambda_o"] = sc.lambda_o if sc.activate_wo else 0.0
    kwargs["lambda_o_pos"] = sc.lambda_o_pos if sc.activate_wo_pos else 0.0
    # Object-as-carrier interaction (object ↔ environment): mirror the robot D/X/P gating.
    kwargs["lambda_d_obj"] = sc.lambda_d_obj if sc.activate_wd_obj else 0.0
    kwargs["lambda_x_obj"] = sc.lambda_x_obj if sc.activate_wx_obj else 0.0
    kwargs["lambda_p_obj"] = sc.lambda_p_obj if sc.activate_wp_obj else 0.0
    kwargs["sigma_v_obj"] = sc.sigma_v_obj
    kwargs["activate_obj_surface_nonpen"] = sc.activate_obj_surface_nonpen
    kwargs["obj_surface_nonpen_tol"] = sc.obj_surface_nonpen_tol
    kwargs["activate_persistence"] = sc.activate_persistence
    kwargs["persistence_tol"] = sc.persistence_tol
    kwargs["n_iter_first"] = sc.n_iter_first
    kwargs["n_iter_per_frame"] = sc.n_iter_per_frame
    kwargs["iterate_step_tol"] = sc.iterate_step_tol
    # Floor-as-entity and object-scene loading pass straight through, like every other
    # field: the builder applies NO hidden rewrites or presets.
    kwargs["load_object_scene"] = sc.load_object_scene

    # No config policing: the builder passes everything through as-is. Physically
    # inconsistent combinations are allowed (e.g. interaction without non-penetration lets
    # the D term push the base through the floor) — that shows up directly in the result.
    # Couplings worth remembering: interaction (D/X/P) needs an object/floor entity and the
    # d>=0 non-penetration constraint to stay stable; the W^o weights need activate_tm.
    rt = cls(**kwargs)

    # Load raw joint positions + per-joint quaternions. Two sources:
    #  - OMOMO .pt (smplh): 52-joint layout, positions + stored quaternions.
    #  - AMASS SMPL-X (data_format="smplx"): a processed .npz (from
    #    data_utils/prep_amass_smplx_for_rt) with 22 body joints + world
    #    orientations, remapped into the SMPLH 52-slot layout the tables expect.
    #    This is the path for flight/locomotion clips (SFU etc.), robot_only.
    # pt_path is defined for both paths (object loading references it); for the
    # smplx path it simply does not exist, and object loading is gated on the
    # task having a real object SDF (robot_only smplx has none).
    pt_path = cfg.data_path / f"{cfg.task_name}.pt"
    rt._smplx_orientations = None   # AMASS 22 SMPL-order joints (for the probe)
    rt._smplx_betas = None
    rt._smplx_gender = "neutral"
    if data_format == "smplx":
        from .targets import load_smplx_to_smplh_layout
        from .tables import HUMAN_BODY_TO_IDX
        npz_path = cfg.data_path / f"{cfg.task_name}.npz"
        raw_joints, human_quat, _smplx_height = load_smplx_to_smplh_layout(
            npz_path, MAPPED_BODY_NAMES, HUMAN_BODY_TO_IDX)
        _smplx_npz = np.load(npz_path)
        rt._smplx_orientations = np.asarray(
            _smplx_npz["global_joint_orientations"], dtype=np.float64)
        if "betas" in _smplx_npz.files:
            rt._smplx_betas = np.asarray(_smplx_npz["betas"], dtype=np.float32)
            rt._smplx_gender = str(_smplx_npz["gender"]) if "gender" in _smplx_npz.files else "neutral"
    else:
        raw_joints = load_pt_joints(pt_path)    # (T, 52, 3) raw positions
        human_quat = load_pt_quaternions(pt_path)  # (T, 52, 4) wxyz

    # Align T between raw_joints and human_quat (both come from the same
    # file so they are equal in length, but guard against edge cases)
    T = min(raw_joints.shape[0], human_quat.shape[0])
    raw_joints = raw_joints[:T]
    human_quat = human_quat[:T]

    # Build per-frame foot sticking sequence from SMPL joint velocities.
    # Gated by activate_foot_sticking (False by default) in solve; building the
    # sequence here is harmless and does not affect the default solve path.
    # The smplx remap does not preserve the full DEMO_JOINTS layout, so skip it
    # there (foot sticking is off by default and only needs the mapped feet).
    if data_format == "smplx":
        rt.foot_sticking_sequences = []
    else:
        from HoloNew.src.utils import extract_foot_sticking_sequence_velocity
        toe_names = cfg.motion_data_config.toe_names
        rt.foot_sticking_sequences = extract_foot_sticking_sequence_velocity(
            raw_joints, constants.DEMO_JOINTS, toe_names)

    # All joints are zero; base is overridden below from the ground pelvis
    q_init_full = np.zeros(rt.nq)

    rt.human_quat = human_quat    # (T, 52, 4) wxyz
    rt.q_init_full = q_init_full  # (nq,) — base will be set from ground below

    # Ground the raw input onto the floor first (like holosoma) so every downstream
    # stage lives in the grounded world. GMR's own floor correction (the 'ground'
    # stage) re-grounds afterwards — a constant z-shift it cancels out — so the solved
    # targets are unchanged, but the mapped/scaled/offset stages now follow the
    # grounded input and the 'Grounded' display stage is the real chain input.
    if data_format == "smplx":
        # raw_joints is in the SMPLH slot layout; ground on the mapped feet.
        from .tables import HUMAN_BODY_TO_IDX
        toe_indices = [HUMAN_BODY_TO_IDX["left_foot"], HUMAN_BODY_TO_IDX["right_foot"]]
    else:
        toe_indices = [constants.DEMO_JOINTS.index(n) for n in cfg.motion_data_config.toe_names]
    rt.gmr_grounded = ground_to_floor(raw_joints, toe_indices)
    # Robot-root placement (config §0). Each axis: None -> AUTO = the per-clip
    # ROBOT_HEIGHT / human_height (computed, not hardcoded), which puts the root at the
    # robot's own height; a float = explicit multiplier on the raw grounded axis. TEST
    # defaults: XY 1.0 (raw, so the GMR targets and the SmplxGroundProbe contact field
    # share one world frame), Z None (auto). Placement is applied inside scale();
    # proportions and the Z floor-drop are unaffected.
    from HoloNew.src.holosoma.preprocess import calculate_scale_factor
    _robot_h = constants.ROBOT_HEIGHT
    if data_format == "smplx":
        smpl_scale = _robot_h / float(_smplx_height)
    else:
        try:
            smpl_scale = calculate_scale_factor(cfg.task_name, _robot_h)
        except Exception:  # noqa: BLE001 - height table may lack this subject
            smpl_scale = _robot_h / (cfg.motion_data_config.default_human_height or 1.78)
    _rob_xy = sc.scale_xy_robot if sc.scale_xy_robot is not None else smpl_scale
    # Robot Z native default is the morphological base (HUMAN_SCALE_TABLE[pelvis]*ratio),
    # NOT smpl_scale: pass None straight through so scale() takes its native branch and the
    # z placement matches GMR / gmr_socp. (XY native = smpl_scale is correct; Z is not.)
    _rob_z = sc.scale_z_robot
    # Object native default per axis (matches gmr_socp): XY -> smpl_scale (placed with the
    # robot), Z -> raw 1.0 (the object is a rigid body, not morphed, so its native Z is the
    # raw grounded height). Our defaults keep the object raw (1.0 / 1.0) so its trajectory
    # is unchanged; the None fallbacks only matter when an axis is explicitly set to AUTO.
    _obj_xy = sc.scale_xy_object if sc.scale_xy_object is not None else smpl_scale
    _obj_z = sc.scale_z_object if sc.scale_z_object is not None else 1.0
    rt.gmr_stages = compute_stages(
        rt.gmr_grounded, human_quat, scale_xy=_rob_xy, scale_z=_rob_z,
    )
    rt.gmr_ground = rt.gmr_stages["ground"]
    ground = rt.gmr_ground
    _pelvis_bi = MAPPED_BODY_NAMES.index(HUMAN_ROOT_NAME)
    rt.q_init_full[:3] = ground["pos"][0, _pelvis_bi]    # base at frame-0 pelvis target
    rt.q_init_full[3:7] = ground["quat"][0, _pelvis_bi]  # base orientation at frame-0 target

    # Raw object poses [qw, qx, qy, qz, x, y, z] used by the smplx_ground_probe
    # and D/X interaction terms. None until the object SDF block loads them.
    rt._obj_poses_raw = None

    # Frame time step for every temporal term, from the config fps (OMOMO is 30 fps).
    rt._dt = 1.0 / sc.fps

    # Precompute the lumped reference angular momentum L_ref(t) for W^L tracking
    # (opt-in). Built from the GMR target mapped-body trajectory + robot link
    # masses; consumed by build_lumped_L_term in the solve.
    rt._lumped_frames = None
    rt._lumped_masses = None
    rt._L_ref_all = None
    if rt.activate_wl_track:
        from .centroidal import (
            mapped_frame_masses_and_names, reference_orbital_angular_momentum)
        rt._lumped_frames, rt._lumped_masses = mapped_frame_masses_and_names(rt)
        rt._L_ref_all = reference_orbital_angular_momentum(
            rt.gmr_ground["pos"], rt._lumped_masses, rt._dt)

    # Load object poses in MuJoCo qpos order for per-frame object qpos drive.
    # Requires the object SCENE (load_object_scene) — that is what swaps in the xml
    # with the object free joint (has_dynamic_object). Without it there is nothing to
    # drive, so leave None and the retarget loop's object-qpos block is skipped. This
    # keeps the loading condition consistent with the model-swap condition; otherwise
    # `activate_obj_non_penetration` alone (with load_object_scene=False, the GMR
    # baseline default) loads poses with no free joint and trips the guard below.
    rt._obj_poses_mj = None
    if (sc.activate_obj_non_penetration and sc.load_object_scene
            and rt.object_name not in (None, "ground")):
        from HoloNew.examples.robot_retarget import convert_object_poses_to_mujoco_order
        from HoloNew.src.utils import load_intermimic_data
        _, obj_poses = load_intermimic_data(str(pt_path))   # (T, 7) [qw,qx,qy,qz,x,y,z]
        obj_poses = obj_poses[:T].copy()
        # Place the object independently of the robot (no-op at the TEST defaults 1.0).
        obj_poses[:, 4:6] *= _obj_xy   # XY
        obj_poses[:, 6] *= _obj_z      # Z
        # Convert from [qw,qx,qy,qz,x,y,z] to MuJoCo order [x,y,z,qw,qx,qy,qz]
        rt._obj_poses_mj = convert_object_poses_to_mujoco_order(obj_poses)

    # Fail loudly on a misconfigured object scene: poses loaded but the model
    # has no object free joint would silently skip the per-frame object drive.
    if rt._obj_poses_mj is not None and not rt.has_dynamic_object:
        raise RuntimeError(
            f"[{cls.__name__}] Object poses loaded but has_dynamic_object is False: "
            f"the scene xml for '{rt.object_name}' did not add a free joint. "
            "Check SCENE_XML_FILE / robot_urdf_file naming."
        )

    # Load the bundled human->G1 correspondence table (data only,
    # NOT used in the solve yet — will be wired in a later task).
    from pathlib import Path
    from HoloNew.src.test_socp.correspondence.build_correspondence import load_correspondence, build_table
    from HoloNew.src.test_socp.correspondence.constants import (
        G1_29DOF_URDF, SMPLX_MODEL_DIR_DEFAULT, HUMAN_GRID_DENSITY, G1_DENSITY, OT_REG,
    )
    _bundled = Path(__file__).resolve().parent.parent.parent / "assets" / "correspondence" / "corr_neutral.npz"
    if _bundled.exists():
        rt.correspondence = load_correspondence(_bundled)
    elif Path(SMPLX_MODEL_DIR_DEFAULT).is_dir():
        rt.correspondence = build_table(SMPLX_MODEL_DIR_DEFAULT, "neutral", None,
                                        G1_29DOF_URDF, HUMAN_GRID_DENSITY, G1_DENSITY, OT_REG)

    # Object SDF: built on demand and disk-cached, keyed by the configured (L, resolution)
    # so the field's band always matches the run's interaction length. No fixed bundled
    # asset, no full-motion contact precompute (that violated the online principle).
    from HoloNew.src.test_socp.contact.backends.sdf import load_or_build_object_sdf
    _contact_assets = Path(__file__).resolve().parent.parent.parent / "assets" / "contact"
    # Effective interaction lengths: master L_interaction, with optional per-channel
    # overrides. The object SDF band must cover L_object; the field query margin must reach
    # max(L_floor, L_object) so nothing is clamped before L (see L audit).
    _L_obj_eff = sc.L_object if sc.L_object is not None else sc.L_interaction
    _L_flr_eff = sc.L_floor if sc.L_floor is not None else sc.L_interaction
    _probe_margin = float(max(_L_flr_eff, _L_obj_eff))
    _sdf_res = sc.sdf_resolution
    _mesh_file = getattr(constants, "OBJECT_MESH_FILE", None)
    # Build/load the object SDF only for tasks that actually have an object. robot_only
    # (object_name "ground", e.g. smplx locomotion clips with no .pt object poses) must not
    # pull in object loading. Floor-only inertia keeps object_sdf=None.
    if rt.object_name not in (None, "ground"):
        if _mesh_file is not None and Path(_mesh_file).exists():
            # Load the cache matching (L, resolution); bake + cache it if absent.
            rt.object_sdf = load_or_build_object_sdf(
                _mesh_file, _L_obj_eff, _sdf_res, cache_dir=_contact_assets)
        else:
            print(f"[TestSocp] object mesh not found ({_mesh_file}); no object SDF "
                  f"(floor-only). Set OBJECT_MESH_FILE to enable the object channel.")

    # Object surface control points (object-local) for the object<->floor
    # inertia term. Sampled once from the object mesh; only needed when the
    # object pose is a variable (movable) on an object task. (_mesh_file computed above.)
    if (rt.object_sdf is not None and _mesh_file is not None
            and Path(_mesh_file).exists()):
        from HoloNew.src.test_socp.movable import sample_object_surface
        rt.object_surface_local = sample_object_surface(_mesh_file)

    # Online SMPL-X -> object-SDF probe (causal, per frame). Built only when the
    # object SDF is available: sample the subject SMPL-X surface once. The human is
    # placed at its Grounded pose in retarget(); the object pose is used as-is (the
    # raw human floats, the object sits correctly, so only the human is grounded).
    # The floor is ALWAYS an interaction target, so the SMPL-X ground field is always
    # built (floor-only when there is no object). Needs the task's SMPL data, present for
    # every TEST clip.
    if True:
        from HoloNew.src.test_socp.contact.constants import CONTACT_MARGIN_M, OMOMO_DIR_DEFAULT
        from HoloNew.src.test_socp.contact.smplx_field import build_smplx_ground_probe
        from HoloNew.src.test_socp.correspondence.human_body import PointCloudCache
        from HoloNew.src.utils import load_intermimic_data
        # When an object SDF is present, load its raw poses so retarget() and
        # build_dx_terms can access them; floor-only mode has no object channel.
        if rt.object_sdf is not None:
            _, obj_poses = load_intermimic_data(str(pt_path))   # (T, 7) [qw,qx,qy,qz,x,y,z]
            obj_poses = obj_poses[:T].copy()
            # Place the object independently (no-op at the TEST defaults 1.0); the D/X
            # interaction and movable terms read these poses.
            obj_poses[:, 4:6] *= _obj_xy   # XY
            obj_poses[:, 6] *= _obj_z      # Z
            rt._obj_poses_raw = obj_poses
            _obj_poses_arg = obj_poses
        else:
            rt._obj_poses_raw = None
            _obj_poses_arg = None
        corr_cache = None
        if rt.correspondence is not None:
            corr_cache = PointCloudCache(tri_idx=rt.correspondence.tri_idx,
                                         bary=rt.correspondence.bary)
        # AMASS (smplx) clips carry their own betas/gender and pose the body from
        # the 22 SMPL-order joints; OMOMO loads betas via task metadata.
        _is_smplx = rt._smplx_betas is not None
        # Query margin = max channel L so the field reports the true distance out to L
        # (no clamp before L); per-channel activation is re-gated by L_floor/L_object.
        rt.smplx_ground_probe = build_smplx_ground_probe(
            cfg.task_name, OMOMO_DIR_DEFAULT, SMPLX_MODEL_DIR_DEFAULT,
            rt.object_sdf, _obj_poses_arg, _probe_margin, HUMAN_GRID_DENSITY,
            cache=corr_cache,
            betas=(rt._smplx_betas if _is_smplx else None),
            gender=(rt._smplx_gender if _is_smplx else None),
            smpl_order=_is_smplx)

    return rt
