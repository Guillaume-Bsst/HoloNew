"""Run the selected retargeting methods, then open the per-method stage viewer.

For every requested method (holosoma native SOCP, GMR-SOCP, TEST-SOCP) this
builds a ``MethodViz`` carrying the solved robot trajectory plus the named
skeleton stages of that method's preprocessing pipeline, then binds them to the
viewer's Method/Stage dropdowns.

Use ``--methods`` to pick a subset, e.g. ``--methods gmr_socp`` to solve only
that optimizer instead of all three.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import trimesh
import tyro

from HoloNew.config_types.data_type import MotionDataConfig
from HoloNew.config_types.robot import RobotConfig
from HoloNew.examples.robot_retarget import (
    DEFAULT_DATA_FORMATS,
    RetargetingConfig,
    convert_object_poses_to_mujoco_order,
    create_task_constants,
    load_motion_data,
    run_headless,
)
from HoloNew.src import skeleton
from HoloNew.src.test_socp.contact.backends.sdf import (
    band_points,
    load_or_build_object_sdf,
    object_sdf_cache_path,
)
from HoloNew.src.test_socp.contact.constants import OMOMO_DIR_DEFAULT
from HoloNew.src.test_socp.contact.viz import signed_distance_colors
from HoloNew.src.test_socp.correspondence.constants import SMPLX_MODEL_DIR_DEFAULT
from HoloNew.src.test_socp.correspondence.human_body import HumanBody
from HoloNew.src.test_socp.correspondence.human_metadata import load_human_metadata
from HoloNew.src.gmr_socp.gmr_socp import _BODY_NAME_REMAP, GmrSocpRetargeter
from HoloNew.src.gmr_socp.tables import HUMAN_BODY_TO_IDX, IK_MATCH_TABLE1
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
from HoloNew.src.holosoma.preprocess import (
    compute_holosoma_stages,
    scale_object_poses_to_center,
)
from HoloNew.src.robot_fk import robot_link_poses
from HoloNew.src.stages import ROBOT_STAGE
from HoloNew.src.utils import load_intermimic_data, load_intermimic_quats, load_object_data
from HoloNew.src.viewer import MethodViz, Viewer

logger = logging.getLogger(__name__)

Method = Literal["holosoma", "gmr_socp", "test_socp"]


@dataclass
class ViewStagesConfig(RetargetingConfig):
    # Which optimizers to solve and show, in the given order. Defaults to all three.
    methods: tuple[Method, ...] = ("holosoma", "gmr_socp", "test_socp")
    # Original OMOMO dataset root (the one holding
    # data/{train,test}_diffusion_manip_seq_joints24.p, NOT OMOMO_new). Supplies
    # the subject's SMPL-X betas + gender for the mesh. Defaults to OMOMO_DIR_DEFAULT
    # so the correct subject shape loads automatically; pass another path to override
    # (a missing/absent file degrades gracefully to the neutral shape).
    omomo_dir: Path | None = Path(OMOMO_DIR_DEFAULT)
    # Cap the number of solved/displayed frames. Long scenes (SFU dance, OMOMO
    # manipulation) run to several hundred frames; capping keeps the solve short
    # when you only want to inspect the motion. None solves the whole clip.
    max_frames: int | None = None


def view(cfg: ViewStagesConfig) -> None:
    if not cfg.methods:
        raise ValueError("--methods must select at least one optimizer")

    data_format = cfg.data_format or DEFAULT_DATA_FORMATS[cfg.task_type]

    # Keep the nested configs consistent with the top-level selections, the same
    # way robot_retarget.main() does before building constants / loading data.
    if cfg.robot_config.robot_type != cfg.robot:
        cfg.robot_config = RobotConfig(robot_type=cfg.robot)
    if cfg.motion_data_config.robot_type != cfg.robot or cfg.motion_data_config.data_format != data_format:
        cfg.motion_data_config = MotionDataConfig(data_format=data_format, robot_type=cfg.robot)

    constants = create_task_constants(
        robot_config=cfg.robot_config,
        motion_data_config=cfg.motion_data_config,
        task_config=cfg.task_config,
        task_type=cfg.task_type,
    )
    # load_motion_data returns RAW joints (preprocessing is a later, separate step).
    raw_joints, _object_poses, smpl_scale = load_motion_data(
        cfg.task_type, data_format, cfg.data_path, cfg.task_name, constants, cfg.motion_data_config
    )

    # Per-joint quaternions only exist for the smplh .pt format; the SMPL-X mesh
    # needs them. Everything else degrades gracefully to skeleton-only.
    original_quats = None
    if data_format == "smplh":
        try:
            original_quats = load_intermimic_quats(str(cfg.data_path / f"{cfg.task_name}.pt"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("No per-joint quats (%s); SMPL-X mesh disabled.", exc)

    # SMPL-X shape (betas) + gender for this subject, read from the original OMOMO
    # .p files keyed by sequence name. Without them the mesh uses the neutral mean
    # shape. The .pt motion (OMOMO_new) does not carry betas, hence the separate dir.
    betas, gender = None, "neutral"
    if cfg.omomo_dir is not None:
        betas, gender = load_human_metadata(cfg.omomo_dir, cfg.task_name)
        if betas is not None:
            logger.info("Loaded SMPL-X shape for %s: gender=%s, %d betas",
                        cfg.task_name, gender, betas.shape[0])
        else:
            logger.warning("No SMPL-X shape found for %s in %s; using neutral shape.",
                           cfg.task_name, cfg.omomo_dir)

    human_body = None
    if original_quats is not None:
        try:
            human_body = HumanBody(SMPLX_MODEL_DIR_DEFAULT, betas, gender)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SMPL-X unavailable (%s); mesh disabled.", exc)

    # NOTE: raw_joints is deliberately NOT re-grounded here. The Original stage must
    # be the exact input each solver's preprocess receives (holosoma and GMR both
    # read the same raw .pt joints), and each solver applies its own floor drop
    # downstream (holosoma Grounded / GMR Ground). The SMPL-X mesh is posed on these
    # same raw joints so it stays aligned with the Original skeleton.

    # Object motion (from the .pt), kept in the object's local frame so the viewer can
    # render it per stage: raw pose on unscaled stages, centred pose
    # (scale_object_poses_to_center) on the scaled stages — native object size in both,
    # since holosoma scales the object's position, not its geometry. The .pt pose is
    # [qw,qx,qy,qz,x,y,z]; the viewer wants MuJoCo order. Object name is the 2nd token
    # (sub3_largebox_003 -> largebox).
    object_mesh_verts = object_mesh_faces = object_points_local = None
    object_pose_raw = object_pose_scaled = None
    object_sdf_pts = object_sdf_cols = None
    object_sdf_floor_pts = object_sdf_floor_cols = None
    if data_format == "smplh":
        # Interaction lengths from the TEST-SOCP config, so the SDF band shells the viewer
        # draws match the bands the solver actually uses (floor + object channels).
        from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
        _sc = (cfg.retargeter if isinstance(cfg.retargeter, TestSocpRetargeterConfig)
               else TestSocpRetargeterConfig())
        _L_flr_view = _sc.L_floor if _sc.L_floor is not None else _sc.L_interaction
        _L_view = _sc.L_object if _sc.L_object is not None else _sc.L_interaction
        # Analytic floor SDF band: a thin sheet stack over the floor extent at z layers in
        # [-L_floor, L_floor], coloured by signed z distance (drawn by the "SDF Floor" toggle).
        from HoloNew.src.test_socp.contact.probes import make_floor_grid
        _cx = float(raw_joints[:, 0, 0].mean())
        _cy = float(raw_joints[:, 0, 1].mean())
        _base = make_floor_grid(center_xy=(_cx, _cy), density=80.0)
        _layers = []
        for _z in np.linspace(-_L_flr_view, _L_flr_view, 3):
            _l = _base.copy()
            _l[:, 2] = _z
            _layers.append(_l)
        object_sdf_floor_pts = np.concatenate(_layers).astype(np.float32)
        object_sdf_floor_cols = signed_distance_colors(object_sdf_floor_pts[:, 2], _L_flr_view)

        parts = cfg.task_name.split("_")
        obj_file = Path("models") / parts[1] / f"{parts[1]}.obj" if len(parts) >= 2 else None
        try:
            _, obj_poses = load_intermimic_data(str(cfg.data_path / f"{cfg.task_name}.pt"))
            if obj_file is not None and obj_file.exists():
                object_pose_raw = convert_object_poses_to_mujoco_order(obj_poses)
                object_pose_scaled = convert_object_poses_to_mujoco_order(
                    scale_object_poses_to_center(obj_poses, smpl_scale))
                mesh = trimesh.load(str(obj_file), force="mesh", process=False)
                object_mesh_verts = np.asarray(mesh.vertices, np.float32)
                object_mesh_faces = np.asarray(mesh.faces, np.uint32)
                # Native-size local samples (same sampling as the solve: sample_count=100,
                # seed=42); placed at the active stage's pose, native size on every stage.
                object_points_local, _ = load_object_data(
                    str(obj_file), smpl_scale=smpl_scale, sample_count=100)

                # Object SDF for the "SDF Object" band-shell viz. Uses the SAME keyed,
                # disk-cached field as the solve (load_or_build_object_sdf), resolved at the
                # run's interaction length (_L_view from the config above), so the viewer
                # draws the band the solver actually uses — no fixed-band file to drift.
                _cache = object_sdf_cache_path(str(obj_file), _L_view, _sc.sdf_resolution,
                                               "assets/contact")
                _existed = _cache.exists()
                sdf = load_or_build_object_sdf(str(obj_file), _L_view, _sc.sdf_resolution,
                                               cache_dir="assets/contact")
                logger.info("Object SDF %s: %s  (L=%.3f m, res=%.3f m, %d nodes)",
                            "loaded" if _existed else "built+cached", _cache.name,
                            _L_view, _sc.sdf_resolution, int(np.prod(sdf.dims)))
                band_pts, band_dist = band_points(sdf, _L_view)
                object_sdf_pts = band_pts
                object_sdf_cols = signed_distance_colors(band_dist, _L_view)
            else:
                logger.warning("No object mesh at %s; object disabled.", obj_file)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Object unavailable (%s); object disabled.", exc)

    toe = [constants.DEMO_JOINTS.index(name) for name in cfg.motion_data_config.toe_names]
    # JOINTS_MAPPING is a dict {human_joint_name -> robot_link_name}; the mapped
    # indices select the human-side joints out of the 52-joint DEMO_JOINTS array.
    mapped = [constants.DEMO_JOINTS.index(name) for name in constants.JOINTS_MAPPING]

    # Bone topologies + the solved-robot link set, shared by every method. The
    # mapped GMR stages and the holosoma Mapped stage carry different joint sets,
    # so each gets its own bones; the robot stage reads g1 link world positions
    # (FK from qpos) drawn with the GMR mapped-body topology.
    gmr_bones = skeleton.bones_for_subset(list(HUMAN_BODY_TO_IDX.values()))
    holo_mapped_bones = skeleton.bones_for_subset(mapped)
    robot_bodies = [_BODY_NAME_REMAP.get(f, f) for f in IK_MATCH_TABLE1]
    robot_mjcf = cfg.robot_config.ROBOT_URDF_FILE.replace(".urdf", ".xml")
    # Holosoma's morphological-graph keypoints: the robot links its Laplacian interaction
    # mesh matches (task_constants.JOINTS_MAPPING values), read by FK from each solved qpos.
    g1_links = list(constants.JOINTS_MAPPING.values())

    def _method_object_pose(key: str):
        """The object pose THIS method places on its scaled stages: the raw object pose
        scaled by the method's own object knobs, resolved exactly like the builders
        (XY: None -> smpl_scale, Z: None -> 1.0). So TEST-SOCP (scale_*_object=1.0) shows
        the RAW object and GMR-SOCP the smpl_scale-centred one. Position is a pure
        multiply, so it is applied directly to the MuJoCo-order [x,y,z, qw..] raw pose -
        decoupled from the solve-only rt._obj_poses_mj (None unless non-penetration is
        on). None when there is no object (viewer falls back to its global scaled pose)."""
        if object_pose_raw is None:
            return None
        from HoloNew.src.gmr_socp.config import GmrSocpRetargeterConfig
        from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
        cfg_cls = TestSocpRetargeterConfig if key == "test_socp" else GmrSocpRetargeterConfig
        sc = cfg.retargeter if isinstance(cfg.retargeter, cfg_cls) else cfg_cls()
        ox = sc.scale_xy_object if sc.scale_xy_object is not None else smpl_scale
        oz = sc.scale_z_object if sc.scale_z_object is not None else 1.0
        placed = object_pose_raw.copy()
        placed[:, 0:2] *= ox   # x, y
        placed[:, 2] *= oz     # z
        return placed

    def build_holosoma() -> MethodViz:
        # holosoma: native solved qpos + reproduced preprocessing stages.
        native = run_headless(cfg=cfg)
        hs = compute_holosoma_stages(raw_joints, smpl_scale, toe, mapped)
        rs, rq = robot_link_poses(robot_mjcf, robot_bodies, native.qpos)
        g1 = robot_link_poses(robot_mjcf, g1_links, native.qpos)[0]
        sb = {"Mapped": holo_mapped_bones, ROBOT_STAGE: gmr_bones}
        return MethodViz("holosoma", "holosoma", native.qpos, hs,
                         stage_bones=sb, robot_skeleton=rs, robot_quats=rq, g1_points=g1)

    def build_gmr(label: str, key: str, cls) -> MethodViz:
        # GMR v1 / v2: solved qpos + full per-stage mapped-body point clouds.
        rt = cls.from_config(cfg)
        # TEST-SOCP can emit CoM / angular-momentum / foot-slip diagnostics for the
        # viewer (no-op for GMR-SOCP, which lacks the flag).
        if hasattr(rt, "collect_diagnostics"):
            rt.collect_diagnostics = True
        res = rt.retarget(max_frames=cfg.max_frames)
        T = res.qpos.shape[0]
        # GMR's own floor correction is labelled "Floor" so the early input-grounding
        # stage can use holosoma's "Grounded" name without collision.
        gmr_labels = {"mapped": "Mapped", "scaled": "Scaled", "offset": "Offset", "ground": "Floor"}
        stages = {gmr_labels[name]: rt.gmr_stages[name]["pos"][:T]
                  for name in ("mapped", "scaled", "offset", "ground")}
        # Per-joint orientations of the mapped bodies, so their joint frames can be drawn.
        sq = {gmr_labels[name]: rt.gmr_stages[name]["quat"][:T]
              for name in ("mapped", "scaled", "offset", "ground")}
        # The 52-joint raw source skeleton, then the exact grounded input the GMR chain
        # consumed (rt.gmr_grounded), so Mapped/Scaled/Offset/Floor stay consistent with it.
        stages = {"Original": raw_joints[:T, :, :], "Grounded": rt.gmr_grounded[:T], **stages}
        rs, rq = robot_link_poses(robot_mjcf, robot_bodies, res.qpos)
        g1 = robot_link_poses(robot_mjcf, g1_links, res.qpos)[0]
        sb = {name: gmr_bones for name in ("Mapped", "Scaled", "Offset", "Floor")}
        sb[ROBOT_STAGE] = gmr_bones
        mv = MethodViz(label, key, res.qpos, stages, stage_bones=sb,
                       robot_skeleton=rs, stage_quats=sq, robot_quats=rq, g1_points=g1)
        # TEST-SOCP exposes the per-frame interaction data (probes at the Grounded pose,
        # their object/floor signed distances + object witness, and the correspondence
        # transported onto the solved robot). Map it onto the MethodViz for the viewer.
        if res.human_probe_pts is not None:
            mv.human_probe_pts = res.human_probe_pts
            mv.human_obj_dist = res.human_obj_dist
            mv.human_flr_dist = res.human_flr_dist
            mv.human_witness = res.human_witness
            mv.human_flr_witness = res.human_flr_witness
            mv.human_dist = np.minimum(res.human_obj_dist, res.human_flr_dist)
        if res.g1_transport_pts is not None:
            mv.g1_transport_pts = res.g1_transport_pts
            mv.g1_obj_dist = res.human_obj_dist[:, res.human_idx]  # each G1 reads its human's object dist
            mv.g1_obj_witness = res.human_witness[:, res.human_idx]  # and its human's object witness
            mv.g1_flr_dist = res.human_flr_dist[:, res.human_idx]  # and its human's floor distance
            mv.g1_flr_witness = res.human_flr_witness[:, res.human_idx]  # world-frame floor witness
            # Contact-cloud colour = strongest proximity across both channels (mirrors
            # human_dist), so a G1 point near the floor colours from the floor channel.
            mv.g1_dist = np.minimum(res.human_obj_dist, res.human_flr_dist)[:, res.human_idx]
        # Object-as-carrier surface samples (object<->floor channel); lifted per frame
        # by the solved/reference object pose in the viewer.
        mv.object_surface_local = res.object_surface_local
        # Object placement for THIS method on its scaled stages (GMR centres it by
        # smpl_scale, TEST keeps it raw). Derived from the raw object pose + the method's
        # object knobs, so it is correct even when the solve-only _obj_poses_mj is None.
        mv.object_pose_scaled = _method_object_pose(key)
        # Solve diagnostics: the method's SOLVED object pose + CoM / momentum / slip.
        mv.solved_object_poses = res.solved_object_poses
        mv.com = res.com
        mv.com_ref = res.com_ref
        mv.angular_momentum = res.angular_momentum
        mv.angular_momentum_ref = res.angular_momentum_ref
        mv.foot_slip = res.foot_slip
        return mv

    builders = {
        "holosoma": build_holosoma,
        "gmr_socp": lambda: build_gmr("GMR-SOCP", "gmr_socp", GmrSocpRetargeter),
        "test_socp": lambda: build_gmr("TEST-SOCP", "test_socp", TestSocpRetargeter),
    }
    methods = [builders[name]() for name in cfg.methods]

    T = min(m.qpos.shape[0] for m in methods)
    for m in methods:
        m.stages["Original"] = raw_joints[:T, :, :]

    keys = tuple(m.robot_key for m in methods)
    # Stages whose skeleton lives in the placed (scaled) world: the object follows the
    # active method's own placement there, and stays at its raw pose on the unscaled
    # stages. 'Mapped' is now the raw pre-scale bodies (commit ccadf61), so it is raw
    # too — only Scaled/Offset/Floor/Robot carry the placement. Native size on every stage.
    object_scaled_stages = ("Scaled", "Offset", "Floor", ROBOT_STAGE)
    viewer = Viewer(
        robot_model_path=cfg.robot_config.ROBOT_URDF_FILE,
        object_model_path=None,
        stage_keys=keys,
        original_joints=raw_joints[:T, :, :],
        original_quats=None if original_quats is None else original_quats[:T],
        object_mesh_verts=object_mesh_verts,
        object_mesh_faces=object_mesh_faces,
        object_points_local=object_points_local,
        object_pose_raw=None if object_pose_raw is None else object_pose_raw[:T],
        object_pose_scaled=None if object_pose_scaled is None else object_pose_scaled[:T],
        object_scaled_stages=object_scaled_stages,
        object_sdf_pts=object_sdf_pts,
        object_sdf_cols=object_sdf_cols,
        object_sdf_floor_pts=object_sdf_floor_pts,
        object_sdf_floor_cols=object_sdf_floor_cols,
        human_body=human_body,
    )
    viewer.bind_methods(methods)
    input("Stage viewer at http://localhost:8080 — Enter to exit ...")


if __name__ == "__main__":
    view(tyro.cli(ViewStagesConfig))
