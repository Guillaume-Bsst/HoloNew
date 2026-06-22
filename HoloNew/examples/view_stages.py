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
from HoloNew.src.data_loaders.hodome import extract_hodome_object_mesh, hodome_object_poses
from HoloNew.src.viewer import MethodViz, Viewer

logger = logging.getLogger(__name__)


def grounded_smplx_skeleton(raw_joints, toe_indices, n_frames):
    """22-joint SMPL-X skeleton for the viewer's Grounded stage, grounded over the FULL
    sequence (matching the solve's gmr_grounded) then sliced to n_frames.

    Grounding over only the displayed [:n_frames] window would use a different floor
    minimum than the solve (which grounds the whole clip), shifting the displayed human
    in Z relative to the contact cloud whenever the sequence's lowest foot falls outside
    the window (e.g. --max_frames 20 on a long HODome clip)."""
    from HoloNew.src.holosoma.preprocess import ground_to_floor
    return ground_to_floor(raw_joints, toe_indices)[:n_frames].astype(np.float32)


def _sdf_floor_band(center_xy, l_floor, density=80.0):
    """Floor SDF band for the viewer's 'SDF Floor' overlay: a stack of z-sheets in
    [-l_floor, l_floor] over the floor extent around ``center_xy``, coloured by signed
    z-distance. Returns (points (N,3) float32, colors (N,3)). Shared by the OMOMO and
    HODome paths so both get the band."""
    from HoloNew.src.test_socp.contact.probes import make_floor_grid
    base = make_floor_grid(center_xy=center_xy, density=density)
    layers = []
    for z in np.linspace(-l_floor, l_floor, 3):
        layer = base.copy()
        layer[:, 2] = z
        layers.append(layer)
    pts = np.concatenate(layers).astype(np.float32)
    return pts, signed_distance_colors(pts[:, 2], l_floor)


def _sdf_object_band(mesh_file, l_object, sdf_resolution, cache_dir="assets/contact"):
    """Object SDF band shell (object-LOCAL frame) + colours for the 'SDF Object' overlay,
    from the same keyed, disk-cached field the solve uses (load_or_build_object_sdf). The
    viewer lifts the points by the per-frame object pose. The mesh must already be in the
    frame its poses reference (the OMOMO resolver pre-centres/scales it; HODome's scanned
    mesh is native). Returns (points (M,3) float32, colors (M,3))."""
    sdf = load_or_build_object_sdf(str(mesh_file), l_object, sdf_resolution, cache_dir=cache_dir)
    band_pts, band_dist = band_points(sdf, l_object)
    return band_pts.astype(np.float32), signed_distance_colors(band_dist, l_object)


def _solve_dataset_key(cfg, dataset):
    """The dataset key to re-expose to the GMR/TEST solve's object resolution.

    view() clears cfg.dataset to None so holosoma's run_headless->main loads via the
    normalized legacy fields. But the GMR/TEST from_config object resolver
    (resolve_object_inputs) then falls onto the legacy OBJECT_MESH_FILE + .pt path, which
    can only reach a dataset whose object is encoded in the motion .pt (OMOMO). HODome's
    object lives outside any .pt (scanned tar + a separate poses .npz), so it resolves
    ONLY via the loader's object_source — which needs cfg.dataset set.

    Re-expose the dataset for the solve precisely when there is NO legacy .pt, so HODome
    loads its object like OMOMO does while OMOMO (which HAS a .pt) keeps its float32 legacy
    path untouched — the loader path promotes poses to float64 (omomo.object_source) and
    would drift tight golden tests. Returns the dataset key, or None to keep legacy."""
    if dataset is None:
        return None
    return None if (cfg.data_path / f"{cfg.task_name}.pt").exists() else dataset


def _window_solve_frames(rt, start: int) -> None:
    """Slice every per-frame solve input on a GMR/TEST retargeter to ``[start:]`` so a
    subsequent ``rt.retarget(max_frames=N)`` solves the window ``[start : start+N]`` rather
    than ``[0:N]``. Mirrors the per-frame arrays the retargeters consume: the GMR ground
    targets (gmr_ground / gmr_stages), the grounded SMPL-X joints (gmr_grounded), the
    object reference/qpos poses (_obj_poses_raw / _obj_poses_mj), the probe orientations
    (_smplx_orientations / human_quat) and the foot-sticking sequences.

    ``rt.gmr_ground`` is the SAME dict object as ``rt.gmr_stages['ground']`` (builder.py
    aliases them), so stage dicts are sliced once via an id() guard to avoid double
    slicing. Frame-independent state (q_init_full, object_surface_local) is left untouched;
    solve OUTPUTS are produced fresh over the window by retarget()."""
    if start <= 0:
        return
    _seen: set[int] = set()

    def _slice_stage(d) -> None:
        if not isinstance(d, dict) or id(d) in _seen:
            return
        _seen.add(id(d))
        for _k in ("pos", "quat"):
            if d.get(_k) is not None:
                d[_k] = d[_k][start:]

    stages = getattr(rt, "gmr_stages", None)
    if isinstance(stages, dict):
        for _st in stages.values():
            _slice_stage(_st)
    _slice_stage(getattr(rt, "gmr_ground", None))   # usually stages['ground'] -> no-op

    for _name in ("gmr_grounded", "_obj_poses_raw", "_obj_poses_mj",
                  "_smplx_orientations", "human_quat"):
        _v = getattr(rt, _name, None)
        if _v is not None:
            setattr(rt, _name, _v[start:])

    _fss = getattr(rt, "foot_sticking_sequences", None)
    if _fss:
        rt.foot_sticking_sequences = list(_fss)[start:]

    # The contact probe keeps its OWN full-clip copy of the object poses (built at
    # construction), separate from rt._obj_poses_raw sliced above. Window it in lockstep,
    # else the probe queries the windowed human (frame start+t) against the un-windowed
    # object (frame t): with a stale floor-object the nearest human probes become the feet,
    # so the object contact/direction overlays attach to the feet instead of the hands.
    _probe = getattr(rt, "smplx_ground_probe", None)
    if _probe is not None and getattr(_probe, "obj_quat", None) is not None:
        _probe.obj_quat = _probe.obj_quat[start:]
        _probe.obj_trans = _probe.obj_trans[start:]


def save_view_result(*, base_dir, robot, task_type, dataset, task_name, method,
                     qpos, human_joints, cost, fps=30, save_dir=None) -> Path:
    """Persist one GMR/TEST solve to a demo_results .npz in the same format
    robot_retarget/holosoma write (``qpos``, ``human_joints``, ``fps``, ``cost``).
    Destination is ``save_dir`` verbatim when given (mirrors robot_retarget's --save-dir),
    else ``<base_dir>/<robot>/<task_type>/<dataset>/``; the file is ``<task_name>_<method>.npz``
    so the per-method results (and holosoma's own ``<task_name>.npz``) never collide."""
    dest_dir = Path(save_dir) if save_dir is not None else Path(base_dir) / robot / task_type / (dataset or "misc")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{task_name}_{method}.npz"
    np.savez(dest, qpos=np.asarray(qpos), human_joints=np.asarray(human_joints),
             fps=fps, cost=cost)
    return dest


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
    # Skip the first ``start_frame`` frames before solving/displaying, so an interaction
    # that begins late in the clip (e.g. HODome subject01_baseball: the bat rests on the
    # floor until ~frame 270, when the subject picks it up) can be inspected without
    # solving the leading frames. With --max_frames it selects the window
    # [start_frame : start_frame + max_frames]. 0 = start at the first frame. Honoured by
    # the GMR-SOCP / TEST-SOCP methods; the holosoma method front-caps its solve in
    # robot_retarget.main and does NOT support it, so start_frame > 0 with --methods
    # holosoma is rejected.
    start_frame: int = 0
    # Persist each GMR-SOCP / TEST-SOCP solve as a demo_results .npz (qpos, human_joints,
    # fps, cost) — the same format robot_retarget writes — so the run leaves a reusable
    # result, not just an on-screen render. Saved to <save_dir> if given, else
    # demo_results/<robot>/<task_type>/<dataset>/<task_name>_<method>.npz. Pass --no-save
    # to view only. (holosoma always saves via its own run_headless path.)
    save: bool = True


def view(cfg: ViewStagesConfig) -> None:
    if not cfg.methods:
        raise ValueError("--methods must select at least one optimizer")
    if cfg.start_frame and "holosoma" in cfg.methods:
        raise ValueError(
            "--start-frame is not supported with the 'holosoma' method (its solve "
            "front-caps frames in robot_retarget.main). Use --methods gmr_socp and/or "
            "test_socp, or leave start_frame at 0.")

    # 3-path façade: when --dataset is set, translate model/motion/obj paths into the
    # legacy data_path/task_name/data_format (+ omomo_dir) fields, then clear `dataset`
    # so every method below (holosoma via run_headless->main, GMR/TEST via from_config)
    # loads uniformly through those normalized fields. `dataset` is kept locally so the
    # object overlay below knows how to resolve the object mesh / poses.
    from HoloNew.src.data_loaders.facade import normalize_dataset_cfg
    normalize_dataset_cfg(cfg)        # also lower-cases cfg.dataset to the canonical key
    dataset = cfg.dataset
    cfg.dataset = None

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

    # Frame window: skip the first ``start`` frames. raw_joints stays FULL here because
    # some display grounding (grounded_smplx_skeleton) needs the whole-clip floor min;
    # windowed views (raw_joints_win / original_quats_win below), the HODome mesh poser,
    # the object poses and the retargeter's per-frame inputs (_window_solve_frames) are
    # each offset by ``start`` so local frame 0 maps to global frame ``start``.
    start = max(0, int(cfg.start_frame or 0))
    if start >= raw_joints.shape[0]:
        raise ValueError(
            f"--start-frame {start} is at/after the clip length {raw_joints.shape[0]}.")

    # Per-joint orientations drive the SMPL-X mesh. The smplh .pt path carries the 52
    # MuJoCo-order quats; the smplx path carries the 22 SMPL-order global orientations
    # (from the processed npz), posed via placed_verts_smpl. Without them the viewer
    # degrades to skeleton-only.
    original_quats = None
    smpl_order = data_format == "smplx"
    smplx_betas, smplx_gender = None, "neutral"
    if data_format == "smplh":
        try:
            original_quats = load_intermimic_quats(str(cfg.data_path / f"{cfg.task_name}.pt"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("No per-joint quats (%s); SMPL-X mesh disabled.", exc)
    elif data_format == "smplx":
        try:
            _npz = np.load(str(cfg.data_path / f"{cfg.task_name}.npz"))
            original_quats = np.asarray(_npz["global_joint_orientations"], dtype=np.float32)
            if "betas" in _npz.files:
                smplx_betas = np.asarray(_npz["betas"], dtype=np.float32)
            if "gender" in _npz.files:
                smplx_gender = str(_npz["gender"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("No SMPL-X orientations (%s); SMPL-X mesh disabled.", exc)

    # Windowed display views (local frame 0 == global ``start``); raw_joints / original_quats
    # are kept full above for whole-clip grounding, these are what the viewer/Original stage use.
    raw_joints_win = raw_joints[start:]
    original_quats_win = None if original_quats is None else original_quats[start:]

    # SMPL-X shape (betas) + gender for this subject, read from the original OMOMO
    # .p files keyed by sequence name. Without them the mesh uses the neutral mean
    # shape. The .pt motion (OMOMO_new) does not carry betas, hence the separate dir.
    # smplx clips carry their own betas in the npz, so this OMOMO lookup is smplh-only.
    betas, gender = None, "neutral"
    if cfg.omomo_dir is not None and data_format == "smplh":
        betas, gender = load_human_metadata(cfg.omomo_dir, cfg.task_name)
        if betas is not None:
            logger.info("Loaded SMPL-X shape for %s: gender=%s, %d betas",
                        cfg.task_name, gender, betas.shape[0])
        else:
            logger.warning("No SMPL-X shape found for %s in %s; using neutral shape.",
                           cfg.task_name, cfg.omomo_dir)

    # smplx subjects carry their own betas/gender in the npz; smplh reads them from the
    # OMOMO metadata loaded above.
    _mesh_betas = smplx_betas if data_format == "smplx" else betas
    _mesh_gender = smplx_gender if data_format == "smplx" else gender
    # HODome poses the mesh from a native SMPL-X forward + Y->Z vertex swap (its raw npz
    # at cfg.motion_path carries the full pose params); the orientation-conjugation path
    # collapses the body. Other sources keep the HumanBody orientation/quat posing.
    human_body = None
    human_mesh_poser = None
    if dataset == "hodome":
        try:
            from HoloNew.src.data_loaders.hodome import HodomeMeshPoser
            human_mesh_poser = HodomeMeshPoser(Path(cfg.motion_path), Path(cfg.model_path))
        except Exception as exc:  # noqa: BLE001
            logger.warning("HODome mesh poser unavailable (%s); mesh disabled.", exc)
    elif original_quats is not None:
        try:
            human_body = HumanBody(SMPLX_MODEL_DIR_DEFAULT, _mesh_betas, _mesh_gender)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SMPL-X unavailable (%s); mesh disabled.", exc)

    # Re-base the HODome mesh poser onto the window: the viewer indexes it by the local
    # frame, so slicing its raw per-frame params makes local frame 0 == global ``start``,
    # matching raw_joints_win / original_quats_win passed to the viewer.
    if start and human_mesh_poser is not None:
        for _k in list(human_mesh_poser._params):
            human_mesh_poser._params[_k] = human_mesh_poser._params[_k][start:]
        human_mesh_poser._cache_idx = -1

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
    # Interaction lengths from the TEST-SOCP config, so the SDF band shells the viewer
    # draws match the bands the solver actually uses (floor + object channels). Shared by
    # the OMOMO (smplh) and HODome (smplx) object paths below; the floor band is centred
    # on the human's pelvis track.
    from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
    _sc = (cfg.retargeter if isinstance(cfg.retargeter, TestSocpRetargeterConfig)
           else TestSocpRetargeterConfig())
    _L_flr_view = _sc.L_floor if _sc.L_floor is not None else _sc.L_interaction
    _L_view = _sc.L_object if _sc.L_object is not None else _sc.L_interaction
    _floor_xy = (float(raw_joints[:, 0, 0].mean()), float(raw_joints[:, 0, 1].mean()))
    if data_format == "smplh":
        # Floor SDF band (drawn by the "SDF Floor" toggle).
        object_sdf_floor_pts, object_sdf_floor_cols = _sdf_floor_band(_floor_xy, _L_flr_view)

        # Object mesh via the shared resolver: bundled (centred + pre-scaled) first, else
        # the captured unit mesh recentred + scaled by obj_scale (same logic the solver
        # loader uses). It returns an already-transformed mesh, so no further
        # recenter/scale is applied here (obj_geom_center = 0, obj_geom_scale = 1).
        from HoloNew.src.data_loaders.omomo import resolve_omomo_object_mesh
        try:
            obj_file = resolve_omomo_object_mesh(cfg.task_name, cfg.omomo_dir)
        except ValueError:
            obj_file = None
        obj_geom_scale, obj_recenter = 1.0, False
        try:
            _, obj_poses = load_intermimic_data(str(cfg.data_path / f"{cfg.task_name}.pt"))
            if obj_file is not None and obj_file.exists():
                object_pose_raw = convert_object_poses_to_mujoco_order(obj_poses)
                object_pose_scaled = convert_object_poses_to_mujoco_order(
                    scale_object_poses_to_center(obj_poses, smpl_scale))
                mesh = trimesh.load(str(obj_file), force="mesh", process=False)
                _verts = np.asarray(mesh.vertices, np.float32)
                # Recentre the off-origin captured mesh onto its centroid (no-op for the
                # already-centred bundled mesh), then resize to the real object size.
                obj_geom_center = _verts.mean(0) if obj_recenter else np.zeros(3, np.float32)
                object_mesh_verts = (_verts - obj_geom_center) * obj_geom_scale
                object_mesh_faces = np.asarray(mesh.faces, np.uint32)
                # Native-size local samples via the solver's sample_object_surface
                # (density-based, same sampler the movable term uses); placed at the active
                # stage's pose, native size on every stage.
                from HoloNew.src.test_socp.movable import sample_object_surface
                object_points_local = sample_object_surface(str(obj_file)).astype(np.float32)
                object_points_local = (object_points_local - obj_geom_center) * obj_geom_scale

                # Object SDF band shell ("SDF Object" toggle) from the same keyed, disk-
                # cached field the solve uses; the resolver pre-transforms the mesh, so the
                # band is already in the object-local frame the poses reference.
                object_sdf_pts, object_sdf_cols = _sdf_object_band(
                    obj_file, _L_view, _sc.sdf_resolution)
            else:
                logger.warning("No object mesh at %s; object disabled.", obj_file)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Object unavailable (%s); object disabled.", exc)

    elif dataset == "hodome" and cfg.obj_path is not None:
        # HODome: single object per sequence. Mesh comes from scaned_object/<token>.tar,
        # poses from the object .npz (object_R/T), both expressed Z-up like the human.
        try:
            token = cfg.task_name.split("_", 1)[1] if "_" in cfg.task_name else cfg.task_name
            scaned = Path(cfg.obj_path).parent.parent / "scaned_object"
            mesh_path = extract_hodome_object_mesh(token, scaned)
            obj_poses = hodome_object_poses(Path(cfg.obj_path))           # (T,7) [qw..,xyz] Z-up
            object_pose_raw = convert_object_poses_to_mujoco_order(obj_poses)
            object_pose_scaled = convert_object_poses_to_mujoco_order(
                scale_object_poses_to_center(obj_poses, smpl_scale))
            mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
            # NeuralDome defines object_R/T relative to the CENTROID-CENTRED mesh: the
            # toolbox does `obj_verts -= obj_verts.mean(0)` before `verts @ R.T + T`
            # (scripts/hodome_visualize_pyrender.py). The scanned .obj origin is arbitrary
            # (e.g. at the bat knob, ~0.30 m off-centroid), so without recentring the object
            # is offset by its centroid along the pose — it sits off the body / penetrates it.
            obj_center = np.asarray(mesh.vertices, np.float64).mean(0).astype(np.float32)
            object_mesh_verts = np.asarray(mesh.vertices, np.float32) - obj_center
            object_mesh_faces = np.asarray(mesh.faces, np.uint32)
            from HoloNew.src.test_socp.movable import sample_object_surface
            object_points_local = sample_object_surface(str(mesh_path)).astype(np.float32) - obj_center
            # SDF band shells ("SDF Floor" / "SDF Object" toggles) — same helpers the OMOMO
            # path uses. The band points live in the raw mesh frame, so recentre them by the
            # same centroid as the mesh/samples above so the shell wraps the placed object.
            object_sdf_floor_pts, object_sdf_floor_cols = _sdf_floor_band(_floor_xy, _L_flr_view)
            object_sdf_pts, object_sdf_cols = _sdf_object_band(
                mesh_path, _L_view, _sc.sdf_resolution)
            if object_sdf_pts is not None:
                object_sdf_pts = np.asarray(object_sdf_pts, np.float32) - obj_center
        except Exception as exc:  # noqa: BLE001
            logger.warning("HODome object unavailable (%s); object disabled.", exc)

    # Window the object reference poses to the displayed frame window. The mesh, surface
    # samples and SDF live in the object's local (frame-independent) frame, so only the
    # per-frame poses shift; _method_object_pose reads object_pose_raw, so do this before
    # the builders run.
    if start:
        if object_pose_raw is not None:
            object_pose_raw = object_pose_raw[start:]
        if object_pose_scaled is not None:
            object_pose_scaled = object_pose_scaled[start:]

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
        # Re-expose the dataset to the solve's object resolver for datasets with no legacy
        # .pt (HODome), so the object (SDF + surface samples -> object_surface_local and the
        # 'Object->Floor' overlays) loads via the loader, matching OMOMO. cfg.dataset is None
        # here (view() cleared it for holosoma's uniform legacy load); restore it afterwards.
        _saved_dataset = cfg.dataset
        cfg.dataset = _solve_dataset_key(cfg, dataset)
        try:
            rt = cls.from_config(cfg)
        finally:
            cfg.dataset = _saved_dataset
        # Offset the per-frame solve inputs by start_frame, so retarget(max_frames=N)
        # solves the window [start : start+N] instead of [0:N] (no-op when start == 0).
        _window_solve_frames(rt, start)
        # TEST-SOCP can emit CoM / angular-momentum / foot-slip diagnostics for the
        # viewer (no-op for GMR-SOCP, which lacks the flag).
        if hasattr(rt, "collect_diagnostics"):
            rt.collect_diagnostics = True
        res = rt.retarget(max_frames=cfg.max_frames)
        T = res.qpos.shape[0]
        # Persist the solve before building the (display-only) viz, so the result lands on
        # disk even if the viewer window is closed immediately. human_joints are the raw
        # input joints for the solved window [start : start+T], frame-aligned with qpos.
        if cfg.save:
            # GMR/TEST leave res.cost at 0; the per-frame SQP objective (per_frame_cost)
            # is the analogous solve-quality scalar, so save its mean when res.cost is unset.
            _cost = float(res.cost)
            if not _cost and res.per_frame_cost is not None:
                _cost = float(np.mean(res.per_frame_cost))
            _dest = save_view_result(
                base_dir="demo_results", robot=cfg.robot, task_type=cfg.task_type,
                dataset=dataset, task_name=cfg.task_name, method=key,
                qpos=res.qpos, human_joints=raw_joints_win[:T], cost=_cost,
                save_dir=cfg.save_dir)
            logger.info("Saved %s result (%d frames, cost=%.4g) -> %s", key, T, _cost, _dest)
        # GMR's own floor correction is labelled "Floor" so the early input-grounding
        # stage can use holosoma's "Grounded" name without collision.
        gmr_labels = {"mapped": "Mapped", "scaled": "Scaled", "offset": "Offset", "ground": "Floor"}
        stages = {gmr_labels[name]: rt.gmr_stages[name]["pos"][:T]
                  for name in ("mapped", "scaled", "offset", "ground")}
        # Per-joint orientations of the mapped bodies, so their joint frames can be drawn.
        sq = {gmr_labels[name]: rt.gmr_stages[name]["quat"][:T]
              for name in ("mapped", "scaled", "offset", "ground")}
        # The raw source skeleton, then the grounded input the GMR chain consumed. For
        # smplh raw_joints is the 52-joint layout and rt.gmr_grounded matches it. For smplx
        # the source is the 22 SMPL-X joints, but rt.gmr_grounded is the SMPLH 52-slot array
        # with only the 14 mapped slots filled (the rest at the origin) — drawing that as a
        # full skeleton scatters stray points at (0,0,0). So re-ground the same 22 joints
        # the Original stage shows, drawn with the SMPL-X topology, for a consistent skeleton.
        if data_format == "smplx":
            from HoloNew.src.test_socp.targets import SMPLX_BODY_JOINT_NAMES
            _toe = [SMPLX_BODY_JOINT_NAMES.index("left_foot"),
                    SMPLX_BODY_JOINT_NAMES.index("right_foot")]
            # Ground over the FULL sequence (matching rt.gmr_grounded used by the contact
            # probe), then slice to the window — NOT ground_to_floor(raw_joints[:T]) which
            # would use the window's floor min and Z-shift the displayed human off the
            # contact cloud. start+T then [start:] keeps the whole-clip floor min.
            grounded = grounded_smplx_skeleton(raw_joints, _toe, start + T)[start:].copy()
            # The solve also drops gmr_grounded by the floor correction (median sole +
            # contact margin); apply the same drop so the displayed human stays on the
            # contact cloud (both rest on the floor). smplh already uses gmr_grounded.
            grounded[:, :, 2] -= float(getattr(rt, "_floor_offset", 0.0))
        else:
            grounded = rt.gmr_grounded[:T]
        stages = {"Original": raw_joints_win[:T, :, :], "Grounded": grounded, **stages}
        rs, rq = robot_link_poses(robot_mjcf, robot_bodies, res.qpos)
        g1 = robot_link_poses(robot_mjcf, g1_links, res.qpos)[0]
        sb = {name: gmr_bones for name in ("Mapped", "Scaled", "Offset", "Floor")}
        sb[ROBOT_STAGE] = gmr_bones
        if data_format == "smplx":
            sb["Grounded"] = skeleton.SMPLX_BODY_BONES
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
        # Ground the displayed object on the scaled/robot stages by the SAME shift the solve
        # applied (rt._obj_ground_shift = object's own floor -> z=0), so the viewer matches
        # the result (object on the z=0 floor with the robot). MuJoCo order -> Z is index 2.
        if mv.object_pose_scaled is not None:
            mv.object_pose_scaled[:, 2] += getattr(rt, "_obj_ground_shift", 0.0)
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
        m.stages["Original"] = raw_joints_win[:T, :, :]

    keys = tuple(m.robot_key for m in methods)
    # Stages whose skeleton lives in the placed (scaled) world: the object follows the
    # active method's own placement there, and stays at its raw pose on the unscaled
    # stages. 'Mapped' is now the raw pre-scale bodies (commit ccadf61), so it is raw
    # too — only Scaled/Offset/Floor/Robot carry the placement. Native size on every stage.
    object_scaled_stages = ("Scaled", "Offset", "Floor", ROBOT_STAGE)
    # Interaction bands for the viewer overlays' active masks + colour scales, from the
    # TEST-SOCP config (per channel), so the viewer matches what the solver activates.
    from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
    _vsc = (cfg.retargeter if isinstance(cfg.retargeter, TestSocpRetargeterConfig)
            else TestSocpRetargeterConfig())
    _vL_flr = _vsc.L_floor if _vsc.L_floor is not None else _vsc.L_interaction
    _vL_obj = _vsc.L_object if _vsc.L_object is not None else _vsc.L_interaction
    viewer = Viewer(
        robot_model_path=cfg.robot_config.ROBOT_URDF_FILE,
        object_model_path=None,
        stage_keys=keys,
        original_joints=raw_joints_win[:T, :, :],
        original_quats=None if original_quats_win is None else original_quats_win[:T],
        # smplx sources carry only the 22 SMPL-X body joints (no fingers), so the
        # Original skeleton uses the SMPL-X topology instead of the 52-joint SMPLH one.
        original_bones=skeleton.SMPLX_BODY_BONES if data_format == "smplx" else None,
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
        interaction_L_floor=_vL_flr,
        interaction_L_object=_vL_obj,
        human_body=human_body,
        human_smpl_order=smpl_order,
        human_mesh_poser=human_mesh_poser,
    )
    viewer.bind_methods(methods)
    input("Stage viewer at http://localhost:8080 — Enter to exit ...")


if __name__ == "__main__":
    view(tyro.cli(ViewStagesConfig))
